"""Group-id key fallback in the QQ Open Platform event converter.

``_convert_event`` used to read only ``group_id``. The channel's group
identifier is in fact an openid, and the v2 payload may carry it under
``group_openid`` instead; when that happens ``group_id`` is the empty
string, the backlog writer drops the message outright, and the group
subject degrades to ``qq::`` which the scope validator rejects.

The fallback order is the load-bearing part: ``group_id`` must keep
winning when present. Flipping it would re-key every group subject_id,
and scope matching is byte equality with no aliasing, so all existing
scoped group memories would be orphaned in one shot.

The ``author`` blocks below carry the real protocol keys. They used to say
``{"id": ...}``, which is the very assumption that turned out to be wrong --
see ``test_qq_open_platform_actor_identity.py`` for where the speaker id
actually comes from.
"""

import pytest

from utils.connection.qq.open_platform import QQOpenPlatformConnection


def _connection() -> QQOpenPlatformConnection:
    conn = QQOpenPlatformConnection.__new__(QQOpenPlatformConnection)
    conn._self_id = "bot-self"
    return conn


def _group_payload(**group_keys) -> dict:
    payload = {
        "id": "msg-1",
        "content": "<@!bot-self> zaima",
        "author": {"member_openid": "user-1"},
    }
    payload.update(group_keys)
    return payload


@pytest.mark.parametrize(
    ("group_keys", "expected"),
    [
        # Today's protocol: the original key is present and still wins.
        ({"group_id": "G-plain"}, "G-plain"),
        # v2 semantics: only the openid key is sent.
        ({"group_openid": "G-openid"}, "G-openid"),
        # Both present — pins the priority so a later refactor can't flip it.
        ({"group_id": "G-plain", "group_openid": "G-openid"}, "G-plain"),
        # Present but empty is the same as absent for the fallback.
        ({"group_id": "", "group_openid": "G-openid"}, "G-openid"),
    ],
)
def test_convert_event_group_id_prefers_group_id_then_group_openid(group_keys, expected):
    msg = _connection()._convert_event("GROUP_AT_MESSAGE_CREATE", _group_payload(**group_keys))

    assert msg is not None
    assert msg["message_type"] == "group"
    assert msg["group_id"] == expected


def test_convert_event_group_id_empty_when_neither_key_present():
    # Neither key → empty string, i.e. the pre-existing discard path
    # (backlog_service returns early on a falsy group_id). The fallback
    # must not invent an id out of some other field.
    msg = _connection()._convert_event("GROUP_AT_MESSAGE_CREATE", _group_payload())

    assert msg is not None
    assert msg["group_id"] == ""


def test_convert_event_private_message_keeps_group_id_empty():
    # The C2C branch has no group at all; a stray group key on the payload
    # must not leak into the private message's group_id.
    msg = _connection()._convert_event(
        "C2C_MESSAGE_CREATE",
        {
            "id": "msg-2",
            "content": "hi",
            "author": {"user_openid": "user-1"},
            "group_openid": "G-openid",
        },
    )

    assert msg is not None
    assert msg["message_type"] == "private"
    assert msg["group_id"] == ""


def test_open_platform_group_mentions_distinguish_bot_from_other_users():
    conn = QQOpenPlatformConnection.__new__(QQOpenPlatformConnection)
    conn._self_id = "10000"

    bot_only = conn._convert_event(
        "GROUP_AT_MESSAGE_CREATE",
        {
            "id": "m1",
            "group_id": "g1",
            "author": {"id": "u1", "username": "Alice"},
            "content": "<@!10000> hello",
        },
    )
    assert bot_only["mentioned_user_ids"] == ["10000"]
    assert bot_only["mentions_other_user"] is False

    with_other_user = conn._convert_event(
        "GROUP_AT_MESSAGE_CREATE",
        {
            "id": "m2",
            "group_id": "g1",
            "author": {"id": "u1", "username": "Alice"},
            "content": "<@!10000> <@!20000> hello",
        },
    )
    assert with_other_user["mentioned_user_ids"] == ["10000", "20000"]
    assert with_other_user["mentions_other_user"] is True


# ── step 1 of the host-channel design: port #3706 onto #3709 ───────────────
#
# Three things the single-slot version could not do: honour the platform's own
# ``is_you`` marker, keep one group's bot openid away from another group, and
# refuse to deliver the same message twice. All three are @-attribution, and all
# three failed silently: the bot either ignored a real @ or answered one message
# twice.

def _bare_connection() -> QQOpenPlatformConnection:
    conn = QQOpenPlatformConnection.__new__(QQOpenPlatformConnection)
    conn._self_id = ""
    conn.logger = None
    return conn


def test_is_you_marks_the_bot_even_when_another_member_is_named_too():
    conn = _bare_connection()
    msg = conn._convert_event(
        "GROUP_MESSAGE_CREATE",
        {
            "id": "m", "group_openid": "G1",
            "content": "<@0123456789ABCDEF> 和 <@FEDCBA9876543210> 看一下",
            "author": {"member_openid": "speaker"},
            "mentions": [
                {"member_openid": "0123456789ABCDEF", "is_you": True, "bot": True},
                {"member_openid": "FEDCBA9876543210"},
            ],
        },
    )
    assert msg["is_at_bot"] is True
    # The other openid is a second target, so this is not "only me".
    assert msg["mentions_other_user"] is True
    # Learning happened, and it landed under this group only.
    assert conn._bot_openid_by_group == {"G1": "0123456789ABCDEF"}


def test_a_flag_for_another_bot_does_not_summon_this_one():
    # ``bot: true`` says the target is *some* bot. In a group with two bots, @-ing
    # the neighbour must not make this bot answer.
    conn = _bare_connection()
    msg = conn._convert_event(
        "GROUP_MESSAGE_CREATE",
        {
            "id": "m2", "group_openid": "G1", "content": "<@AAAAAAAAAAAAAAAA>",
            "author": {"member_openid": "speaker"},
            "mentions": [{"member_openid": "AAAAAAAAAAAAAAAA", "bot": True}],
        },
    )
    assert msg["is_at_bot"] is False
    assert conn._bot_openid_by_group == {}


def test_one_groups_learned_openid_does_not_answer_for_another_group():
    conn = _bare_connection()
    conn._convert_event(
        "GROUP_MESSAGE_CREATE",
        {
            "id": "a", "group_openid": "G1", "content": "hi <@BBBBBBBBBBBBBBBB>",
            "author": {"member_openid": "s"},
            "mentions": [{"member_openid": "BBBBBBBBBBBBBBBB", "is_you": True}],
        },
    )
    inline_only_g2 = conn._convert_event(
        "GROUP_MESSAGE_CREATE",
        {"id": "b", "group_openid": "G2", "content": "hi <@BBBBBBBBBBBBBBBB>",
         "author": {"member_openid": "s"}},
    )
    # G1 learned its own openid; G2 gets a different conversation openid, so the
    # same token there is not proof this bot was summoned.
    assert inline_only_g2["is_at_bot"] is False
    assert conn._remembered_bot_openid("G1") == "BBBBBBBBBBBBBBBB"
    assert conn._remembered_bot_openid("G2") == ""
    # Once G2 sees its own is_you, later inline-only payloads resolve per group.
    conn._convert_event(
        "GROUP_MESSAGE_CREATE",
        {"id": "c", "group_openid": "G2", "content": "hi <@CCCCCCCCCCCCCCCC>",
         "author": {"member_openid": "s"},
         "mentions": [{"member_openid": "CCCCCCCCCCCCCCCC", "is_you": True}]},
    )
    after = conn._convert_event(
        "GROUP_MESSAGE_CREATE",
        {"id": "d", "group_openid": "G2", "content": "hi <@CCCCCCCCCCCCCCCC>",
         "author": {"member_openid": "s"}},
    )
    assert after["is_at_bot"] is True
    assert conn._convert_event(
        "GROUP_MESSAGE_CREATE",
        {"id": "e", "group_openid": "G1", "content": "hi <@CCCCCCCCCCCCCCCC>",
         "author": {"member_openid": "s"}},
    )["is_at_bot"] is False


def test_repeated_inline_at_is_counted_as_one_target():
    conn = _bare_connection()
    conn._learn_bot_openid("G1", "DDDDDDDDDDDDDDDD")
    msg = conn._convert_event(
        "GROUP_MESSAGE_CREATE",
        {"id": "f", "group_openid": "G1", "content": "<@DDDDDDDDDDDDDDDD> <@DDDDDDDDDDDDDDDD>",
         "author": {"member_openid": "s"}},
    )
    # Two occurrences of the same target are one target: otherwise "summoned only
    # me, twice" reads as "named me and someone else".
    assert msg["mentioned_user_ids"] == ["DDDDDDDDDDDDDDDD"]
    assert msg["is_at_bot"] is True
    assert msg["mentions_other_user"] is False


import asyncio


def _queue_connection(size: int) -> QQOpenPlatformConnection:
    conn = QQOpenPlatformConnection.__new__(QQOpenPlatformConnection)
    conn._self_id = ""
    conn.logger = None
    conn._message_queue = asyncio.Queue(maxsize=size)
    return conn


def _msg(message_id: str, *, at_bot: bool) -> dict:
    return {"message_id": message_id, "is_at_bot": at_bot, "content": message_id}


def test_a_redelivered_id_is_queued_once():
    conn = _queue_connection(4)
    assert conn._enqueue_inbound(_msg("m1", at_bot=True)) is None
    assert conn._enqueue_inbound(_msg("m1", at_bot=True)) == "duplicate"
    assert conn._message_queue.qsize() == 1


def test_an_evicted_message_is_forgotten_so_its_redelivery_still_counts():
    conn = _queue_connection(1)
    assert conn._enqueue_inbound(_msg("old", at_bot=False)) is None
    # The @bot arrival pushes "old" out. "old" must stop being remembered: a
    # redelivery of it is then the only remaining chance to process it, and
    # suppressing it while it is pending nowhere would lose it for good.
    assert conn._enqueue_inbound(_msg("new", at_bot=True)) is None
    assert conn._is_duplicate_inbound("old") is False
    # With a free slot, the redelivery lands.
    conn._message_queue.get_nowait()
    assert conn._enqueue_inbound(_msg("old", at_bot=False)) is None
    assert conn._message_queue.get_nowait()["message_id"] == "old"


def test_a_full_queue_of_at_messages_is_not_cleared_by_ordinary_chatter():
    conn = _queue_connection(2)
    assert conn._enqueue_inbound(_msg("summon-1", at_bot=True)) is None
    assert conn._enqueue_inbound(_msg("summon-2", at_bot=True)) is None
    # Every queued message is an @ and the arrival is chatter: refuse the arrival
    # rather than trade a queued summon for it.
    assert conn._enqueue_inbound(_msg("chatter", at_bot=False)) == "queue_full"
    queued = [conn._message_queue.get_nowait()["message_id"] for _ in range(2)]
    assert queued == ["summon-1", "summon-2"]


def test_ordinary_chatter_yields_to_an_incoming_at():
    conn = _queue_connection(2)
    conn._enqueue_inbound(_msg("chatter-1", at_bot=False))
    conn._enqueue_inbound(_msg("chatter-2", at_bot=False))
    assert conn._enqueue_inbound(_msg("summon", at_bot=True)) is None
    queued = [conn._message_queue.get_nowait()["message_id"] for _ in range(2)]
    assert "summon" in queued
    assert "chatter-1" not in queued       # the oldest chatter went first


def test_full_message_first_does_not_swallow_the_later_at_signal():
    # Greptile r4574476364: with the privilege on, GROUP_MESSAGE_CREATE can land
    # first and read is_at_bot=False because this group's bot openid is not known
    # yet. Dropping the AT frame as a plain duplicate then dropped the summon,
    # and the bot never answered a direct @.
    import asyncio

    conn = QQOpenPlatformConnection.__new__(QQOpenPlatformConnection)
    conn._self_id = ""
    conn.logger = None
    conn._message_queue = asyncio.Queue(maxsize=8)
    payload = {
        "id": "summon-1", "group_openid": "G9",
        "content": "喂 <@EEEEEEEEEEEEEEEE>",
        "author": {"member_openid": "MA"},
    }
    at_payload = dict(payload)
    at_payload["mentions"] = [{"member_openid": "EEEEEEEEEEEEEEEE", "is_you": True}]

    async def drive(event_type, body):
        await conn._dispatch_event(event_type, body, "e1")
        await asyncio.sleep(0)

    asyncio.run(drive("GROUP_MESSAGE_CREATE", payload))
    asyncio.run(drive("GROUP_AT_MESSAGE_CREATE", at_payload))

    assert conn._message_queue.qsize() == 1          # still delivered once
    queued = conn._message_queue.get_nowait()
    assert queued["is_at_bot"] is True               # ... and the @ survived
    assert queued["mentions_other_user"] is False    # ... with correct ownership
    assert "EEEEEEEEEEEEEEEE" in queued["mentioned_user_ids"]


def test_at_frame_first_stays_correct_when_the_full_copy_arrives_after():
    import asyncio

    conn = QQOpenPlatformConnection.__new__(QQOpenPlatformConnection)
    conn._self_id = ""
    conn.logger = None
    conn._message_queue = asyncio.Queue(maxsize=8)
    payload = {
        "id": "summon-2", "group_openid": "G9", "content": "喂 <@FFFFFFFFFFFFFFFF>",
        "author": {"member_openid": "MA"},
        "mentions": [{"member_openid": "FFFFFFFFFFFFFFFF", "is_you": True}],
    }

    async def drive(body):
        await conn._dispatch_event("GROUP_AT_MESSAGE_CREATE", body, "e1")
        await asyncio.sleep(0)

    asyncio.run(drive(payload))
    asyncio.run(drive(dict(payload, mentions=None)))
    assert conn._message_queue.qsize() == 1
    assert conn._message_queue.get_nowait()["is_at_bot"] is True


def test_full_mode_and_at_event_for_one_message_deliver_once():
    # With the group privilege on, the platform can push both
    # GROUP_AT_MESSAGE_CREATE and GROUP_MESSAGE_CREATE for the same msg_id. Two
    # queue entries mean two replies and a burned passive-reply msg_seq slot.
    import asyncio

    conn = QQOpenPlatformConnection.__new__(QQOpenPlatformConnection)
    conn._self_id = ""
    conn.logger = None
    conn._message_queue = asyncio.Queue(maxsize=8)
    payload = {
        "id": "same-msg", "group_openid": "G1",
        "content": "<@BBBBBBBBBBBBBBBB> 在吗",
        "author": {"member_openid": "MA"},
        "mentions": [{"member_openid": "BBBBBBBBBBBBBBBB", "is_you": True}],
    }

    async def drive(event_type):
        await conn._dispatch_event(event_type, payload, "e1")
        await asyncio.sleep(0)

    asyncio.run(drive("GROUP_AT_MESSAGE_CREATE"))
    asyncio.run(drive("GROUP_MESSAGE_CREATE"))
    assert conn._message_queue.qsize() == 1
