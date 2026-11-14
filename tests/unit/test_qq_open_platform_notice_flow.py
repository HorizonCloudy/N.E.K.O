"""Notice handling in the QQ Open Platform connector: sinks and reception state.

Two families of defect lived here.

Sinks: ``_dispatch_event`` awaited every listener inline, inside the one
``_receive_loop`` task. A listener that waits on a slow request therefore stopped the
connector from reading the next message, and a listener that never returned stopped it
indefinitely -- catching the exception cannot help when nothing is raised.

Reception state: ``receives_all_messages`` and ``needs_attention`` were one-way
latches. The first full-group push set them and nothing cleared them -- not the group
turning push off, not the bot being removed, not a reconnect -- so the connector
advertised a mode the platform had taken away, for the life of the process.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from utils.connection.qq import open_platform as mod
from utils.connection.qq.open_platform import QQOpenPlatformConnection


class _Log:
    def __init__(self):
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, msg):
        self.warnings.append(str(msg))

    def info(self, msg):
        self.infos.append(str(msg))


def _conn():
    conn = QQOpenPlatformConnection(app_id="1", client_secret="2")
    conn.logger = _Log()
    return conn


def run(coro):
    return asyncio.run(coro)


def _prove_full_push(conn, group_id="G1"):
    run(conn._dispatch_event("GROUP_MESSAGE_CREATE", {
        "id": "m-" + group_id, "group_openid": group_id, "content": "hi",
        "author": {"member_openid": "MA"}}, "e0"))


# ---- sinks ---------------------------------------------------------------

def test_a_hanging_sink_does_not_stop_the_next_notice_from_being_handled():
    conn = _conn()
    started = asyncio.Event()
    release = asyncio.Event()
    seen: list[str] = []

    async def slow_sink(event_type, payload):
        if event_type == "GROUP_ADD_ROBOT":
            started.set()
            await release.wait()              # never returns unless released
        seen.append(event_type)

    conn.add_event_sink(slow_sink)

    async def go():
        await conn._dispatch_event("GROUP_ADD_ROBOT", {"group_openid": "G1"}, "e1")
        await asyncio.sleep(0)                 # let the detached task start
        assert started.is_set()
        await conn._dispatch_event("FRIEND_ADD", {"user_openid": "U1"}, "e2")
        await asyncio.sleep(0)             # the second detached task needs one turn
        return list(seen)

    # Dispatch returned while the first listener was still blocked, so the second
    # notice reached the same sink instead of queueing behind it. Under the old inline
    # await this deadlocked: the receive loop was waiting on release.
    assert run(go()) == ["FRIEND_ADD"]
    release.set()


def test_sink_timeout_is_reported_and_the_notice_still_drains(monkeypatch):
    conn = _conn()
    monkeypatch.setattr(mod, "_NOTICE_SINK_TIMEOUT_SECONDS", 0.01)
    reached: list[str] = []

    async def sleeper(event_type, payload):
        await asyncio.sleep(5)
        reached.append(event_type)

    conn.add_event_sink(sleeper)

    async def go():
        await conn._dispatch_event("GROUP_ADD_ROBOT", {"group_openid": "G1"}, "e1")
        await asyncio.sleep(0.08)
        return list(conn._sink_tasks)

    tasks = run(go())
    assert reached == []                        # gave up rather than waited
    assert any("timeout" in line or "超时" in line for line in conn.logger.warnings)
    assert not tasks                            # a finished task leaves the set


def test_too_many_outstanding_sinks_drop_the_notice_instead_of_piling_tasks(monkeypatch):
    conn = _conn()
    monkeypatch.setattr(mod, "_NOTICE_SINK_MAX_TASKS", 2)
    gate = asyncio.Event()

    async def blocker(event_type, payload):
        await gate.wait()

    conn.add_event_sink(blocker)

    async def go():
        for index in range(5):
            await conn._dispatch_event(
                "GROUP_ADD_ROBOT", {"group_openid": "G" + str(index)}, "e" + str(index))
        return len(conn._sink_tasks)

    outstanding = run(go())
    assert outstanding == 2                     # bounded, not five
    assert conn.logger.warnings                 # and the drop was said out loud
    gate.set()


def test_disconnect_cancels_outstanding_sink_tasks():
    conn = _conn()
    gate = asyncio.Event()

    async def blocker(event_type, payload):
        await gate.wait()

    conn.add_event_sink(blocker)

    async def go():
        await conn._dispatch_event("GROUP_ADD_ROBOT", {"group_openid": "G1"}, "e1")
        await asyncio.sleep(0)
        tasks = list(conn._sink_tasks)
        await conn.disconnect()
        await asyncio.sleep(0)             # let the cancellation land
        return tasks

    tasks = run(go())
    assert len(tasks) == 1
    assert tasks[0].cancelled() or tasks[0].done()


def test_a_sync_sink_that_raises_is_logged_and_the_transport_survives():
    conn = _conn()
    calls: list[str] = []

    def bad_sink(event_type, payload):
        calls.append(event_type)
        raise RuntimeError("consumer exploded")

    conn.add_event_sink(bad_sink)
    run(conn._dispatch_event("FRIEND_DEL", {"user_openid": "U1"}, "e1"))
    assert calls == ["FRIEND_DEL"]
    assert any("事件监听器异常" in line for line in conn.logger.warnings)


# ---- reception state -----------------------------------------------------

def test_turning_push_off_takes_the_group_out_of_full_message_mode():
    conn = _conn()
    _prove_full_push(conn, "G1")
    assert conn.receives_all_messages is True
    assert conn.group_receives_all_messages("G1") is True

    run(conn._dispatch_event("GROUP_MSG_REJECT", {"group_openid": "G1"}, "e2"))
    assert conn.group_receives_all_messages("G1") is False
    assert conn.receives_all_messages is False
    assert conn.needs_attention is False


def test_one_group_turning_push_off_does_not_silence_another():
    conn = _conn()
    _prove_full_push(conn, "G1")
    _prove_full_push(conn, "G2")
    run(conn._dispatch_event("GROUP_MSG_REJECT", {"group_openid": "G1"}, "e3"))
    assert conn.group_receives_all_messages("G1") is False
    assert conn.group_receives_all_messages("G2") is True
    assert conn.receives_all_messages is True


def test_group_msg_receive_proves_the_mode_without_any_traffic():
    conn = _conn()
    run(conn._dispatch_event("GROUP_MSG_RECEIVE", {"group_openid": "G7"}, "e4"))
    assert conn.group_recv_msg_setting("G7") == "all"
    assert conn.group_receives_all_messages("G7") is True


def test_being_removed_from_a_group_forgets_what_that_group_proved():
    conn = _conn()
    _prove_full_push(conn, "G1")
    conn._learn_bot_openid("G1", "BOTID1")
    conn._bot_member_role_by_group["G1"] = "admin"
    run(conn._dispatch_event("GROUP_DEL_ROBOT", {"group_openid": "G1"}, "e5"))
    assert conn.group_receives_all_messages("G1") is False
    assert conn._remembered_bot_openid("G1") == ""
    assert conn.group_bot_member_role("G1") == ""
    assert "G1" not in conn._seen_groups


def test_identify_relearns_the_mode_that_a_resume_keeps():
    # A brand-new session can find the privilege withdrawn while the bot was away; a
    # resumed session never lost it, so only Identify may clear the proof.
    conn = _conn()
    _prove_full_push(conn, "G1")

    class _WS:
        def __init__(self):
            self.sent: list[dict] = []

        async def send(self, raw):
            self.sent.append(json.loads(raw))

        async def recv(self):
            await asyncio.sleep(0)
            return json.dumps({
                "op": 0, "t": "READY", "s": 1,
                "d": {"session_id": "S1", "user": {"id": "u1"}}})

        async def close(self):
            return None

    ws = _WS()
    conn._ws = ws
    conn._access_token = "t"
    conn._token_expires_at = float("inf")

    run(conn._handshake(is_reconnect=False))
    assert conn._all_messages_observed_groups == set()


def test_a_successful_resume_keeps_the_mode_it_already_proved():
    conn = _conn()
    _prove_full_push(conn, "G1")

    class _WS:
        def __init__(self):
            self.frames = [
                json.dumps({"op": 10, "d": {"heartbeat_interval": 30000,
                                            "session_id": "S0"}}),
                json.dumps({"op": 0, "t": "RESUMED", "s": 5, "d": {}}),
            ]

        async def send(self, raw):
            return None

        async def recv(self):
            await asyncio.sleep(0)
            return self.frames.pop(0) if self.frames else json.dumps(
                {"op": 0, "t": "READY", "s": 6, "d": {"session_id": "S0",
                                                      "user": {"id": "u1"}}})

        async def close(self):
            return None

    conn._ws = _WS()
    conn._access_token = "t"
    conn._token_expires_at = float("inf")
    conn._session_id = "S0"
    conn._last_seq = 4

    run(conn._handshake(is_reconnect=True))
    assert conn._all_messages_observed_groups == {"G1"}
