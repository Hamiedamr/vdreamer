"""Milestone-4 event log: the shared in-memory log, the hub, seq retries, the listener."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from vdreamer import eventlog
from vdreamer.eventlog import (
    EVENTS_NOTIFY_CHANNEL,
    EventHub,
    InMemoryEventLog,
    PostgresEventListener,
    PostgresEventLog,
    _row_to_envelope,
)
from vdreamer.events import EventEnvelope, EventType


def _env(run_id: str, seq: int = 0) -> EventEnvelope:
    return EventEnvelope(
        type=EventType.LOG,
        run_id=run_id,
        thread_id="t",
        agent="a",
        seq=seq,
        payload={"level": "info", "message": "x"},
    )


async def test_in_memory_log_owns_the_sequence() -> None:
    log = InMemoryEventLog()
    for seq in (1, 1, 99):  # a recovered attempt's driver restarts at 1
        await log.write(_env("r1", seq))
    await log.write(_env("r2", 5))
    assert [e.seq for e in log.events_for("r1")] == [1, 2, 3]
    assert [e.seq for e in await log.read("r1", after_seq=1)] == [2, 3]
    assert [e.seq for e in await log.read("r1", after_seq=0, limit=1)] == [1]
    assert [e.seq for e in log.events_for("r2")] == [1]
    assert await log.read("ghost") == []
    await log.close()  # shared: closing is a no-op
    assert len(log.events_for("r1")) == 3


async def test_hub_wakes_only_its_runs_subscribers() -> None:
    hub = EventHub()
    with hub.subscribe("r1") as one, hub.subscribe("r2") as two:
        assert hub.subscriber_count == 2
        hub.notify("r1")
        assert await one.wait(0.01) is True
        assert await two.wait(0.01) is False
        hub.notify("r1")
        hub.notify("r1")  # coalesced
        assert await one.wait(0.01) is True
        assert await one.wait(0.01) is False
        hub.notify_all()
        assert await one.wait(0.01) is True and await two.wait(0.01) is True
    assert hub.subscriber_count == 0


async def test_in_memory_log_notifies_the_hub() -> None:
    log = InMemoryEventLog()
    with log.hub.subscribe("r1") as sub:
        await log.write(_env("r1"))
        assert await sub.wait(0.01) is True


class _UniqueViolation(Exception):
    sqlstate = "23505"


class _Tx:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> None:
        return None


class _Cursor:
    def __init__(self, row: Any) -> None:
        self._row = row

    async def fetchone(self) -> Any:
        return self._row


class _RacyConnection:
    """The first insert loses a seq race (PK violation), the retry wins."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.inserts = 0
        self.notifies: list[tuple[Any, ...]] = []

    def transaction(self) -> _Tx:
        return _Tx()

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> _Cursor:
        if sql == eventlog.RUN_EVENTS_INSERT_SQL:
            self.inserts += 1
            if self.failures:
                self.failures -= 1
                raise _UniqueViolation()
            return _Cursor((4,))
        assert params is not None
        self.notifies.append(params)
        return _Cursor(None)


async def test_postgres_write_retries_a_raced_seq() -> None:
    conn = _RacyConnection(failures=1)
    envelope = _env("r1")
    await PostgresEventLog(conn).write(envelope)
    assert conn.inserts == 2 and envelope.seq == 4
    assert conn.notifies == [(EVENTS_NOTIFY_CHANNEL, json.dumps({"run_id": "r1", "seq": 4}))]


async def test_postgres_write_gives_up_after_the_retry_budget() -> None:
    conn = _RacyConnection(failures=10)
    with pytest.raises(_UniqueViolation):
        await PostgresEventLog(conn).write(_env("r1"))
    assert conn.inserts == eventlog._SEQ_RETRIES


async def test_postgres_write_does_not_retry_other_errors() -> None:
    class _Broken(_RacyConnection):
        async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> _Cursor:
            self.inserts += 1
            raise RuntimeError("connection lost")

    conn = _Broken(failures=0)
    with pytest.raises(RuntimeError):
        await PostgresEventLog(conn).write(_env("r1"))
    assert conn.inserts == 1


def test_row_to_envelope_rebuilds_the_wire_envelope() -> None:
    from datetime import UTC, datetime

    row = ("r1", 3, "log", ["child:1"], {"level": "info", "message": "m"},
           datetime(2026, 10, 14, tzinfo=UTC))
    envelope = _row_to_envelope(row, agent="a", thread_id="t")
    assert envelope.sse_id == "r1:3" and envelope.ns == ("child:1",)
    assert (envelope.agent, envelope.thread_id) == ("a", "t")
    assert envelope.ts == "2026-10-14T00:00:00+00:00"


async def test_listener_dispatches_notifications_and_reconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LISTEN`` → hub wake-ups; a dropped connection wakes everyone, then reconnects."""

    class _Notify:
        def __init__(self, payload: str) -> None:
            self.payload = payload

    connections: list[Any] = []

    class _Conn:
        def __init__(self, payloads: list[str]) -> None:
            self.payloads = payloads
            self.executed: list[str] = []
            self.closed = False

        @classmethod
        async def connect(cls, dsn: str, autocommit: bool) -> _Conn:
            batch = (
                [json.dumps({"run_id": "r1", "seq": 1}), "not json"]
                if not connections
                else []
            )
            conn = cls(batch)
            connections.append(conn)
            return conn

        async def execute(self, sql: str) -> None:
            self.executed.append(sql)

        async def notifies(self) -> Any:
            for payload in self.payloads:
                yield _Notify(payload)
            if len(connections) == 1:
                raise ConnectionError("dropped")  # the first connection dies
            await asyncio.Event().wait()  # the second one idles

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(eventlog, "_load_async_connection_class", lambda: _Conn)
    hub = EventHub()
    listener = PostgresEventListener("postgresql://x", hub, max_backoff_s=0.01)
    with hub.subscribe("r1") as sub:
        listener.start()
        assert await sub.wait(1.0) is True
        for _ in range(200):
            if len(connections) == 2 and listener.connected:
                break
            await asyncio.sleep(0.005)
        assert len(connections) == 2 and listener.connected
        await listener.probe()
        await listener.stop()
    assert connections[0].executed == [f"LISTEN {EVENTS_NOTIFY_CHANNEL}"]
    assert connections[0].closed and connections[1].closed
    assert listener.connected is False
    with pytest.raises(RuntimeError):
        await listener.probe()


def test_json_text_stringifies_what_json_cannot_hold() -> None:
    import datetime
    import decimal

    text = eventlog.json_text(
        {"cost": decimal.Decimal("1.50"), "at": datetime.date(2026, 10, 14), "n": 1}
    )
    assert json.loads(text) == {"cost": "1.50", "at": "2026-10-14", "n": 1}


async def test_postgres_write_survives_non_json_payload_values() -> None:
    import decimal

    conn = _RacyConnection(failures=0)
    envelope = _env("r1")
    envelope.payload["stats"] = {"rows": decimal.Decimal("3")}
    await PostgresEventLog(conn).write(envelope)  # must not raise TypeError
    assert envelope.seq == 4
