"""Event sink tests: SeqCounter, InMemoryEventSink, PostgresEventLog (offline).

No live Postgres here: the writer's SQL and DDL are asserted directly, and the
psycopg connection is faked. The lazy-import seam failure path is monkeypatched.
"""

import sys

import pytest

from vdreamer import (
    RUN_EVENTS_DDL,
    EventEnvelope,
    EventType,
    InMemoryEventSink,
    MissingPostgresDependencyError,
    PostgresEventLog,
    SeqCounter,
    eventlog,
)


def test_seq_counter_per_run_from_one() -> None:
    counter = SeqCounter()
    assert counter.next_seq("r_1") == 1
    assert counter.next_seq("r_1") == 2
    assert counter.next_seq("r_1") == 3
    # Per-run: another run counts from 1 independently (PRD §3.3, A3).
    assert counter.next_seq("r_2") == 1


async def test_in_memory_sink_collects_in_order() -> None:
    sink = InMemoryEventSink()
    for seq in (1, 2):
        await sink.write(
            EventEnvelope(
                type=EventType.LOG,
                run_id="r_1",
                thread_id="t_1",
                agent="a",
                seq=seq,
                payload={"level": "info", "message": "hi"},
            )
        )
    assert [e.seq for e in sink.events] == [1, 2]
    assert sink.events[0].type == EventType.LOG
    # close() is a safe no-op.
    await sink.close()


def test_run_events_ddl_matches_prd() -> None:
    assert "CREATE TABLE IF NOT EXISTS run_events" in RUN_EVENTS_DDL
    # A3: exactly this DDL — payload is jsonb and the PK is (run_id, seq).
    assert "payload jsonb" in RUN_EVENTS_DDL
    assert "PRIMARY KEY (run_id, seq)" in RUN_EVENTS_DDL
    for column in ("run_id", "seq", "type", "ns", "ts"):
        assert column in RUN_EVENTS_DDL


def test_insert_sql_allocates_seq_in_transaction() -> None:
    sql = eventlog.RUN_EVENTS_INSERT_SQL
    assert "INSERT INTO run_events" in sql
    # Seq derives from the per-run max+1 inside the same statement (§3.3).
    assert "COALESCE(MAX(seq), 0) + 1" in sql
    assert "WHERE run_id = %s" in sql
    assert "RETURNING seq" in sql


class _FakeCursor:
    def __init__(self, row: tuple[int]) -> None:
        self._row = row

    async def fetchone(self) -> tuple[int]:
        return self._row


class _FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeConnection:
    """Stands in for psycopg.AsyncConnection; records statements."""

    def __init__(self, next_seq: int = 7) -> None:
        self.next_seq = next_seq
        self.statements: list[tuple[str, tuple[object, ...] | None]] = []
        self.closed = False

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    async def execute(self, sql: str, params: tuple[object, ...] | None = None) -> _FakeCursor:
        self.statements.append((sql, params))
        return _FakeCursor((self.next_seq,))

    async def close(self) -> None:
        self.closed = True


def _envelope() -> EventEnvelope:
    return EventEnvelope(
        type=EventType.TOOL_CALL,
        run_id="r_1",
        thread_id="t_1",
        agent="sql-analyst",
        seq=0,  # placeholder: the log allocates the real seq
        ns=("child:abc",),
        payload={"tool_call_id": "tc_1"},
        ts="2026-10-13T12:00:00+00:00",
    )


async def test_postgres_log_write_assembles_sql_and_allocates_seq() -> None:
    conn = _FakeConnection(next_seq=7)
    log = PostgresEventLog(conn)
    envelope = _envelope()

    await log.write(envelope)

    sql, params = conn.statements[0]
    assert sql == eventlog.RUN_EVENTS_INSERT_SQL
    assert params == (
        "r_1",  # the CTE's per-run WHERE
        "r_1",  # the INSERT's run_id
        "tool.call",
        ["child:abc"],  # ns -> JSON array
        {"tool_call_id": "tc_1"},
        "2026-10-13T12:00:00+00:00",
    )
    # The allocated seq lands on the envelope so sse_id is usable right away.
    assert envelope.seq == 7
    assert envelope.sse_id == "r_1:7"


async def test_postgres_log_setup_runs_ddl() -> None:
    conn = _FakeConnection()
    log = PostgresEventLog(conn)
    await log.setup()
    assert conn.statements == [(RUN_EVENTS_DDL, None)]


async def test_postgres_log_close_closes_connection() -> None:
    conn = _FakeConnection()
    log = PostgresEventLog(conn)
    await log.close()
    assert conn.closed is True


async def test_postgres_log_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None in sys.modules makes `from psycopg import AsyncConnection` raise
    # ImportError, which the lazy-import seam converts to the friendly error.
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(MissingPostgresDependencyError):
        await PostgresEventLog.connect("postgresql://vd:vd@localhost/vd")
