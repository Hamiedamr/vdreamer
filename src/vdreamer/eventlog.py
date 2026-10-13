"""Event sinks and the Postgres ``run_events`` log (PRD §3.3, A3).

Milestone 2 scope (WP1): the sink abstraction (``EventSink``), the in-memory
collector tests and ``vdreamer dev`` use (``InMemoryEventSink``), the per-run
``seq`` allocator, and the Postgres writer (``PostgresEventLog``) that
persists envelopes to ``run_events(run_id, seq, type, ns, payload jsonb, ts)``
with PK ``(run_id, seq)`` (A3). The WP2 stream drivers emit *into* a sink;
the SSE replay transport (§2.5) is a later work package and reads the table
back out.

``run_events`` DDL lives here as ``RUN_EVENTS_DDL`` so the Milestone 10
``migrate`` command can reuse the exact same statement.
"""

from __future__ import annotations

import abc
from typing import Any, Final

from vdreamer.events import EventEnvelope
from vdreamer.persistence import MissingPostgresDependencyError

#: The ``run_events`` table (PRD §3.3, A3): exact columns
#: ``run_events(run_id, seq, type, ns, payload jsonb, ts)``, PK ``(run_id, seq)``.
#: Idempotent — Milestone 10 ``migrate`` reuses this constant.
RUN_EVENTS_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS run_events (
    run_id  text        NOT NULL,
    seq     bigint       NOT NULL,
    type    text        NOT NULL,
    ns      jsonb        NOT NULL DEFAULT '[]'::jsonb,
    payload jsonb        NOT NULL DEFAULT '{}'::jsonb,
    ts      timestamptz  NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, seq)
)
"""

#: One insert + per-run seq allocation in a single statement: the CTE reads
#: the run's ``MAX(seq) + 1`` and the INSERT consumes it atomically, so the
#: allocated ``seq`` and the row land in the same transaction (PRD §3.3).
#:
#: Concurrency note: two concurrent writers for the same run could both
#: compute the same ``MAX(seq) + 1`` and the second INSERT would then fail
#: on the PK. That is acceptable: concurrent writers for one run are not
#: expected until Milestone 3 (one worker claims one run), and the PK makes
#: any violation loud rather than silent. Revisit with the queue (M3).
RUN_EVENTS_INSERT_SQL: Final[str] = """
WITH next_seq AS (
    SELECT COALESCE(MAX(seq), 0) + 1 AS seq
    FROM run_events
    WHERE run_id = %s
)
INSERT INTO run_events (run_id, seq, type, ns, payload, ts)
SELECT %s, next_seq.seq, %s, %s, %s, %s
FROM next_seq
RETURNING seq
"""


class EventSink(abc.ABC):
    """Where the stream drivers write enveloped events (PRD §3.3).

    One sink per run; ``write`` must be safe to call from any await point of
    the run's driver. ``close`` releases resources and defaults to no-op —
    in-memory sinks need nothing.
    """

    @abc.abstractmethod
    async def write(self, envelope: EventEnvelope) -> None:
        """Persist one envelope."""

    async def close(self) -> None:
        """Release sink resources. Default: nothing."""
        return None


class InMemoryEventSink(EventSink):
    """A thread-free async list collector — for tests and ``vdreamer dev`` (D11).

    ``write`` never blocks; ``events`` snapshots what landed, in write order.
    """

    def __init__(self) -> None:
        self._events: list[EventEnvelope] = []

    @property
    def events(self) -> tuple[EventEnvelope, ...]:
        """The envelopes written so far, in order."""
        return tuple(self._events)

    async def write(self, envelope: EventEnvelope) -> None:
        self._events.append(envelope)


class SeqCounter:
    """Per-run ``seq`` allocation: monotonically increasing ints from 1.

    PRD §3.3: ``seq`` is per-``run_id`` (the PK is ``(run_id, seq)``), so each
    run counts from 1 independently. This is the in-process allocator the
    in-memory/dev path and the envelope builders use; the Postgres log
    allocates its own ``seq`` from the table (see ``PostgresEventLog.write``).
    """

    def __init__(self) -> None:
        self._next: dict[str, int] = {}

    def next_seq(self, run_id: str) -> int:
        """Return the next ``seq`` for ``run_id``, starting at 1."""
        seq = self._next.get(run_id, 1)
        self._next[run_id] = seq + 1
        return seq


def _load_async_connection_class() -> Any:
    """Import ``psycopg.AsyncConnection`` lazily — the ``[postgres]`` extra seam.

    Same pattern as ``persistence._load_saver_class``: environments without
    psycopg fail with a friendly error at the point of use instead of an
    ``ImportError`` at import time, and tests monkeypatch this seam to fake
    the connection offline.
    """
    try:
        from psycopg import AsyncConnection
    except ImportError as exc:
        raise MissingPostgresDependencyError(
            "psycopg is required for the vdreamer[postgres] extra: "
            "pip install 'vdreamer[postgres]'"
        ) from exc
    return AsyncConnection


class PostgresEventLog(EventSink):
    """Persists envelopes to ``run_events`` (PRD §3.3, A3).

    ``seq`` is *not* the caller's to guess: ``write`` allocates it from the
    run's ``MAX(seq) + 1`` inside the same transaction as the INSERT (see
    ``RUN_EVENTS_INSERT_SQL``) and writes the allocated value back onto the
    envelope, so the caller can use ``envelope.sse_id`` and
    ``Last-Event-ID`` replay (§2.5) right after ``await write``.
    """

    def __init__(self, connection: Any) -> None:
        """Wrap an open async connection (``psycopg.AsyncConnection``).

        Prefer ``connect`` for the default wiring; the constructor accepting
        a connection keeps the psycopg type out of the vdreamer surface and
        lets tests pass a fake.
        """
        self._conn = connection

    @classmethod
    async def connect(cls, dsn: str) -> PostgresEventLog:
        """Open a dedicated async connection to ``dsn``.

        ``autocommit`` is on: each statement commits on its own and explicit
        ``transaction()`` blocks (``write``) still get real transactions.
        The DSN comes from the environment, never YAML (A23).
        """
        conn_cls = _load_async_connection_class()
        conn = await conn_cls.connect(dsn, autocommit=True)
        return cls(conn)

    async def setup(self) -> None:
        """Create ``run_events`` if missing (idempotent; M10 ``migrate`` reuses
        the same ``RUN_EVENTS_DDL``)."""
        async with self._conn.transaction():
            await self._conn.execute(RUN_EVENTS_DDL)

    async def write(self, envelope: EventEnvelope) -> None:
        """Persist one envelope; allocate its ``seq`` from the table.

        The INSERT..SELECT..RETURNING allocates ``MAX(seq) + 1`` per run
        inside the insert's transaction and the allocated value is written
        back to ``envelope.seq``. Concurrent writers for one run are not
        expected until M3 — see the ``RUN_EVENTS_INSERT_SQL`` note.
        """
        async with self._conn.transaction():
            cur = await self._conn.execute(
                RUN_EVENTS_INSERT_SQL,
                (
                    envelope.run_id,
                    envelope.run_id,
                    envelope.type.value,
                    list(envelope.ns),
                    envelope.payload,
                    envelope.ts,
                ),
            )
            row = await cur.fetchone()
        if row is None:  # pragma: no cover - INSERT without RETURNING row
            raise RuntimeError("run_events insert returned no seq")
        envelope.seq = int(row[0])

    async def close(self) -> None:
        """Close the connection."""
        await self._conn.close()
