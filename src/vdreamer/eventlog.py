"""Event sinks, the ``run_events`` log, and its live fan-out (PRD §3.3, A3, A7).

Milestone 2 (WP1) landed the write side: the sink abstraction
(``EventSink``), the in-memory collector (``InMemoryEventSink``), the per-run
``seq`` allocator, and the Postgres writer (``PostgresEventLog``) that persists
envelopes to ``run_events(run_id, seq, type, ns, payload jsonb, ts)`` with PK
``(run_id, seq)`` (A3).

Milestone 4 adds the read side the gateway's SSE transport needs (§2.5:
"replay ``seq > N``, then tail live; any replica serves any reconnect"):

- :class:`EventReader` — ``read(run_id, after_seq)`` for ``Last-Event-ID``
  replay; :class:`PostgresEventStore` (short-lived connections, any gateway
  replica) and :class:`InMemoryEventLog` (``vdreamer dev``, D11, and tests —
  one shared log that is sink *and* reader) implement it.
- :class:`EventHub` — per-run wake-ups for live tailing. Writers notify it
  directly in-process; across processes ``PostgresEventLog.write`` issues
  ``NOTIFY vdreamer_events, '{run_id, seq}'`` in the insert's transaction (A7)
  and :class:`PostgresEventListener` feeds the hub from ``LISTEN``. A wake-up
  only means "re-read": the log stays the source of truth, so a lost
  notification costs latency (the SSE loop re-reads on every heartbeat), never
  an event.

``run_events`` DDL lives here as ``RUN_EVENTS_DDL`` so ``migrate`` reuses the
exact same statement.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import json
import logging
from collections.abc import Iterator
from typing import Any, Final

from vdreamer.events import EventEnvelope, EventType
from vdreamer.persistence import MissingPostgresDependencyError

_log = logging.getLogger(__name__)

#: The ``run_events`` table (PRD §3.3, A3): exact columns
#: ``run_events(run_id, seq, type, ns, payload jsonb, ts)``, PK ``(run_id, seq)``.
#: Idempotent — ``migrate`` reuses this constant.
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
#: ``ns`` and ``payload`` travel as JSON text cast to ``jsonb`` — psycopg
#: cannot adapt a ``dict`` without a wrapper and would send a ``list`` as a
#: Postgres array.
#:
#: Concurrency note: one worker writes one run's events, and the gateway
#: writes only for runs no worker holds (a cancelled ``queued`` /
#: ``awaiting_approval`` run, an ``interrupt.resolved``), so concurrent writers
#: for one run are rare. Should two collide, both compute the same ``MAX(seq)
#: + 1`` and the PK rejects the second; ``PostgresEventLog.write`` retries it
#: with a fresh allocation (``_SEQ_RETRIES``).
RUN_EVENTS_INSERT_SQL: Final[str] = """
WITH next_seq AS (
    SELECT COALESCE(MAX(seq), 0) + 1 AS seq
    FROM run_events
    WHERE run_id = %s
)
INSERT INTO run_events (run_id, seq, type, ns, payload, ts)
SELECT %s, next_seq.seq, %s, %s::jsonb, %s::jsonb, %s::timestamptz
FROM next_seq
RETURNING seq
"""

#: The ``LISTEN``/``NOTIFY`` channel of the live fan-out (A7): the payload is
#: ``{"run_id": …, "seq": …}``, sent in the insert's transaction so it is
#: delivered exactly when the row becomes visible.
EVENTS_NOTIFY_CHANNEL: Final[str] = "vdreamer_events"

_NOTIFY_SQL: Final[str] = "SELECT pg_notify(%s, %s)"

#: Replay read for ``Last-Event-ID`` (§2.5): ``seq > N`` in order.
RUN_EVENTS_READ_SQL: Final[str] = """
SELECT run_id, seq, type, ns, payload, ts
FROM run_events
WHERE run_id = %s AND seq > %s
ORDER BY seq
LIMIT %s
"""

#: Attempts for one write when a concurrent writer took the allocated ``seq``.
_SEQ_RETRIES: Final[int] = 3

#: The SQLSTATE of ``unique_violation`` — the PK collision of a raced ``seq``.
_UNIQUE_VIOLATION: Final[str] = "23505"

#: Replay page size: one ``read`` returns at most this many envelopes.
READ_PAGE_SIZE: Final[int] = 500


def json_text(value: Any) -> str:
    """JSON text for a ``jsonb`` parameter, stringifying what JSON cannot hold.

    Payloads carry agent-supplied values (an approval payload's ``Decimal``
    stats, a ``datetime``): a value the protocol cannot represent becomes its
    ``str`` rather than failing a healthy run over serialization.
    """
    return json.dumps(value, default=str)


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


class EventReader(abc.ABC):
    """The read side of the event log: ``Last-Event-ID`` replay (§2.5, A3)."""

    @abc.abstractmethod
    async def read(
        self,
        run_id: str,
        after_seq: int = 0,
        limit: int = READ_PAGE_SIZE,
        *,
        agent: str = "",
        thread_id: str = "",
    ) -> list[EventEnvelope]:
        """The run's envelopes with ``seq > after_seq``, in ``seq`` order.

        ``agent`` / ``thread_id`` fill the envelope fields a backend does not
        store (``run_events`` keeps A3's exact columns only); backends that
        keep whole envelopes ignore them.
        """


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


# ---------------------------------------------------------------------------
# Live fan-out
# ---------------------------------------------------------------------------


class EventSubscription:
    """One SSE stream's wake-up channel for one run (see :class:`EventHub`)."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._event = asyncio.Event()

    def _wake(self) -> None:
        self._event.set()

    async def wait(self, timeout: float) -> bool:
        """Block until a wake-up or ``timeout``; ``True`` when woken.

        Wake-ups coalesce: several notifications before the wait return once.
        The caller re-reads the log either way.
        """
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
        except TimeoutError:
            return False
        self._event.clear()
        return True


class EventHub:
    """Per-run wake-ups for live tailing (§2.5), one per gateway process.

    Subscribe *before* the first replay read: a write landing between the
    read and the subscribe would otherwise be noticed only at the next
    heartbeat. All methods run on the event loop that owns the hub.
    """

    def __init__(self) -> None:
        self._subs: dict[str, set[EventSubscription]] = {}

    @contextlib.contextmanager
    def subscribe(self, run_id: str) -> Iterator[EventSubscription]:
        """Subscribe to ``run_id``'s wake-ups for the duration of the block."""
        sub = EventSubscription(run_id)
        self._subs.setdefault(run_id, set()).add(sub)
        try:
            yield sub
        finally:
            subs = self._subs.get(run_id)
            if subs is not None:
                subs.discard(sub)
                if not subs:
                    del self._subs[run_id]

    def notify(self, run_id: str) -> None:
        """Wake every subscriber of ``run_id``."""
        for sub in self._subs.get(run_id, ()):
            sub._wake()

    def notify_all(self) -> None:
        """Wake every subscriber — after a listener reconnect, so streams re-read."""
        for subs in self._subs.values():
            for sub in subs:
                sub._wake()

    @property
    def subscriber_count(self) -> int:
        """Live subscriptions across all runs."""
        return sum(len(subs) for subs in self._subs.values())


class InMemoryEventLog(EventSink, EventReader):
    """One shared in-process log: every run's sink, reader, and fan-out (D11).

    The ``vdreamer dev`` topology and the gateway tests use one instance for
    everything: the worker writes through it (``close`` is a no-op, so the
    per-run sink wrapper may close it freely), the gateway reads from it, and
    ``hub`` wakes the SSE streams. ``write`` allocates ``seq`` like the
    Postgres log does — a recovered attempt's driver restarts its own count at
    1, so the log, never the caller, owns the run's sequence.
    """

    def __init__(self, hub: EventHub | None = None) -> None:
        self.hub = hub if hub is not None else EventHub()
        self._events: dict[str, list[EventEnvelope]] = {}

    async def write(self, envelope: EventEnvelope) -> None:
        events = self._events.setdefault(envelope.run_id, [])
        envelope.seq = len(events) + 1
        events.append(envelope)
        self.hub.notify(envelope.run_id)

    async def read(
        self,
        run_id: str,
        after_seq: int = 0,
        limit: int = READ_PAGE_SIZE,
        *,
        agent: str = "",
        thread_id: str = "",
    ) -> list[EventEnvelope]:
        events = self._events.get(run_id, [])
        return events[after_seq : after_seq + limit]

    def events_for(self, run_id: str) -> tuple[EventEnvelope, ...]:
        """Every envelope of ``run_id``, in ``seq`` order."""
        return tuple(self._events.get(run_id, ()))


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------


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
    """Persists envelopes to ``run_events`` (PRD §3.3, A3) and announces them (A7).

    ``seq`` is *not* the caller's to guess: ``write`` allocates it from the
    run's ``MAX(seq) + 1`` inside the same transaction as the INSERT (see
    ``RUN_EVENTS_INSERT_SQL``) and writes the allocated value back onto the
    envelope, so the caller can use ``envelope.sse_id`` and
    ``Last-Event-ID`` replay (§2.5) right after ``await write``. The same
    transaction issues ``NOTIFY vdreamer_events, '{run_id, seq}'``, which
    Postgres delivers on commit — the gateway's
    :class:`PostgresEventListener` turns it into a live wake-up.
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
        """Create ``run_events`` if missing (idempotent; ``migrate`` reuses
        the same ``RUN_EVENTS_DDL``)."""
        async with self._conn.transaction():
            await self._conn.execute(RUN_EVENTS_DDL)

    async def write(self, envelope: EventEnvelope) -> None:
        """Persist one envelope; allocate its ``seq`` from the table; NOTIFY.

        A ``seq`` raced by a concurrent writer for the same run surfaces as a
        PK ``unique_violation``; the write retries with a fresh allocation, up
        to ``_SEQ_RETRIES`` attempts.
        """
        for attempt in range(1, _SEQ_RETRIES + 1):
            try:
                seq = await self._insert(envelope)
            except Exception as error:
                if getattr(error, "sqlstate", None) != _UNIQUE_VIOLATION or (
                    attempt == _SEQ_RETRIES
                ):
                    raise
                continue
            envelope.seq = seq
            return

    async def _insert(self, envelope: EventEnvelope) -> int:
        async with self._conn.transaction():
            cur = await self._conn.execute(
                RUN_EVENTS_INSERT_SQL,
                (
                    envelope.run_id,
                    envelope.run_id,
                    envelope.type.value,
                    json.dumps(list(envelope.ns)),
                    json_text(envelope.payload),
                    envelope.ts,
                ),
            )
            row = await cur.fetchone()
            if row is None:  # pragma: no cover - INSERT without RETURNING row
                raise RuntimeError("run_events insert returned no seq")
            seq = int(row[0])
            await self._conn.execute(
                _NOTIFY_SQL,
                (EVENTS_NOTIFY_CHANNEL, json.dumps({"run_id": envelope.run_id, "seq": seq})),
            )
        return seq

    async def close(self) -> None:
        """Close the connection."""
        await self._conn.close()


def _row_to_envelope(row: Any, agent: str = "", thread_id: str = "") -> EventEnvelope:
    """Rebuild an envelope from a ``run_events`` row (see ``PostgresEventStore``)."""
    ts = row[5]
    return EventEnvelope(
        type=EventType(str(row[2])),
        run_id=str(row[0]),
        thread_id=thread_id,
        agent=agent,
        seq=int(row[1]),
        ts=ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
        ns=tuple(str(part) for part in (row[3] or ())),
        payload=dict(row[4] or {}),
    )


class PostgresEventStore(EventSink, EventReader):
    """The gateway's handle on ``run_events``: replay reads plus occasional writes.

    Stateless like the gateway (§2.1): every call opens a short-lived
    autocommit connection, so any replica serves any reconnect. ``run_events``
    stores neither ``thread_id`` nor ``agent`` (A3's exact columns); the
    caller passes them to :meth:`read` from the run row so replayed envelopes
    carry the full §3.3 envelope.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    async def write(self, envelope: EventEnvelope) -> None:
        log = await PostgresEventLog.connect(self._dsn)
        try:
            await log.write(envelope)
        finally:
            await log.close()

    async def read(
        self,
        run_id: str,
        after_seq: int = 0,
        limit: int = READ_PAGE_SIZE,
        *,
        agent: str = "",
        thread_id: str = "",
    ) -> list[EventEnvelope]:
        conn_cls = _load_async_connection_class()
        conn = await conn_cls.connect(self._dsn, autocommit=True)
        try:
            cur = await conn.execute(RUN_EVENTS_READ_SQL, (run_id, after_seq, limit))
            rows = await cur.fetchall()
        finally:
            await conn.close()
        return [_row_to_envelope(row, agent=agent, thread_id=thread_id) for row in rows]


class PostgresEventListener:
    """``LISTEN vdreamer_events`` → :class:`EventHub` wake-ups (A7), one per gateway.

    Holds one dedicated connection in a background task. On any failure it
    logs, wakes every subscriber (they re-read, so nothing written during the
    outage is missed), and reconnects with capped exponential backoff.
    ``connected`` feeds ``/readyz``.
    """

    def __init__(
        self,
        dsn: str,
        hub: EventHub,
        *,
        max_backoff_s: float = 10.0,
    ) -> None:
        self._dsn = dsn
        self._hub = hub
        self._max_backoff_s = max_backoff_s
        self._task: asyncio.Task[None] | None = None
        self.connected = False

    def start(self) -> None:
        """Start the listen loop on the running event loop (idempotent)."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="vdreamer-event-listener")

    async def stop(self) -> None:
        """Cancel the listen loop and wait for it to unwind."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.connected = False

    async def probe(self) -> None:
        """Readiness probe: raise unless the ``LISTEN`` connection is up."""
        if not self.connected:
            raise RuntimeError(f"not listening on {EVENTS_NOTIFY_CHANNEL!r}")

    def _dispatch(self, payload: str) -> None:
        try:
            run_id = json.loads(payload)["run_id"]
        except (ValueError, KeyError, TypeError):
            _log.warning("ignoring malformed %s payload: %r", EVENTS_NOTIFY_CHANNEL, payload)
            return
        self._hub.notify(str(run_id))

    async def _run(self) -> None:
        backoff = 0.5
        while True:
            try:
                conn_cls = _load_async_connection_class()
                conn = await conn_cls.connect(self._dsn, autocommit=True)
                try:
                    await conn.execute(f"LISTEN {EVENTS_NOTIFY_CHANNEL}")
                    self.connected = True
                    backoff = 0.5
                    # Anything written while we were away: let streams re-read.
                    self._hub.notify_all()
                    async for notify in conn.notifies():
                        self._dispatch(notify.payload)
                finally:
                    self.connected = False
                    await conn.close()
            except asyncio.CancelledError:
                raise
            except MissingPostgresDependencyError:
                raise
            except Exception as error:
                _log.warning("event listener lost its connection (%s); reconnecting", error)
            self._hub.notify_all()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff_s)
