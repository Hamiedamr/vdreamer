"""SQLite-backed gateway/worker state for ``vdreamer dev`` (D11, testing phase).

The testing version runs on SQLite instead of Postgres: one ``vdreamer dev``
process (gateway + worker) keeps its runs, event log, threads, and interrupt
records in the same file as its LangGraph checkpoints, so a restart loses
nothing. The Postgres stores stay the production path for the later move.

Runs, threads, and interrupts are *write-through*: each store subclasses its
in-memory twin (the tested semantics — lifecycle map, one-active-run guard,
CAS resolve — stay in one place), loads the table at boot, and after every
mutating call upserts the records that changed. Single process only: the
in-memory dict is the source of truth while the process runs, SQLite is what
the next process loads. The event log is append-only and read straight from
SQLite, so a long history never has to fit in memory.

Artifacts (§3.4) keep their metadata row in ``vdreamer_artifacts`` and their
bytes on the local filesystem next to the database (D11: "a local filesystem
store"), at ``artifacts/{thread_id}/{run_id}/{artifact_id}.{ext}`` — the same
key layout MinIO gets in Milestone 6. Bytes are written before the row (D7).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeVar

import aiosqlite

from vdreamer.artifacts import ArtifactRecord, ArtifactStore
from vdreamer.eventlog import READ_PAGE_SIZE, EventHub, EventReader, EventSink, json_text
from vdreamer.events import EventEnvelope
from vdreamer.interrupts import (
    InMemoryInterruptStore,
    InterruptRecord,
    InterruptStatus,
)
from vdreamer.queue import InMemoryRunQueue, RunRecord, RunState
from vdreamer.threads import InMemoryThreadStore, ThreadRecord

_T = TypeVar("_T")

#: One table per record kind (``id`` → JSON record) plus the append-only log.
STATE_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS vdreamer_runs (
    id   TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vdreamer_threads (
    id   TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vdreamer_interrupts (
    id   TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vdreamer_artifacts (
    id   TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vdreamer_run_events (
    run_id TEXT    NOT NULL,
    seq    INTEGER NOT NULL,
    data   TEXT    NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""

_TABLES: Final[frozenset[str]] = frozenset(
    {"vdreamer_runs", "vdreamer_threads", "vdreamer_interrupts", "vdreamer_artifacts"}
)


async def open_state_db(path: str | Path) -> aiosqlite.Connection:
    """Open (and migrate) the state database at ``path``.

    WAL lets the checkpointer's own connection on the same file read and
    write alongside this one; ``busy_timeout`` absorbs the brief write locks.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(path), isolation_level=None)  # autocommit
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    await conn.executescript(STATE_DDL)
    return conn


async def _load(conn: aiosqlite.Connection, table: str) -> list[dict[str, Any]]:
    assert table in _TABLES
    async with conn.execute(f"SELECT data FROM {table}") as cur:  # noqa: S608 - fixed names
        rows = await cur.fetchall()
    return [json.loads(row[0]) for row in rows]


async def _upsert(
    conn: aiosqlite.Connection, table: str, records: Sequence[tuple[str, Any]]
) -> None:
    if not records:
        return
    assert table in _TABLES
    await conn.executemany(
        f"INSERT OR REPLACE INTO {table} (id, data) VALUES (?, ?)",  # noqa: S608
        [(key, json_text(dataclasses.asdict(record))) for key, record in records],
    )


async def _write_through(
    conn: aiosqlite.Connection,
    table: str,
    records: dict[str, Any],
    op: Awaitable[_T],
) -> _T:
    """Run one in-memory mutation, then upsert every record it replaced.

    Records are frozen dataclasses replaced on change, so identity tells what
    changed. The upsert writes each key's *current* value, built synchronously
    and submitted in one call: a later mutation's flush can only ever write a
    newer value, never be overwritten by an older one.
    """
    before = dict(records)
    try:
        return await op
    finally:
        changed = [(key, value) for key, value in records.items() if before.get(key) is not value]
        await _upsert(conn, table, changed)


def _run_from(data: dict[str, Any]) -> RunRecord:
    data["status"] = RunState(data["status"])
    data["roles"] = tuple(data.get("roles") or ())
    return RunRecord(**data)


def _thread_from(data: dict[str, Any]) -> ThreadRecord:
    return ThreadRecord(**data)


def _interrupt_from(data: dict[str, Any]) -> InterruptRecord:
    data["status"] = InterruptStatus(data["status"])
    data["actions"] = tuple(data.get("actions") or ())
    return InterruptRecord(**data)


class SqliteRunQueue(InMemoryRunQueue):
    """The in-memory queue, persisted to ``vdreamer_runs`` (wall-clock timestamps)."""

    _TABLE: Final = "vdreamer_runs"

    def __init__(self, conn: aiosqlite.Connection) -> None:
        super().__init__(clock=time.time)  # wall clock: timestamps outlive the process
        self._db = conn

    async def load(self) -> None:
        """Read every persisted run back into memory (call once, at boot).

        One process owns the file, so a run still ``running`` here belonged to
        the previous process: its lease is marked lapsed, and the worker's boot
        recovery sweep (§3.2) requeues it — or expires it after MAX_ATTEMPTS.
        """
        for data in await _load(self._db, self._TABLE):
            record = _run_from(data)
            if record.status is RunState.RUNNING:
                record = dataclasses.replace(record, lease_expires_at=0.0)
            self._runs[record.id] = record

    def _tracked(self, op: Awaitable[_T]) -> Awaitable[_T]:
        return _write_through(self._db, self._TABLE, self._runs, op)

    async def enqueue(self, *args: Any, **kwargs: Any) -> RunRecord:
        return await self._tracked(super().enqueue(*args, **kwargs))

    async def claim(self, claimant: str, *, agent: str | None = None) -> RunRecord | None:
        return await self._tracked(super().claim(claimant, agent=agent))

    async def heartbeat(self, run_id: str) -> bool:
        return await self._tracked(super().heartbeat(run_id))

    async def finish(self, run_id: str, status: RunState) -> None:
        await self._tracked(super().finish(run_id, status))

    async def suspend(self, run_id: str) -> None:
        await self._tracked(super().suspend(run_id))

    async def resume(self, run_id: str, resume: Mapping[str, Any] | None = None) -> None:
        await self._tracked(super().resume(run_id, resume))

    async def cancel(self, run_id: str) -> RunState | None:
        return await self._tracked(super().cancel(run_id))

    async def release(self, run_id: str) -> None:
        await self._tracked(super().release(run_id))

    async def requeue_expired(self) -> int:
        return await self._tracked(super().requeue_expired())


class SqliteThreadStore(InMemoryThreadStore):
    """The in-memory thread store, persisted to ``vdreamer_threads``."""

    _TABLE: Final = "vdreamer_threads"

    def __init__(self, conn: aiosqlite.Connection) -> None:
        super().__init__()
        self._db = conn

    async def load(self) -> None:
        for data in await _load(self._db, self._TABLE):
            record = _thread_from(data)
            self._threads[record.id] = record

    async def create(self, *args: Any, **kwargs: Any) -> ThreadRecord:
        return await _write_through(
            self._db, self._TABLE, self._threads, super().create(*args, **kwargs)
        )

    async def soft_delete(self, thread_id: str) -> bool:
        return await _write_through(
            self._db, self._TABLE, self._threads, super().soft_delete(thread_id)
        )


class SqliteInterruptStore(InMemoryInterruptStore):
    """The in-memory interrupt store, persisted to ``vdreamer_interrupts``."""

    _TABLE: Final = "vdreamer_interrupts"

    def __init__(self, conn: aiosqlite.Connection) -> None:
        super().__init__()
        self._db = conn

    async def load(self) -> None:
        for data in await _load(self._db, self._TABLE):
            record = _interrupt_from(data)
            self._records[record.interrupt_id] = record

    def _tracked(self, op: Awaitable[_T]) -> Awaitable[_T]:
        return _write_through(self._db, self._TABLE, self._records, op)

    async def create(self, record: InterruptRecord) -> InterruptRecord:
        return await self._tracked(super().create(record))

    async def resolve(
        self,
        interrupt_id: str,
        *,
        expected_version: int,
        resolved_by: str,
        resolution: Any,
    ) -> InterruptRecord:
        return await self._tracked(
            super().resolve(
                interrupt_id,
                expected_version=expected_version,
                resolved_by=resolved_by,
                resolution=resolution,
            )
        )

    async def cancel_for_run(self, run_id: str) -> int:
        return await self._tracked(super().cancel_for_run(run_id))


class SqliteEventLog(EventSink, EventReader):
    """The shared dev event log on ``vdreamer_run_events`` (the A3 replay source).

    Like ``InMemoryEventLog``: one instance is every run's sink and the
    gateway's reader, ``write`` allocates ``seq`` (a recovered attempt's driver
    restarts its count, so the log owns the sequence), ``hub`` wakes the SSE
    streams, and ``close`` is a no-op (per-run sink wrappers close it freely).
    """

    def __init__(self, conn: aiosqlite.Connection, hub: EventHub | None = None) -> None:
        self.hub = hub if hub is not None else EventHub()
        self._db = conn
        self._lock = asyncio.Lock()
        self._last_seq: dict[str, int] = {}

    async def write(self, envelope: EventEnvelope) -> None:
        async with self._lock:
            last = self._last_seq.get(envelope.run_id)
            if last is None:
                async with self._db.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM vdreamer_run_events WHERE run_id = ?",
                    (envelope.run_id,),
                ) as cur:
                    row = await cur.fetchone()
                last = int(row[0]) if row is not None else 0
            envelope.seq = last + 1
            await self._db.execute(
                "INSERT INTO vdreamer_run_events (run_id, seq, data) VALUES (?, ?, ?)",
                (envelope.run_id, envelope.seq, json_text(envelope.model_dump())),
            )
            self._last_seq[envelope.run_id] = envelope.seq
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
        async with self._db.execute(
            "SELECT data FROM vdreamer_run_events WHERE run_id = ? AND seq > ? "
            "ORDER BY seq LIMIT ?",
            (run_id, after_seq, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [EventEnvelope.model_validate_json(row[0]) for row in rows]


_SAFE_PART = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_EXTENSIONS: Final[dict[str, str]] = {
    "application/json": "json",
    "text/plain": "txt",
    "text/html": "html",
    "text/csv": "csv",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/svg+xml": "svg",
    "application/pdf": "pdf",
    "application/vnd.apache.parquet": "parquet",
}


def _part(value: str) -> str:
    """One path segment from a server-made id; anything else is refused."""
    if not _SAFE_PART.fullmatch(value) or value in (".", ".."):
        raise ValueError(f"unsafe artifact path segment {value!r}")
    return value


class SqliteArtifactStore(ArtifactStore):
    """Dev artifacts: rows in ``vdreamer_artifacts``, bytes under ``root`` (D7, D11)."""

    _TABLE: Final = "vdreamer_artifacts"

    def __init__(self, conn: aiosqlite.Connection, root: str | Path) -> None:
        self._db = conn
        self._root = Path(root)

    def object_path(self, record: ArtifactRecord) -> Path:
        """``{root}/{thread_id}/{run_id}/{artifact_id}.{ext}`` (the §3.4 key layout)."""
        ext = _EXTENSIONS.get(record.mime_type.split(";", 1)[0].strip().lower(), "bin")
        return (
            self._root
            / _part(record.thread_id)
            / _part(record.run_id)
            / f"{_part(record.artifact_id)}.{ext}"
        )

    async def put(self, record: ArtifactRecord, data: bytes | None) -> None:
        path = self.object_path(record)
        if data is None:
            await asyncio.to_thread(path.unlink, missing_ok=True)
        else:

            def _write() -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)

            await asyncio.to_thread(_write)
        await _upsert(self._db, self._TABLE, [(record.artifact_id, record)])

    async def get(self, artifact_id: str) -> ArtifactRecord | None:
        async with self._db.execute(
            "SELECT data FROM vdreamer_artifacts WHERE id = ?", (artifact_id,)
        ) as cur:
            row = await cur.fetchone()
        return ArtifactRecord(**json.loads(row[0])) if row is not None else None

    async def read_bytes(self, record: ArtifactRecord) -> bytes | None:
        path = self.object_path(record)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError:
            return None


@dataclass
class SqliteState:
    """Every dev store over one SQLite connection."""

    conn: aiosqlite.Connection
    queue: SqliteRunQueue
    threads: SqliteThreadStore
    interrupts: SqliteInterruptStore
    events: SqliteEventLog
    artifacts: SqliteArtifactStore


@asynccontextmanager
async def open_sqlite_state(
    path: str | Path,
    *,
    hub: EventHub | None = None,
    artifacts_dir: str | Path | None = None,
) -> AsyncIterator[SqliteState]:
    """Open the state file, load the persisted records, yield the stores, close.

    Artifact bytes go to ``artifacts_dir`` (default: ``artifacts/`` next to
    the database file).
    """
    conn = await open_state_db(path)
    root = Path(artifacts_dir) if artifacts_dir is not None else Path(path).parent / "artifacts"
    try:
        state = SqliteState(
            conn=conn,
            queue=SqliteRunQueue(conn),
            threads=SqliteThreadStore(conn),
            interrupts=SqliteInterruptStore(conn),
            events=SqliteEventLog(conn, hub),
            artifacts=SqliteArtifactStore(conn, root),
        )
        await state.queue.load()
        await state.threads.load()
        await state.interrupts.load()
        yield state
    finally:
        await conn.close()
