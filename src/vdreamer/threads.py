"""Threads: the session record behind ``/agents/{name}/threads`` (PRD §2.5, §3.1).

``thread_id`` is the session (§3.1). The gateway creates one per
``POST /agents/{name}/threads`` and scopes every read to its ``owner_id``
(``X-User-Id``, D13): ``GET /threads`` lists only the caller's threads, and a
thread someone else owns is indistinguishable from a missing one. Deletion is
soft (§2.5): the row stays, stamped ``deleted_at``, and disappears from every
read.

The thread's conversation state is *not* here — it lives in the agent's
checkpoints, keyed by the same ``thread_id`` (§2.4); ``GET /threads/{id}``
reads its history from the checkpointer. This table only holds what the
checkpointer does not know: which agent the thread belongs to, who owns it,
and whether it was deleted.

``InMemoryThreadStore`` backs ``vdreamer dev`` (D11) and the tests;
``PostgresThreadStore`` is production and reuses the queue's lazy-psycopg
seam. ``THREADS_DDL`` is applied by ``vdreamer migrate``.
"""

from __future__ import annotations

import abc
import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Final

from vdreamer.queue import _load_async_connection_class


@dataclass(frozen=True)
class ThreadRecord:
    """One ``threads`` row. Timestamps are epoch seconds."""

    id: str
    agent: str
    owner_id: str
    created_at: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    deleted_at: float | None = None


def new_thread_id() -> str:
    """A fresh thread id — one thread is one session (PRD §3.1)."""
    return f"thread_{uuid.uuid4().hex}"


class ThreadStore(abc.ABC):
    """Owner-scoped thread records (§2.5). Soft-deleted threads are invisible."""

    @abc.abstractmethod
    async def create(
        self,
        agent: str,
        owner_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> ThreadRecord:
        """Create a thread owned by ``owner_id``; ``thread_id`` defaults to a fresh one."""

    @abc.abstractmethod
    async def get(self, thread_id: str) -> ThreadRecord | None:
        """The live thread, or ``None`` when unknown or soft-deleted."""

    @abc.abstractmethod
    async def list(
        self,
        owner_id: str,
        *,
        agent: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ThreadRecord]:
        """The owner's live threads, newest first; ``agent`` narrows the list."""

    @abc.abstractmethod
    async def soft_delete(self, thread_id: str) -> bool:
        """Stamp ``deleted_at``; ``False`` when the thread is unknown or already gone."""


class InMemoryThreadStore(ThreadStore):
    """Dict-backed store — ``vdreamer dev`` (D11) and the tests."""

    def __init__(self) -> None:
        self._threads: dict[str, ThreadRecord] = {}
        self._lock = asyncio.Lock()
        self._tick = 0

    def _now(self) -> float:
        # Strictly increasing even within one clock tick, so "newest first" is
        # deterministic for threads created back to back.
        self._tick += 1
        return time.time() + self._tick * 1e-6

    async def create(
        self,
        agent: str,
        owner_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> ThreadRecord:
        async with self._lock:
            tid = thread_id if thread_id is not None else new_thread_id()
            if tid in self._threads:
                raise ValueError(f"thread {tid!r} already exists")
            record = ThreadRecord(
                id=tid,
                agent=agent,
                owner_id=owner_id,
                created_at=self._now(),
                metadata=dict(metadata or {}),
            )
            self._threads[tid] = record
            return record

    async def get(self, thread_id: str) -> ThreadRecord | None:
        async with self._lock:
            record = self._threads.get(thread_id)
            return record if record is not None and record.deleted_at is None else None

    async def list(
        self,
        owner_id: str,
        *,
        agent: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ThreadRecord]:
        async with self._lock:
            live = sorted(
                (
                    record
                    for record in self._threads.values()
                    if record.owner_id == owner_id
                    and record.deleted_at is None
                    and (agent is None or record.agent == agent)
                ),
                key=lambda record: (record.created_at, record.id),
                reverse=True,
            )
            return live[offset : offset + limit]

    async def soft_delete(self, thread_id: str) -> bool:
        async with self._lock:
            record = self._threads.get(thread_id)
            if record is None or record.deleted_at is not None:
                return False
            self._threads[thread_id] = replace(record, deleted_at=self._now())
            return True


#: The ``threads`` table (§2.5). Idempotent; ``vdreamer migrate`` applies it.
THREADS_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS threads (
    id          text        PRIMARY KEY,
    agent       text        NOT NULL,
    owner_id    text        NOT NULL,
    metadata    jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    deleted_at  timestamptz
);

CREATE INDEX IF NOT EXISTS threads_owner_live_idx
    ON threads (owner_id, created_at DESC)
    WHERE deleted_at IS NULL
"""

_THREAD_COLUMNS: Final[str] = "id, agent, owner_id, metadata, created_at, deleted_at"

_CREATE_SQL: Final[str] = f"""
INSERT INTO threads (id, agent, owner_id, metadata)
VALUES (%s, %s, %s, %s::jsonb)
RETURNING {_THREAD_COLUMNS}
"""

_GET_SQL: Final[str] = f"""
SELECT {_THREAD_COLUMNS}
FROM threads
WHERE id = %s AND deleted_at IS NULL
"""

_LIST_SQL: Final[str] = f"""
SELECT {_THREAD_COLUMNS}
FROM threads
WHERE owner_id = %s
  AND deleted_at IS NULL
  AND (%s::text IS NULL OR agent = %s::text)
ORDER BY created_at DESC, id DESC
LIMIT %s OFFSET %s
"""

_SOFT_DELETE_SQL: Final[str] = """
UPDATE threads
SET deleted_at = now()
WHERE id = %s AND deleted_at IS NULL
"""


class PostgresThreadStore(ThreadStore):
    """The ``threads`` table — short-lived autocommit connections, like the queue."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    @asynccontextmanager
    async def _conn(self) -> AsyncGenerator[Any]:
        conn_cls = _load_async_connection_class()
        conn = await conn_cls.connect(self._dsn, autocommit=True)
        try:
            yield conn
        finally:
            await conn.close()

    @staticmethod
    def _row_to_record(row: Any) -> ThreadRecord:
        return ThreadRecord(
            id=str(row[0]),
            agent=str(row[1]),
            owner_id=str(row[2]),
            metadata=dict(row[3] or {}),
            created_at=row[4].timestamp(),
            deleted_at=row[5].timestamp() if row[5] is not None else None,
        )

    async def setup(self) -> None:
        """Create the table if missing (``migrate`` reuses :data:`THREADS_DDL`)."""
        async with self._conn() as conn:
            await conn.execute(THREADS_DDL)

    async def create(
        self,
        agent: str,
        owner_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> ThreadRecord:
        tid = thread_id if thread_id is not None else new_thread_id()
        async with self._conn() as conn:
            cur = await conn.execute(
                _CREATE_SQL, (tid, agent, owner_id, json.dumps(dict(metadata or {})))
            )
            row = await cur.fetchone()
        if row is None:  # pragma: no cover - INSERT..RETURNING always yields a row
            raise RuntimeError("threads insert returned no row")
        return self._row_to_record(row)

    async def get(self, thread_id: str) -> ThreadRecord | None:
        async with self._conn() as conn:
            cur = await conn.execute(_GET_SQL, (thread_id,))
            row = await cur.fetchone()
        return self._row_to_record(row) if row is not None else None

    async def list(
        self,
        owner_id: str,
        *,
        agent: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ThreadRecord]:
        async with self._conn() as conn:
            cur = await conn.execute(_LIST_SQL, (owner_id, agent, agent, limit, offset))
            rows = await cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def soft_delete(self, thread_id: str) -> bool:
        async with self._conn() as conn:
            cur = await conn.execute(_SOFT_DELETE_SQL, (thread_id,))
            return bool(cur.rowcount)
