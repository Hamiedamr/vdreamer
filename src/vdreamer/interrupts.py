"""The interrupt record (PRD §4.2) — the approver inbox behind ``/interrupts``.

Every pending LangGraph interrupt of a run becomes one record (A44: one
record, one event, one endpoint, whatever raised it):

``interrupts(interrupt_id, run_id, kind, scope, approver_role, actions[],
payload, status: pending|resolved|expired|cancelled, version, expires_at,
resolved_by, resolution)``

plus the columns the gateway needs to serve the inbox without a join:
``thread_id``, ``agent``, ``edit_schema``, ``initiator_id`` (authority defaults
to the initiator, §4.3) and the timestamps.

Who writes what:

- the **worker** creates the record when its stream driver emits
  ``interrupt.requested`` (:func:`record_from_event`); ``create`` is
  idempotent on ``interrupt_id`` — LangGraph's own interrupt id — because a
  recovered attempt re-raises the same interrupt;
- the **gateway** resolves it (``POST /interrupts/{id}/resolve``) with a
  compare-and-swap on ``version`` (A12: the first resolver wins, the second
  gets ``409``), then plants the resume map on the run and requeues it.

Milestone 4 scope: the record, the store, and the resolve route's version /
expiry / role / action checks. ``edit_schema`` validation, four-eyes
(``HITL_STRICT_SCOPES``, A32), the expiry sweeper, headless ``reject →
rejected``, and the non-``require_approval`` sources' policies
(``interrupt_policies``) are Milestone 8.
"""

from __future__ import annotations

import abc
import asyncio
import enum
import json
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Final

from vdreamer.eventlog import json_text
from vdreamer.events import EventEnvelope, EventType
from vdreamer.queue import _load_async_connection_class


class InterruptStatus(enum.StrEnum):
    """The §4.2 record statuses."""

    PENDING = "pending"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class InterruptRecord:
    """One ``interrupts`` row (§4.2). Timestamps are epoch seconds (wall clock)."""

    interrupt_id: str
    run_id: str
    thread_id: str
    agent: str
    kind: str
    actions: tuple[str, ...]
    payload: Mapping[str, Any] = field(default_factory=dict)
    scope: str | None = None
    approver_role: str | None = None
    edit_schema: Mapping[str, Any] | None = None
    status: InterruptStatus = InterruptStatus.PENDING
    version: int = 1
    expires_at: float | None = None
    initiator_id: str | None = None
    created_at: float = 0.0
    resolved_by: str | None = None
    resolution: Any = None
    resolved_at: float | None = None

    def is_expired(self, now: float) -> bool:
        """Past ``expires_at`` — resolvable no more, whatever the sweeper has done (§4.4)."""
        return self.expires_at is not None and now >= self.expires_at


class InterruptConflict(Exception):
    """The compare-and-swap lost: stale ``expected_version`` or no longer pending (A12)."""

    def __init__(self, interrupt_id: str) -> None:
        super().__init__(f"interrupt {interrupt_id!r} changed under the resolver")
        self.interrupt_id = interrupt_id


def _parse_ts(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def record_from_event(
    envelope: EventEnvelope,
    *,
    initiator_id: str | None,
    now: float | None = None,
) -> InterruptRecord:
    """The record for one ``interrupt.requested`` envelope (§4.2).

    The payload is the driver's normalized ``InterruptRequestedPayload``
    (``streaming.normalize_interrupt``, A44), so every source maps the same.
    """
    if envelope.type is not EventType.INTERRUPT_REQUESTED:
        raise ValueError(f"not an interrupt.requested envelope: {envelope.type}")
    data = envelope.payload
    edit_schema = data.get("edit_schema")
    payload = data.get("payload")
    return InterruptRecord(
        interrupt_id=str(data["interrupt_id"]),
        run_id=envelope.run_id,
        thread_id=envelope.thread_id,
        agent=envelope.agent,
        kind=str(data.get("kind", "generic")),
        actions=tuple(str(action) for action in data.get("actions") or ("respond",)),
        payload=dict(payload) if isinstance(payload, Mapping) else {},
        scope=data.get("scope") if isinstance(data.get("scope"), str) else None,
        approver_role=(
            data.get("approver_role") if isinstance(data.get("approver_role"), str) else None
        ),
        edit_schema=dict(edit_schema) if isinstance(edit_schema, Mapping) else None,
        expires_at=_parse_ts(data.get("expires_at")),
        initiator_id=initiator_id,
        created_at=now if now is not None else time.time(),
    )


@dataclass(frozen=True)
class Viewer:
    """Who is looking at the inbox: the caller's id and roles (D13)."""

    user_id: str
    roles: tuple[str, ...] = ()


def visible_to(record: InterruptRecord, viewer: Viewer) -> bool:
    """§4.3 visibility: the approver role's holders, plus the run's initiator.

    (Authority to *resolve* is narrower: with an ``approver_role`` set, the
    initiator may look but not resolve — the gateway's check.)
    """
    if record.approver_role is not None and record.approver_role in viewer.roles:
        return True
    return record.initiator_id is not None and record.initiator_id == viewer.user_id


_visible = visible_to  # the stores' ``visible_to`` parameter shadows the function


class InterruptStore(abc.ABC):
    """The §4.2 records: created by workers, resolved by the gateway."""

    @abc.abstractmethod
    async def create(self, record: InterruptRecord) -> InterruptRecord:
        """Insert ``record``; an existing ``interrupt_id`` is kept and returned (idempotent)."""

    @abc.abstractmethod
    async def get(self, interrupt_id: str) -> InterruptRecord | None:
        """The record, or ``None``."""

    @abc.abstractmethod
    async def list(
        self,
        *,
        status: InterruptStatus | None = InterruptStatus.PENDING,
        run_id: str | None = None,
        visible_to: Viewer | None = None,
        limit: int | None = None,
    ) -> list[InterruptRecord]:
        """Records oldest first, narrowed by ``status`` (``None`` = any) and ``run_id``.

        ``visible_to`` keeps what that viewer may see (see :func:`visible_to`);
        ``limit`` caps the page. Both are pushed into the query, so the inbox
        never reads other users' history.
        """

    @abc.abstractmethod
    async def resolve(
        self,
        interrupt_id: str,
        *,
        expected_version: int,
        resolved_by: str,
        resolution: Any,
    ) -> InterruptRecord:
        """``pending → resolved`` iff ``version == expected_version``; bumps ``version``.

        Raises:
            InterruptConflict: unknown, not pending, or a stale version (A12).
        """

    @abc.abstractmethod
    async def cancel_for_run(self, run_id: str) -> int:
        """Mark the run's pending interrupts ``cancelled`` (the run was cancelled)."""


class InMemoryInterruptStore(InterruptStore):
    """Dict-backed store — ``vdreamer dev`` (D11) and the tests."""

    def __init__(self) -> None:
        self._records: dict[str, InterruptRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, record: InterruptRecord) -> InterruptRecord:
        async with self._lock:
            existing = self._records.get(record.interrupt_id)
            if existing is not None:
                return existing
            self._records[record.interrupt_id] = record
            return record

    async def get(self, interrupt_id: str) -> InterruptRecord | None:
        async with self._lock:
            return self._records.get(interrupt_id)

    async def list(
        self,
        *,
        status: InterruptStatus | None = InterruptStatus.PENDING,
        run_id: str | None = None,
        visible_to: Viewer | None = None,
        limit: int | None = None,
    ) -> list[InterruptRecord]:
        viewer = visible_to
        async with self._lock:
            records = sorted(
                (
                    record
                    for record in self._records.values()
                    if (status is None or record.status is status)
                    and (run_id is None or record.run_id == run_id)
                    and (viewer is None or _visible(record, viewer))
                ),
                key=lambda record: (record.created_at, record.interrupt_id),
            )
            return records if limit is None else records[:limit]

    async def resolve(
        self,
        interrupt_id: str,
        *,
        expected_version: int,
        resolved_by: str,
        resolution: Any,
    ) -> InterruptRecord:
        async with self._lock:
            record = self._records.get(interrupt_id)
            if (
                record is None
                or record.status is not InterruptStatus.PENDING
                or record.version != expected_version
            ):
                raise InterruptConflict(interrupt_id)
            resolved = replace(
                record,
                status=InterruptStatus.RESOLVED,
                version=record.version + 1,
                resolved_by=resolved_by,
                resolution=resolution,
                resolved_at=time.time(),
            )
            self._records[interrupt_id] = resolved
            return resolved

    async def cancel_for_run(self, run_id: str) -> int:
        async with self._lock:
            cancelled = 0
            for record in list(self._records.values()):
                if record.run_id == run_id and record.status is InterruptStatus.PENDING:
                    self._records[record.interrupt_id] = replace(
                        record, status=InterruptStatus.CANCELLED, version=record.version + 1
                    )
                    cancelled += 1
            return cancelled


#: The ``interrupts`` table (§4.2). Idempotent; ``vdreamer migrate`` applies it.
INTERRUPTS_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS interrupts (
    interrupt_id  text        PRIMARY KEY,
    run_id        text        NOT NULL,
    thread_id     text        NOT NULL,
    agent         text        NOT NULL,
    kind          text        NOT NULL,
    scope         text,
    approver_role text,
    actions       jsonb       NOT NULL DEFAULT '[]'::jsonb,
    payload       jsonb       NOT NULL DEFAULT '{}'::jsonb,
    edit_schema   jsonb,
    status        text        NOT NULL DEFAULT 'pending',
    version       int         NOT NULL DEFAULT 1,
    expires_at    timestamptz,
    initiator_id  text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    resolved_by   text,
    resolution    jsonb,
    resolved_at   timestamptz
);

CREATE INDEX IF NOT EXISTS interrupts_status_idx ON interrupts (status, created_at);
CREATE INDEX IF NOT EXISTS interrupts_run_idx ON interrupts (run_id)
"""

_INTERRUPT_COLUMNS: Final[str] = (
    "interrupt_id, run_id, thread_id, agent, kind, scope, approver_role, actions, "
    "payload, edit_schema, status, version, expires_at, initiator_id, created_at, "
    "resolved_by, resolution, resolved_at"
)

#: ``ON CONFLICT DO NOTHING`` + the follow-up SELECT make ``create`` idempotent.
_CREATE_SQL: Final[str] = """
INSERT INTO interrupts (
    interrupt_id, run_id, thread_id, agent, kind, scope, approver_role, actions,
    payload, edit_schema, expires_at, initiator_id
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb,
    to_timestamp(%s::double precision), %s
)
ON CONFLICT (interrupt_id) DO NOTHING
"""

_GET_SQL: Final[str] = f"SELECT {_INTERRUPT_COLUMNS} FROM interrupts WHERE interrupt_id = %s"

#: The visibility clause mirrors :func:`visible_to`: a NULL ``approver_role``
#: never matches ``= ANY(...)``, so it falls through to the initiator test.
_LIST_SQL: Final[str] = f"""
SELECT {_INTERRUPT_COLUMNS}
FROM interrupts
WHERE (%s::text IS NULL OR status = %s::text)
  AND (%s::text IS NULL OR run_id = %s::text)
  AND (
      %s::text IS NULL
      OR approver_role = ANY(%s::text[])
      OR initiator_id = %s::text
  )
ORDER BY created_at, interrupt_id
LIMIT %s
"""

#: The compare-and-swap (A12): one UPDATE guarded by status *and* version.
_RESOLVE_SQL: Final[str] = f"""
UPDATE interrupts
SET status = 'resolved',
    version = version + 1,
    resolved_by = %s,
    resolution = %s::jsonb,
    resolved_at = now()
WHERE interrupt_id = %s AND status = 'pending' AND version = %s
RETURNING {_INTERRUPT_COLUMNS}
"""

_CANCEL_FOR_RUN_SQL: Final[str] = """
UPDATE interrupts
SET status = 'cancelled', version = version + 1
WHERE run_id = %s AND status = 'pending'
"""


def _ts(value: Any) -> float | None:
    return value.timestamp() if value is not None else None


class PostgresInterruptStore(InterruptStore):
    """The ``interrupts`` table — short-lived autocommit connections, like the queue."""

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
    def _row_to_record(row: Any) -> InterruptRecord:
        actions: Sequence[Any] = row[7] or ()
        return InterruptRecord(
            interrupt_id=str(row[0]),
            run_id=str(row[1]),
            thread_id=str(row[2]),
            agent=str(row[3]),
            kind=str(row[4]),
            scope=row[5],
            approver_role=row[6],
            actions=tuple(str(action) for action in actions),
            payload=dict(row[8] or {}),
            edit_schema=row[9],
            status=InterruptStatus(str(row[10])),
            version=int(row[11]),
            expires_at=_ts(row[12]),
            initiator_id=row[13],
            created_at=row[14].timestamp(),
            resolved_by=row[15],
            resolution=row[16],
            resolved_at=_ts(row[17]),
        )

    async def setup(self) -> None:
        """Create the table if missing (``migrate`` reuses :data:`INTERRUPTS_DDL`)."""
        async with self._conn() as conn:
            await conn.execute(INTERRUPTS_DDL)

    async def create(self, record: InterruptRecord) -> InterruptRecord:
        async with self._conn() as conn:
            await conn.execute(
                _CREATE_SQL,
                (
                    record.interrupt_id,
                    record.run_id,
                    record.thread_id,
                    record.agent,
                    record.kind,
                    record.scope,
                    record.approver_role,
                    json.dumps(list(record.actions)),
                    json_text(dict(record.payload)),
                    json_text(dict(record.edit_schema)) if record.edit_schema else None,
                    record.expires_at,
                    record.initiator_id,
                ),
            )
            cur = await conn.execute(_GET_SQL, (record.interrupt_id,))
            row = await cur.fetchone()
        if row is None:  # pragma: no cover - the row was just inserted or existed
            raise RuntimeError("interrupts insert left no row")
        return self._row_to_record(row)

    async def get(self, interrupt_id: str) -> InterruptRecord | None:
        async with self._conn() as conn:
            cur = await conn.execute(_GET_SQL, (interrupt_id,))
            row = await cur.fetchone()
        return self._row_to_record(row) if row is not None else None

    async def list(
        self,
        *,
        status: InterruptStatus | None = InterruptStatus.PENDING,
        run_id: str | None = None,
        visible_to: Viewer | None = None,
        limit: int | None = None,
    ) -> list[InterruptRecord]:
        status_value = status.value if status is not None else None
        user_id = visible_to.user_id if visible_to is not None else None
        roles = list(visible_to.roles) if visible_to is not None else []
        params = (status_value, status_value, run_id, run_id, user_id, roles, user_id, limit)
        async with self._conn() as conn:
            cur = await conn.execute(_LIST_SQL, params)
            rows = await cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def resolve(
        self,
        interrupt_id: str,
        *,
        expected_version: int,
        resolved_by: str,
        resolution: Any,
    ) -> InterruptRecord:
        async with self._conn() as conn:
            cur = await conn.execute(
                _RESOLVE_SQL,
                (resolved_by, json_text(resolution), interrupt_id, expected_version),
            )
            row = await cur.fetchone()
        if row is None:
            raise InterruptConflict(interrupt_id)
        return self._row_to_record(row)

    async def cancel_for_run(self, run_id: str) -> int:
        async with self._conn() as conn:
            cur = await conn.execute(_CANCEL_FOR_RUN_SQL, (run_id,))
            return int(cur.rowcount)
