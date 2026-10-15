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

Milestone 8 adds the policy around the record:

- :func:`apply_interrupt_policy` — ``BaseAgent.interrupt_policies`` attach
  ``approver_role`` / ``scope`` / TTL to interrupts raised outside
  ``require_approval`` (§4.5); the worker applies it before the record and the
  event exist, so both carry the same policy;
- :func:`is_strict_scope` — four-eyes (``HITL_STRICT_SCOPES``, A32);
- :class:`InterruptSweeper` — the leader-elected expiry sweeper (§4.4): an
  overdue interrupt → ``expired``, its run → ``expired`` (the thread unlocks);
- :func:`end_waiting_run` — how a run no worker holds is ended (headless
  ``reject → rejected`` by the gateway, expiry by the sweeper).
"""

from __future__ import annotations

import abc
import asyncio
import builtins
import contextlib
import enum
import fnmatch
import json
import logging
import time
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol

from vdreamer.eventlog import EventSink, json_text
from vdreamer.events import ErrorInfo, EventEnvelope, EventType, RunStatusPayload
from vdreamer.queue import (
    InvalidTransition,
    RunQueue,
    RunRecord,
    RunState,
    _load_async_connection_class,
)
from vdreamer.tools import InterruptPolicy

_log = logging.getLogger(__name__)


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

#: How many overdue interrupts one ``expire_overdue`` call expires at most.
EXPIRY_BATCH: Final[int] = 100


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

    @abc.abstractmethod
    async def expire_overdue(
        self, now: float | None = None, *, limit: int = EXPIRY_BATCH
    ) -> builtins.list[InterruptRecord]:
        """``pending → expired`` for up to ``limit`` records past ``expires_at`` (§4.4).

        Bumps ``version`` (a resolver holding the old one gets ``409``/``410``)
        and returns the expired records, oldest deadline first.
        """


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

    async def expire_overdue(
        self, now: float | None = None, *, limit: int = EXPIRY_BATCH
    ) -> builtins.list[InterruptRecord]:
        at = now if now is not None else time.time()
        async with self._lock:
            overdue = sorted(
                (
                    record
                    for record in self._records.values()
                    if record.status is InterruptStatus.PENDING and record.is_expired(at)
                ),
                key=lambda record: (record.expires_at or 0.0, record.interrupt_id),
            )[:limit]
            expired = [
                replace(record, status=InterruptStatus.EXPIRED, version=record.version + 1)
                for record in overdue
            ]
            for record in expired:
                self._records[record.interrupt_id] = record
            return expired


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

#: §4.4: one batch of overdue interrupts → ``expired``. ``SKIP LOCKED`` leaves a
#: row a resolver is updating to that resolver's compare-and-swap.
_EXPIRE_OVERDUE_SQL: Final[str] = f"""
UPDATE interrupts
SET status = 'expired', version = version + 1
WHERE interrupt_id IN (
    SELECT interrupt_id
    FROM interrupts
    WHERE status = 'pending' AND expires_at <= to_timestamp(%s::double precision)
    ORDER BY expires_at, interrupt_id
    LIMIT %s
    FOR UPDATE SKIP LOCKED
)
RETURNING {_INTERRUPT_COLUMNS}
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

    async def expire_overdue(
        self, now: float | None = None, *, limit: int = EXPIRY_BATCH
    ) -> builtins.list[InterruptRecord]:
        at = now if now is not None else time.time()
        async with self._conn() as conn:
            cur = await conn.execute(_EXPIRE_OVERDUE_SQL, (at, limit))
            rows = await cur.fetchall()
        records = [self._row_to_record(row) for row in rows]
        return sorted(records, key=lambda record: (record.expires_at or 0.0, record.interrupt_id))


# ---------------------------------------------------------------------------
# Policy: interrupt_policies (§4.5) and four-eyes (A32)
# ---------------------------------------------------------------------------

#: The §4.5 kinds raised outside ``require_approval`` — what ``interrupt_policies`` cover.
TOOL_REVIEW_KIND: Final[str] = "tool_review"
MCP_ELICITATION_KIND: Final[str] = "mcp_elicitation"
GENERIC_KIND: Final[str] = "generic"
POLICY_KINDS: Final[frozenset[str]] = frozenset(
    {TOOL_REVIEW_KIND, MCP_ELICITATION_KIND, GENERIC_KIND}
)


def interrupt_tool_names(kind: str, payload: Mapping[str, Any]) -> list[str]:
    """The tools an interrupt is about: a review's actions, an elicitation's tool."""
    if kind == TOOL_REVIEW_KIND:
        requests = payload.get("action_requests")
        return [
            str(request["name"])
            for request in (requests if isinstance(requests, list) else ())
            if isinstance(request, Mapping) and request.get("name")
        ]
    if kind == MCP_ELICITATION_KIND and payload.get("tool_name"):
        return [str(payload["tool_name"])]
    return []


def _matching_policies(
    kind: str, payload: Mapping[str, Any], policies: Mapping[str, InterruptPolicy]
) -> list[InterruptPolicy]:
    """Tool-name keys (exact, else the first ``fnmatch`` glob) in request order, then the kind."""
    matched: list[InterruptPolicy] = []
    for name in interrupt_tool_names(kind, payload):
        if name in policies:
            matched.append(policies[name])
            continue
        for key, policy in policies.items():
            if key not in POLICY_KINDS and fnmatch.fnmatchcase(name, key):
                matched.append(policy)
                break
    if kind in policies:
        matched.append(policies[kind])
    return matched


def apply_interrupt_policy(
    event_payload: Mapping[str, Any],
    policies: Mapping[str, InterruptPolicy],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Attach ``interrupt_policies`` to one ``interrupt.requested`` payload (§4.5).

    Only the sources outside ``require_approval`` (:data:`POLICY_KINDS`) take
    a policy — ``require_approval`` declares its own. A tool-name key beats the
    kind key; per field, the first matching policy that sets it wins, and an
    unset field keeps the source's own value. ``ttl_hours`` restarts the
    clock: ``expires_at = now + ttl``.
    """
    updated = dict(event_payload)
    kind = str(updated.get("kind") or "")
    if kind not in POLICY_KINDS or not policies:
        return updated
    payload = updated.get("payload")
    matched = _matching_policies(kind, payload if isinstance(payload, Mapping) else {}, policies)
    approver_role = next((p.approver_role for p in matched if p.approver_role), None)
    scope = next((p.scope for p in matched if p.scope), None)
    ttl_hours = next((p.ttl_hours for p in matched if p.ttl_hours is not None), None)
    if approver_role is not None:
        updated["approver_role"] = approver_role
    if scope is not None:
        updated["scope"] = scope
    if ttl_hours is not None:
        start = datetime.fromtimestamp(now if now is not None else time.time(), UTC)
        updated["expires_at"] = (start + timedelta(hours=ttl_hours)).isoformat()
    return updated


def is_strict_scope(scope: str | None, strict_scopes: Sequence[str]) -> bool:
    """A32: ``scope`` matches a ``HITL_STRICT_SCOPES`` pattern (``fnmatch``) → four-eyes."""
    return scope is not None and any(
        fnmatch.fnmatchcase(scope, pattern) for pattern in strict_scopes
    )


# ---------------------------------------------------------------------------
# Ending a run no worker holds; the expiry sweeper (§4.3, §4.4)
# ---------------------------------------------------------------------------


async def end_waiting_run(
    queue: RunQueue,
    interrupts: InterruptStore,
    writer: EventSink,
    run: RunRecord,
    status: RunState,
    *,
    error: ErrorInfo | None = None,
) -> bool:
    """Move an ``awaiting_approval`` run to a terminal ``status`` and log it.

    No worker holds a waiting run (approvals free the worker, §3.2), so
    whoever ends it — the gateway on a headless ``reject``, the sweeper on
    expiry — writes its final ``run.status``; its other pending interrupts are
    ``cancelled``. ``False`` when the run already left ``awaiting_approval``
    (a concurrent cancel or resume won): nothing is written then.
    """
    try:
        await queue.finish(run.id, status)
    except InvalidTransition:
        return False
    await interrupts.cancel_for_run(run.id)
    await writer.write(
        EventEnvelope(
            type=EventType.RUN_STATUS,
            run_id=run.id,
            thread_id=run.thread_id or run.id,
            agent=run.agent,
            seq=0,  # the log allocates the real seq
            payload=RunStatusPayload(
                status=status.value, attempt=run.attempt, error=error
            ).event_payload(),
        )
    )
    return True


class Leader(Protocol):
    """Leader election for the sweeper: only the leader sweeps (§4.4)."""

    async def is_leader(self) -> bool:
        """Try to hold leadership; ``True`` while this process holds it."""
        ...

    async def close(self) -> None:
        """Give leadership up."""
        ...


class SingleProcessLeader:
    """Always the leader — ``vdreamer dev`` and the tests run one gateway (D11)."""

    async def is_leader(self) -> bool:
        return True

    async def close(self) -> None:
        return None


#: The advisory-lock key the gateway replicas' sweepers contend for.
SWEEPER_LOCK_KEY: Final[int] = 0x76_6452_5357  # "vdRSW"


class PostgresAdvisoryLeader:
    """Leadership = a session-level ``pg_try_advisory_lock`` on a held connection.

    Every gateway replica tries on each sweep; the one whose connection holds
    the lock sweeps. A dropped connection releases the lock server-side, so
    another replica takes over on its next try.
    """

    def __init__(self, dsn: str, key: int = SWEEPER_LOCK_KEY) -> None:
        self._dsn = dsn
        self._key = key
        self._conn: Any = None

    async def is_leader(self) -> bool:
        if self._conn is not None:
            try:
                await self._conn.execute("SELECT 1")
                return True
            except Exception:  # the connection, and with it the lock, is gone
                await self.close()
        try:
            conn = await _load_async_connection_class().connect(self._dsn, autocommit=True)
        except Exception as error:
            _log.warning("interrupt sweeper: cannot reach Postgres (%s)", error)
            return False
        try:
            cur = await conn.execute("SELECT pg_try_advisory_lock(%s)", (self._key,))
            row = await cur.fetchone()
        except Exception:
            await conn.close()
            raise
        if row is not None and row[0]:
            self._conn = conn
            return True
        await conn.close()
        return False

    async def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            with contextlib.suppress(Exception):
                await conn.close()


#: How often the sweeper looks for overdue interrupts.
SWEEP_INTERVAL_S: Final[float] = 30.0

#: ``run.status.error.code`` of a run whose interrupt lapsed.
INTERRUPT_EXPIRED_CODE: Final[str] = "interrupt_expired"


class InterruptSweeper:
    """The leader-elected expiry sweeper (§4.4).

    Each sweep (leader only): overdue ``pending`` interrupts → ``expired``;
    each one's run, still ``awaiting_approval``, → ``expired`` with a final
    ``run.status`` (``error.code = "interrupt_expired"``), which unlocks the
    thread. The resolve route answers ``410`` for an overdue interrupt even
    before a sweep reaches it.
    """

    def __init__(
        self,
        interrupts: InterruptStore,
        queue: RunQueue,
        writer: EventSink,
        *,
        leader: Leader | None = None,
        interval_s: float = SWEEP_INTERVAL_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._interrupts = interrupts
        self._queue = queue
        self._writer = writer
        self._leader: Leader = leader if leader is not None else SingleProcessLeader()
        self._interval_s = interval_s
        self._clock = clock

    async def sweep_once(self) -> list[InterruptRecord]:
        """One sweep; returns the interrupts it expired (none when not the leader)."""
        if not await self._leader.is_leader():
            return []
        expired: list[InterruptRecord] = []
        while True:
            batch = await self._interrupts.expire_overdue(self._clock())
            expired += batch
            if len(batch) < EXPIRY_BATCH:
                break
        for run_id in dict.fromkeys(record.run_id for record in expired):
            run = await self._queue.get(run_id)
            if run is None or run.status is not RunState.AWAITING_APPROVAL:
                continue
            ended = await end_waiting_run(
                self._queue,
                self._interrupts,
                self._writer,
                run,
                RunState.EXPIRED,
                error=ErrorInfo(
                    code=INTERRUPT_EXPIRED_CODE,
                    message="an approval was not resolved before it expired",
                ),
            )
            if ended:
                _log.info("run %s expired: its interrupt lapsed (§4.4)", run_id)
        return expired

    async def run(self, stop: asyncio.Event) -> None:
        """Sweep every ``interval_s`` until ``stop`` is set; leadership is released at exit."""
        try:
            while not stop.is_set():
                try:
                    await self.sweep_once()
                except Exception:  # a failed sweep is retried next interval
                    _log.exception("interrupt sweep failed")
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), self._interval_s)
        finally:
            await self._leader.close()
