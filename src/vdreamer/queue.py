"""The run queue: the ``runs`` table, the §3.2 lifecycle, claims and leases.

Milestone 3 (WP1). PRD §3.2 is the whole spec:

- lifecycle ``queued → running → (awaiting_approval → running)* → completed |
  failed | rejected | expired | cancelled`` (``RunState`` and
  ``_ALLOWED_TRANSITIONS``),
- claiming via ``FOR UPDATE SKIP LOCKED`` with a 10 s lease heartbeat
  (``LEASE_INTERVAL_S``),
- crash recovery: an expired lease requeues the run — at most ``MAX_ATTEMPTS``
  attempts — and the in-flight node re-executes from the checkpoint (A6),
- cancel and approvals: ``awaiting_approval`` holds no lease (approvals free
  the worker); ``cancel`` marks the row while WP2's ``RunControl`` drains the
  live worker.

Postgres is the only gateway↔worker transport (A7): gateway and worker meet
on the same ``runs`` rows, never over a direct API. ``InMemoryRunQueue``
backs the tests and ``vdreamer dev`` (D11); ``PostgresRunQueue`` is the
production queue and mirrors ``PostgresEventLog``'s lazy-psycopg seam.

Consumers: the worker loop (WP2), the gateway's ``409 {active_run_id,
reason}`` render (M4), and ``migrate`` (WP3), which reuses :data:`RUNS_DDL`.
"""

from __future__ import annotations

import abc
import asyncio
import enum
import json
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, Final, TypeAlias

from vdreamer.persistence import MissingPostgresDependencyError


#: The run lifecycle (PRD §3.2):
#: ``queued → running → (awaiting_approval → running)* → completed | failed |
#: rejected | expired | cancelled``.
class RunState(enum.StrEnum):
    """One run status; the members are exactly PRD §3.2's states."""

    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


#: The statuses that occupy a thread — the one-active-run-per-thread guard
#: (D12/A11) and the partial index in :data:`RUNS_DDL` cover exactly these.
ACTIVE_STATES: Final[frozenset[RunState]] = frozenset(
    {RunState.QUEUED, RunState.RUNNING, RunState.AWAITING_APPROVAL}
)

#: The subset ``?wait_for_idle=true`` queues behind (PRD §2.5: "queues behind
#: ``running`` only"). A run already ``queued`` ahead is nothing to wait on.
WAITABLE_STATES: Final[frozenset[RunState]] = frozenset(
    {RunState.RUNNING, RunState.AWAITING_APPROVAL}
)

#: The final statuses — no transition leaves them (PRD §3.2).
TERMINAL_STATES: Final[frozenset[RunState]] = frozenset(
    {
        RunState.COMPLETED,
        RunState.FAILED,
        RunState.REJECTED,
        RunState.EXPIRED,
        RunState.CANCELLED,
    }
)

#: Lease window for one claim; the worker heartbeats inside it (PRD §3.2).
LEASE_INTERVAL_S: Final[float] = 10.0

#: Recovery ceiling: ``requeue_expired`` expires a run whose ``attempt`` has
#: reached this (PRD §3.2: "requeue with attempt+1 (max 3)").
MAX_ATTEMPTS: Final[int] = 3

#: The §3.2 lifecycle as a from→allowed-targets map; terminal states map to
#: nothing (they are final). ``rejected`` and ``expired`` are reachable from
#: ``running`` because §3.2's terminal line follows the
#: ``(awaiting_approval → running)*`` loop (``requeue_expired`` expires an
#: exhausted run), and from ``awaiting_approval`` because no worker holds a
#: waiting run: the gateway rejects it on a headless ``reject`` (§4.3) and the
#: expiry sweeper expires it once its interrupt lapses (§4.4).
#: ``running → queued`` is deliberately absent: returning a lost run to the
#: queue is the recovery sweep's *requeue* (the PRD's own word), not a
#: lifecycle step — see ``RunQueue.requeue_expired``.
_ALLOWED_TRANSITIONS: Final[Mapping[RunState, frozenset[RunState]]] = {
    RunState.QUEUED: frozenset({RunState.RUNNING, RunState.CANCELLED}),
    RunState.RUNNING: frozenset(
        {
            RunState.AWAITING_APPROVAL,
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.REJECTED,
            RunState.EXPIRED,
            RunState.CANCELLED,
        }
    ),
    RunState.AWAITING_APPROVAL: frozenset(
        {RunState.QUEUED, RunState.REJECTED, RunState.EXPIRED, RunState.CANCELLED}
    ),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.REJECTED: frozenset(),
    RunState.EXPIRED: frozenset(),
    RunState.CANCELLED: frozenset(),
}


class ActiveRunConflict(Exception):
    """The thread already has an active run and the caller did not queue behind.

    D12/A11: the gateway renders this as ``409 {active_run_id, reason}``
    (PRD §2.5).
    """

    def __init__(self, active_run_id: str, reason: str = "thread_has_active_run") -> None:
        super().__init__(f"thread already has active run {active_run_id!r} ({reason})")
        self.active_run_id = active_run_id
        self.reason = reason


class InvalidTransition(Exception):
    """A run-status change outside the §3.2 lifecycle map.

    ``from_state`` is ``None`` when the source state could not be determined
    (the Postgres guard matched no row: run missing or not in the required
    state).
    """

    def __init__(self, run_id: str, from_state: RunState | None, to_state: RunState) -> None:
        source = from_state.value if from_state is not None else "<unknown>"
        super().__init__(f"run {run_id}: cannot transition {source} -> {to_state.value}")
        self.run_id = run_id
        self.from_state = from_state
        self.to_state = to_state


#: The queue's clock: epoch-seconds-like floats, injectable for tests.
Clock: TypeAlias = Callable[[], float]


@dataclass(frozen=True)
class RunRecord:
    """One row of ``runs`` (PRD §3.2).

    Timestamps are plain floats from the backend's clock — the in-memory
    queue's injectable clock, epoch seconds for Postgres.

    The Milestone-4 columns carry what the gateway knows at enqueue time:
    ``user_id`` / ``roles`` are the initiator's identity (``X-User-Id`` /
    ``X-User-Roles``, D13) — the worker binds them into the run's
    ``VDreamerContext`` (§2.7) and approval authority defaults to the
    initiator (§4.3); ``config`` is the validated ``{input, config?}`` run
    config (§2.5); ``resume`` is the resume map ``{interrupt_id: value}`` an
    interrupt resolution plants before requeueing (§3.2: "resolution enqueues
    a resume") — the worker turns it into ``Command(resume=...)``.
    """

    id: str
    thread_id: str | None
    agent: str
    input: Any
    status: RunState
    attempt: int
    claimed_by: str | None
    lease_expires_at: float | None
    created_at: float
    updated_at: float
    user_id: str | None = None
    roles: tuple[str, ...] = ()
    config: Mapping[str, Any] | None = None
    resume: Mapping[str, Any] | None = None


def _new_run_id() -> str:
    """A fresh run id — one execution is one ``run_id`` (PRD §3.1)."""
    return f"run_{uuid.uuid4().hex}"


class RunQueue(abc.ABC):
    """The queue operations the worker (WP2) and the gateway (M4) need.

    Every method is a single async step; backends implement the atomicity
    (the in-memory queue holds an asyncio lock, Postgres relies on row locks
    and transactions).
    """

    @abc.abstractmethod
    async def enqueue(
        self,
        agent: str,
        input: Any,
        *,
        thread_id: str | None,
        run_id: str | None,
        wait_for_idle: bool = False,
        user_id: str | None = None,
        roles: Sequence[str] = (),
        config: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        """Create a ``queued`` run and return its record.

        ``run_id`` may be supplied (the gateway does, for ``/trigger``
        replay, A18); ``None`` generates one. ``user_id`` / ``roles`` /
        ``config`` are stored on the row as-is (see :class:`RunRecord`).
        The one-active-run-per-thread
        guard (D12/A11): when the thread already has a run in an active
        state, the call conflicts — unless ``wait_for_idle`` is set *and*
        every active run of the thread is :data:`WAITABLE_STATES`
        (``running``/``awaiting_approval``), in which case the new run is
        still created ``queued`` and simply waits its turn (a ``queued`` run
        ahead is nothing to wait on and still conflicts). A ``None``
        ``thread_id`` has no thread to guard against.

        Raises:
            ActiveRunConflict: the thread is busy and the caller did not
                wait (the gateway renders 409, PRD §2.5).
        """

    @abc.abstractmethod
    async def claim(self, claimant: str, *, agent: str | None = None) -> RunRecord | None:
        """Atomically start the oldest claimable ``queued`` run, or ``None``.

        A run is claimable when its thread has no run in ``running`` or
        ``awaiting_approval`` — that is what makes a ``wait_for_idle`` run
        wait. ``agent`` restricts the claim to that agent's runs: one worker
        serves one agent (§2.1), and every agent's workers share the one
        ``runs`` table, so a worker must never claim another agent's run.
        ``None`` claims any agent's run (single-agent tests). The claim
        stamps ``claimed_by``, sets
        ``lease_expires_at = now + LEASE_INTERVAL_S``, and increments
        ``attempt``.
        """

    @abc.abstractmethod
    async def heartbeat(self, run_id: str) -> bool:
        """Extend a running run's lease by :data:`LEASE_INTERVAL_S`.

        ``False`` means the run is gone or no longer ``running`` — the
        worker must drain (WP2) instead of continuing blind.
        """

    @abc.abstractmethod
    async def finish(self, run_id: str, status: RunState) -> None:
        """Move a run to a terminal status, clearing its lease and claim.

        ``status`` must be terminal: ``completed``/``failed`` from
        ``running``, ``rejected``/``expired`` from ``running`` or
        ``awaiting_approval`` (the gateway's headless reject and the expiry
        sweeper, §4.3–§4.4), or ``cancelled`` from any active state (the
        ``cancel`` method is the friendlier spelling of the latter).

        Raises:
            InvalidTransition: the run is missing or its current state does
                not allow the target.
        """

    @abc.abstractmethod
    async def suspend(self, run_id: str) -> None:
        """``running → awaiting_approval``, clearing the lease.

        Approvals free the worker (PRD §3.2) — nothing is held while the run
        waits for a human.

        Raises:
            InvalidTransition: the run is not ``running``.
        """

    @abc.abstractmethod
    async def resume(self, run_id: str, resume: Mapping[str, Any] | None = None) -> None:
        """``awaiting_approval → queued`` (resolution enqueues a resume).

        The stale claim is cleared; the next claim stamps a fresh one.
        ``resume`` — the resume map ``{interrupt_id: value}`` the gateway built
        from the resolutions (§4.3) — is stored on the row for the worker's
        ``Command(resume=...)``; ``None`` keeps whatever the row carries.

        Raises:
            InvalidTransition: the run is not ``awaiting_approval``.
        """

    @abc.abstractmethod
    async def cancel(self, run_id: str) -> RunState | None:
        """Mark a run ``cancelled``; return its prior status, ``None`` when not active.

        Any active state (``queued``, ``running``, ``awaiting_approval``)
        may be cancelled. The prior status tells the caller who owes the
        run's final ``run.status`` event: a ``running`` run's worker notices
        on its next heartbeat and drains (WP2's ``RunControl``), emitting it;
        a ``queued`` or ``awaiting_approval`` run has no worker, so the
        gateway emits it (M4).
        """

    @abc.abstractmethod
    async def list_for_thread(self, thread_id: str) -> list[RunRecord]:
        """Every run of ``thread_id``, oldest first (``created_at``, ``id``).

        The gateway's thread views (§2.5): ``GET /threads/{id}`` lists them and
        ``GET /threads/{id}/events`` joins the active — else the latest — one.
        """

    @abc.abstractmethod
    async def count_by_status(self) -> dict[RunState, int]:
        """Run counts per status — the ``queue depth`` metric (§5.4).

        Statuses with no runs are omitted.
        """

    @abc.abstractmethod
    async def release(self, run_id: str) -> None:
        """``running → queued``: the SIGTERM drain path, attempt not consumed.

        PRD §3.2: "On worker SIGTERM, every active run is drained and requeued
        without consuming an attempt." The claim that started the current
        attempt is given back — ``attempt`` is decremented to its pre-claim
        value and the lease and claim are cleared — so the next claim
        re-increments to the same number and the run resumes from its
        checkpoint. Like ``requeue_expired`` this is a recovery-style requeue,
        not a §3.2 lifecycle step (``_ALLOWED_TRANSITIONS`` has no
        ``running → queued``); it is the WP2 worker's drain handler, the
        counterpart of the sweep's expired-lease requeue.

        Raises:
            InvalidTransition: the run is missing or not ``running``.
        """

    @abc.abstractmethod
    async def get(self, run_id: str) -> RunRecord | None:
        """The run's current record, or ``None`` when unknown."""

    @abc.abstractmethod
    async def requeue_expired(self) -> int:
        """Recovery sweep (PRD §3.2): handle every expired-lease run.

        Finds ``running`` runs whose ``lease_expires_at`` has lapsed. A run
        with ``attempt >= MAX_ATTEMPTS`` is dead → ``expired``; otherwise it
        returns to ``queued`` with the lease cleared — the in-flight node
        re-executes from its checkpoint on resume, so side effects use
        ``ctx.idempotency_key`` (A6). ``attempt`` is unchanged here; it
        increments on claim. Returns how many runs were handled.
        """


class InMemoryRunQueue(RunQueue):
    """Dict-backed queue behind an asyncio lock — tests and ``vdreamer dev`` (D11).

    The ``clock`` callable (default ``time.monotonic``) produces every stored
    timestamp (``created_at``/``updated_at``/``lease_expires_at``), so tests
    drive lease expiry deterministically instead of sleeping.
    """

    def __init__(self, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        self._runs: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    def _transition(self, run: RunRecord, to: RunState, now: float) -> RunRecord:
        """Apply one §3.2 lifecycle step, raising ``InvalidTransition`` otherwise."""
        if to not in _ALLOWED_TRANSITIONS[run.status]:
            raise InvalidTransition(run.id, run.status, to)
        return replace(run, status=to, updated_at=now)

    async def enqueue(
        self,
        agent: str,
        input: Any,
        *,
        thread_id: str | None,
        run_id: str | None,
        wait_for_idle: bool = False,
        user_id: str | None = None,
        roles: Sequence[str] = (),
        config: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        async with self._lock:
            now = self._clock()
            if run_id is None:
                run_id = _new_run_id()
            if run_id in self._runs:
                raise ValueError(f"run {run_id!r} already exists")
            if thread_id is not None:
                active = sorted(
                    (
                        run
                        for run in self._runs.values()
                        if run.thread_id == thread_id and run.status in ACTIVE_STATES
                    ),
                    key=lambda run: (run.created_at, run.id),
                )
                if active and not (
                    wait_for_idle and all(run.status in WAITABLE_STATES for run in active)
                ):
                    raise ActiveRunConflict(
                        active_run_id=active[0].id, reason="thread_has_active_run"
                    )
            record = RunRecord(
                id=run_id,
                thread_id=thread_id,
                agent=agent,
                input=input,
                status=RunState.QUEUED,
                attempt=0,
                claimed_by=None,
                lease_expires_at=None,
                created_at=now,
                updated_at=now,
                user_id=user_id,
                roles=tuple(roles),
                config=dict(config) if config is not None else None,
            )
            self._runs[record.id] = record
            return record

    async def claim(self, claimant: str, *, agent: str | None = None) -> RunRecord | None:
        async with self._lock:
            now = self._clock()
            # Threads with a running/awaiting run are busy: their queued runs
            # wait (this is what a wait_for_idle run queues behind). A
            # thread-less run has no thread to be blocked by or block.
            blocked = {
                run.thread_id
                for run in self._runs.values()
                if run.thread_id is not None
                and run.status in (RunState.RUNNING, RunState.AWAITING_APPROVAL)
            }
            candidates = sorted(
                (
                    run
                    for run in self._runs.values()
                    if run.status is RunState.QUEUED
                    and run.thread_id not in blocked
                    and (agent is None or run.agent == agent)
                ),
                key=lambda run: (run.created_at, run.id),
            )
            if not candidates:
                return None
            run = candidates[0]
            claimed = self._transition(run, RunState.RUNNING, now)
            claimed = replace(
                claimed,
                attempt=run.attempt + 1,
                claimed_by=claimant,
                lease_expires_at=now + LEASE_INTERVAL_S,
            )
            self._runs[claimed.id] = claimed
            return claimed

    async def heartbeat(self, run_id: str) -> bool:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.status is not RunState.RUNNING:
                return False
            now = self._clock()
            self._runs[run_id] = replace(
                run, lease_expires_at=now + LEASE_INTERVAL_S, updated_at=now
            )
            return True

    async def finish(self, run_id: str, status: RunState) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise InvalidTransition(run_id, None, status)
            updated = self._transition(run, status, self._clock())
            self._runs[run_id] = replace(updated, claimed_by=None, lease_expires_at=None)

    async def suspend(self, run_id: str) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise InvalidTransition(run_id, None, RunState.AWAITING_APPROVAL)
            updated = self._transition(run, RunState.AWAITING_APPROVAL, self._clock())
            # Approvals free the worker (§3.2): no lease while suspended.
            self._runs[run_id] = replace(updated, lease_expires_at=None)

    async def resume(self, run_id: str, resume: Mapping[str, Any] | None = None) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise InvalidTransition(run_id, None, RunState.QUEUED)
            updated = self._transition(run, RunState.QUEUED, self._clock())
            # Back to the queue: the claim is stale, the next claim restamps it.
            self._runs[run_id] = replace(
                updated,
                claimed_by=None,
                lease_expires_at=None,
                resume=dict(resume) if resume is not None else run.resume,
            )

    async def cancel(self, run_id: str) -> RunState | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.status not in ACTIVE_STATES:
                return None
            updated = self._transition(run, RunState.CANCELLED, self._clock())
            self._runs[run_id] = replace(updated, claimed_by=None, lease_expires_at=None)
            return run.status

    async def list_for_thread(self, thread_id: str) -> list[RunRecord]:
        async with self._lock:
            return sorted(
                (run for run in self._runs.values() if run.thread_id == thread_id),
                key=lambda run: (run.created_at, run.id),
            )

    async def count_by_status(self) -> dict[RunState, int]:
        async with self._lock:
            counts: dict[RunState, int] = {}
            for run in self._runs.values():
                counts[run.status] = counts.get(run.status, 0) + 1
            return counts

    async def release(self, run_id: str) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise InvalidTransition(run_id, None, RunState.QUEUED)
            if run.status is not RunState.RUNNING:
                raise InvalidTransition(run_id, run.status, RunState.QUEUED)
            # The SIGTERM drain path: the run re-enters the queue with the
            # claim's attempt increment given back (PRD §3.2, see the ABC).
            now = self._clock()
            self._runs[run_id] = replace(
                run,
                status=RunState.QUEUED,
                attempt=run.attempt - 1,
                claimed_by=None,
                lease_expires_at=None,
                updated_at=now,
            )

    async def get(self, run_id: str) -> RunRecord | None:
        async with self._lock:
            return self._runs.get(run_id)

    async def requeue_expired(self) -> int:
        async with self._lock:
            now = self._clock()
            handled = 0
            for run in list(self._runs.values()):
                if run.status is not RunState.RUNNING:
                    continue
                if run.lease_expires_at is None or run.lease_expires_at >= now:
                    continue
                handled += 1
                # The recovery sweep *requeues* (§3.2's own wording): a lost
                # lease re-enters the queue and re-executes from its
                # checkpoint, so this is not an ``_ALLOWED_TRANSITIONS`` step.
                # ``attempt`` is unchanged — it increments on claim.
                next_status = RunState.EXPIRED if run.attempt >= MAX_ATTEMPTS else RunState.QUEUED
                self._runs[run.id] = replace(
                    run,
                    status=next_status,
                    claimed_by=None,
                    lease_expires_at=None,
                    updated_at=now,
                )
            return handled


def _load_async_connection_class() -> Any:
    """Import ``psycopg.AsyncConnection`` lazily — the ``[postgres]`` extra seam.

    Same pattern as ``eventlog._load_async_connection_class`` and
    ``persistence._load_saver_class``: environments without psycopg fail with
    a friendly error at the point of use instead of an ``ImportError`` at
    import time.
    """
    try:
        from psycopg import AsyncConnection
    except ImportError as exc:
        raise MissingPostgresDependencyError(
            "psycopg is required for the vdreamer[postgres] extra: "
            "pip install 'vdreamer[postgres]'"
        ) from exc
    return AsyncConnection


#: The ``runs`` columns every RETURNING/SELECT projects, in table order.
#: Indices matter: :meth:`PostgresRunQueue._row_to_record` maps by position.
_RUN_COLUMNS: Final[str] = (
    "id, thread_id, agent, input, status, attempt, "
    "claimed_by, lease_expires_at, created_at, updated_at, "
    "user_id, roles, config, resume"
)

#: Idempotent DDL — WP3's ``migrate`` reuses this constant. The partial index
#: is the one-active-run-per-thread guard (D12/A11): it covers exactly the
#: active statuses, so the conflict check and the claim's thread-busy check
#: are index scans. The ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` lines
#: upgrade a Milestone-3 ``runs`` table in place (the M4 identity, config and
#: resume columns); on a fresh table they are no-ops.
RUNS_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS runs (
    id               text        PRIMARY KEY,
    thread_id        text        NOT NULL,
    agent            text        NOT NULL,
    input            jsonb,
    status           text        NOT NULL,
    attempt          int         NOT NULL DEFAULT 0,
    claimed_by       text,
    lease_expires_at timestamptz,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    user_id          text,
    roles            jsonb       NOT NULL DEFAULT '[]'::jsonb,
    config           jsonb,
    resume           jsonb
);

ALTER TABLE runs ADD COLUMN IF NOT EXISTS user_id text;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS roles jsonb NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS config jsonb;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS resume jsonb;

CREATE INDEX IF NOT EXISTS runs_active_thread_idx
    ON runs (thread_id)
    WHERE status IN ('queued', 'running', 'awaiting_approval');

CREATE INDEX IF NOT EXISTS runs_thread_created_idx
    ON runs (thread_id, created_at)
"""

#: Serializes every enqueue for one thread (D12/A11): a transaction-scoped
#: advisory lock keyed on the thread id, held until the enqueue transaction
#: commits. Row locks cannot do this job — with no active row there is nothing
#: to lock, so two racing enqueues would both pass the check — and ``SKIP
#: LOCKED`` would even skip an active row a concurrent claim holds, letting the
#: guard leak.
_ENQUEUE_LOCK_SQL: Final[str] = "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))"

#: The one-active-run check (D12/A11), run under :data:`_ENQUEUE_LOCK_SQL`.
_ENQUEUE_CHECK_SQL: Final[str] = """
SELECT id, status
FROM runs
WHERE thread_id = %s
  AND status IN ('queued', 'running', 'awaiting_approval')
ORDER BY created_at, id
"""

_ENQUEUE_SQL: Final[str] = f"""
INSERT INTO runs (id, thread_id, agent, input, status, attempt, user_id, roles, config)
VALUES (%s, %s, %s, %s::jsonb, 'queued', 0, %s, %s::jsonb, %s::jsonb)
RETURNING {_RUN_COLUMNS}
"""

#: The claim: one statement — pick the oldest ``queued`` run of this worker's
#: agent whose thread has no ``running``/``awaiting_approval`` run, lock it
#: (``SKIP LOCKED`` so concurrent workers never block each other), and flip it
#: to ``running`` with a fresh lease and ``attempt + 1``. A single statement is
#: its own transaction. Residual race (accepted for M3): two queued runs of
#: one thread — only possible via ``wait_for_idle`` — could be claimed by two
#: workers in the same instant, because the second worker's snapshot predates
#: the first one's commit; the enqueue-time guard remains the primary defense.
_CLAIM_SQL: Final[str] = f"""
WITH candidate AS (
    SELECT r.id
    FROM runs AS r
    WHERE r.status = 'queued'
      AND (%s::text IS NULL OR r.agent = %s::text)
      AND NOT EXISTS (
          SELECT 1
          FROM runs AS other
          WHERE other.thread_id = r.thread_id
            AND other.status IN ('running', 'awaiting_approval')
      )
    ORDER BY r.created_at, r.id
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
UPDATE runs
SET status = 'running',
    claimed_by = %s,
    lease_expires_at = now() + make_interval(secs => %s),
    attempt = runs.attempt + 1,
    updated_at = now()
FROM candidate
WHERE runs.id = candidate.id
RETURNING {_RUN_COLUMNS}
"""

_HEARTBEAT_SQL: Final[str] = """
UPDATE runs
SET lease_expires_at = now() + make_interval(secs => %s),
    updated_at = now()
WHERE id = %s AND status = 'running'
RETURNING id
"""

#: ``finish`` for every terminal status but ``cancelled``: the allowed source
#: states (``_finish_sources``) guard the UPDATE, like the in-memory map.
_FINISH_SQL: Final[str] = """
UPDATE runs
SET status = %s,
    claimed_by = NULL,
    lease_expires_at = NULL,
    updated_at = now()
WHERE id = %s AND status = ANY(%s::text[])
"""


def _finish_sources(status: RunState) -> list[str]:
    """The states :data:`_ALLOWED_TRANSITIONS` lets ``finish`` reach ``status`` from."""
    return sorted(
        source.value for source, targets in _ALLOWED_TRANSITIONS.items() if status in targets
    )

#: Cancel any active run and report the status it had: the ``prior`` CTE locks
#: and reads the row before the UPDATE rewrites it, so the caller learns who
#: owes the final ``run.status`` event (see ``RunQueue.cancel``).
_CANCEL_SQL: Final[str] = """
WITH prior AS (
    SELECT id, status
    FROM runs
    WHERE id = %s AND status IN ('queued', 'running', 'awaiting_approval')
    FOR UPDATE
)
UPDATE runs
SET status = 'cancelled',
    claimed_by = NULL,
    lease_expires_at = NULL,
    updated_at = now()
FROM prior
WHERE runs.id = prior.id
RETURNING prior.status
"""

#: The SIGTERM drain path (PRD §3.2: requeued without consuming an attempt):
#: the claim's attempt increment is given back in the same UPDATE that clears
#: the lease and claim. ``status = 'running'`` makes an illegal source state
#: match zero rows (surfaced as ``InvalidTransition``), like every mutation
#: above; a ``running`` row always has ``attempt >= 1`` (claim increments).
_RELEASE_SQL: Final[str] = """
UPDATE runs
SET status = 'queued',
    attempt = runs.attempt - 1,
    claimed_by = NULL,
    lease_expires_at = NULL,
    updated_at = now()
WHERE id = %s AND status = 'running'
"""

_SUSPEND_SQL: Final[str] = """
UPDATE runs
SET status = 'awaiting_approval',
    lease_expires_at = NULL,
    updated_at = now()
WHERE id = %s AND status = 'running'
"""

#: ``COALESCE`` keeps the row's resume map when the caller passes none.
_RESUME_SQL: Final[str] = """
UPDATE runs
SET status = 'queued',
    claimed_by = NULL,
    lease_expires_at = NULL,
    resume = COALESCE(%s::jsonb, resume),
    updated_at = now()
WHERE id = %s AND status = 'awaiting_approval'
"""

#: Recovery: one statement per branch (§3.2). ``attempt`` is untouched — it
#: increments on claim — and a NULL lease never matches (NULL < now() is
#: NULL), so ``awaiting_approval`` runs are out of scope by construction.
_EXPIRE_SQL: Final[str] = """
UPDATE runs
SET status = 'expired',
    claimed_by = NULL,
    lease_expires_at = NULL,
    updated_at = now()
WHERE status = 'running'
  AND lease_expires_at < now()
  AND attempt >= %s
"""

_REQUEUE_SQL: Final[str] = """
UPDATE runs
SET status = 'queued',
    claimed_by = NULL,
    lease_expires_at = NULL,
    updated_at = now()
WHERE status = 'running'
  AND lease_expires_at < now()
  AND attempt < %s
"""

_GET_SQL: Final[str] = f"""
SELECT {_RUN_COLUMNS}
FROM runs
WHERE id = %s
"""

_LIST_THREAD_SQL: Final[str] = f"""
SELECT {_RUN_COLUMNS}
FROM runs
WHERE thread_id = %s
ORDER BY created_at, id
"""

_COUNT_BY_STATUS_SQL: Final[str] = "SELECT status, count(*) FROM runs GROUP BY status"


def _json_param(value: Any) -> str | None:
    """A ``jsonb`` parameter: ``None`` stays SQL NULL, anything else is JSON text."""
    return None if value is None else json.dumps(value)


class PostgresRunQueue(RunQueue):
    """The production queue over the ``runs`` table (A7: Postgres-only transport).

    Mirrors ``PostgresEventLog``: the DSN comes from the environment (A23),
    psycopg imports lazily (the ``[postgres]`` extra), and each operation
    works on a short-lived autocommit connection, opening an explicit
    ``transaction()`` block where several statements must land atomically.
    The gateway and workers are stateless clients of the same rows.

    Transitions are legal by construction: every mutation is a single UPDATE
    whose ``WHERE`` clause names the source state(s), so an illegal step
    matches zero rows (surfaced as ``InvalidTransition``) instead of
    corrupting the lifecycle.
    """

    def __init__(self, dsn: str) -> None:
        """Queue over ``dsn`` — resolved from the environment by the caller (A23)."""
        self._dsn = dsn

    @asynccontextmanager
    async def _conn(self) -> AsyncGenerator[Any]:
        """One short-lived autocommit connection per call."""
        conn_cls = _load_async_connection_class()
        conn = await conn_cls.connect(self._dsn, autocommit=True)
        try:
            yield conn
        finally:
            await conn.close()

    @staticmethod
    def _row_to_record(row: Any) -> RunRecord:
        """Map a ``runs`` row (:data:`_RUN_COLUMNS` order) onto a ``RunRecord``.

        ``timestamptz`` cells arrive as aware datetimes and become epoch
        seconds, matching the in-memory queue's clock numbers; ``jsonb`` cells
        arrive already decoded.
        """
        lease, created, updated = row[7], row[8], row[9]
        roles = row[11] or []
        return RunRecord(
            id=str(row[0]),
            thread_id=str(row[1]),
            agent=str(row[2]),
            input=row[3],
            status=RunState(str(row[4])),
            attempt=int(row[5]),
            claimed_by=str(row[6]) if row[6] is not None else None,
            lease_expires_at=lease.timestamp() if lease is not None else None,
            created_at=created.timestamp(),
            updated_at=updated.timestamp(),
            user_id=str(row[10]) if row[10] is not None else None,
            roles=tuple(str(role) for role in roles),
            config=row[12],
            resume=row[13],
        )

    async def setup(self) -> None:
        """Create the ``runs`` table + indexes if missing (idempotent).

        ``migrate`` (WP3) reuses :data:`RUNS_DDL`; this boot-time guard lets
        a fresh database work without it. The DDL is parameterless, so
        psycopg's simple query protocol runs every statement in one call.
        """
        async with self._conn() as conn:
            await conn.execute(RUNS_DDL)

    async def probe(self) -> None:
        """Readiness probe (``/readyz``): one round trip, raising when Postgres is down."""
        async with self._conn() as conn:
            await conn.execute("SELECT 1")

    async def enqueue(
        self,
        agent: str,
        input: Any,
        *,
        thread_id: str | None,
        run_id: str | None,
        wait_for_idle: bool = False,
        user_id: str | None = None,
        roles: Sequence[str] = (),
        config: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        if run_id is None:
            run_id = _new_run_id()
        if thread_id is None:
            raise ValueError(
                "thread_id is required: the runs table stores every run against "
                "its thread (PRD §3.1) — resolve the thread before enqueueing"
            )
        async with self._conn() as conn:
            # One transaction: the per-thread lock, the active-run check and
            # the INSERT land together (D12/A11).
            async with conn.transaction():
                await conn.execute(_ENQUEUE_LOCK_SQL, (thread_id,))
                cur = await conn.execute(_ENQUEUE_CHECK_SQL, (thread_id,))
                active = await cur.fetchall()
                if active:
                    statuses = [RunState(str(row[1])) for row in active]  # (id, status)
                    if not (
                        wait_for_idle and all(state in WAITABLE_STATES for state in statuses)
                    ):
                        raise ActiveRunConflict(
                            active_run_id=str(active[0][0]), reason="thread_has_active_run"
                        )
                cur = await conn.execute(
                    _ENQUEUE_SQL,
                    (
                        run_id,
                        thread_id,
                        agent,
                        json.dumps(input),
                        user_id,
                        json.dumps(list(roles)),
                        _json_param(dict(config) if config is not None else None),
                    ),
                )
                row = await cur.fetchone()
        if row is None:  # pragma: no cover - INSERT..RETURNING always yields a row
            raise RuntimeError("runs insert returned no row")
        return self._row_to_record(row)

    async def claim(self, claimant: str, *, agent: str | None = None) -> RunRecord | None:
        async with self._conn() as conn:
            cur = await conn.execute(_CLAIM_SQL, (agent, agent, claimant, LEASE_INTERVAL_S))
            row = await cur.fetchone()
        return self._row_to_record(row) if row is not None else None

    async def heartbeat(self, run_id: str) -> bool:
        async with self._conn() as conn:
            cur = await conn.execute(_HEARTBEAT_SQL, (LEASE_INTERVAL_S, run_id))
            row = await cur.fetchone()
        return row is not None

    async def finish(self, run_id: str, status: RunState) -> None:
        if status not in TERMINAL_STATES:
            raise ValueError(f"finish() needs a terminal status, got {status!r}")
        sql = _CANCEL_SQL if status is RunState.CANCELLED else _FINISH_SQL
        params: tuple[Any, ...] = (
            (run_id,)
            if status is RunState.CANCELLED
            else (status.value, run_id, _finish_sources(status))
        )
        async with self._conn() as conn:
            cur = await conn.execute(sql, params)
            if cur.rowcount == 0:
                raise InvalidTransition(run_id, None, status)

    async def suspend(self, run_id: str) -> None:
        async with self._conn() as conn:
            cur = await conn.execute(_SUSPEND_SQL, (run_id,))
            if cur.rowcount == 0:
                raise InvalidTransition(run_id, None, RunState.AWAITING_APPROVAL)

    async def resume(self, run_id: str, resume: Mapping[str, Any] | None = None) -> None:
        resume_param = _json_param(dict(resume) if resume is not None else None)
        async with self._conn() as conn:
            cur = await conn.execute(_RESUME_SQL, (resume_param, run_id))
            if cur.rowcount == 0:
                raise InvalidTransition(run_id, None, RunState.QUEUED)

    async def cancel(self, run_id: str) -> RunState | None:
        async with self._conn() as conn:
            cur = await conn.execute(_CANCEL_SQL, (run_id,))
            row = await cur.fetchone()
        return RunState(str(row[0])) if row is not None else None

    async def release(self, run_id: str) -> None:
        async with self._conn() as conn:
            cur = await conn.execute(_RELEASE_SQL, (run_id,))
            if cur.rowcount == 0:
                raise InvalidTransition(run_id, None, RunState.QUEUED)

    async def get(self, run_id: str) -> RunRecord | None:
        async with self._conn() as conn:
            cur = await conn.execute(_GET_SQL, (run_id,))
            row = await cur.fetchone()
        return self._row_to_record(row) if row is not None else None

    async def list_for_thread(self, thread_id: str) -> list[RunRecord]:
        async with self._conn() as conn:
            cur = await conn.execute(_LIST_THREAD_SQL, (thread_id,))
            rows = await cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def count_by_status(self) -> dict[RunState, int]:
        async with self._conn() as conn:
            cur = await conn.execute(_COUNT_BY_STATUS_SQL)
            rows = await cur.fetchall()
        return {RunState(str(row[0])): int(row[1]) for row in rows}

    async def requeue_expired(self) -> int:
        async with self._conn() as conn:
            expired_cur = await conn.execute(_EXPIRE_SQL, (MAX_ATTEMPTS,))
            requeued_cur = await conn.execute(_REQUEUE_SQL, (MAX_ATTEMPTS,))
        return int(expired_cur.rowcount) + int(requeued_cur.rowcount)
