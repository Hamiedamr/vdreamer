"""Run-queue tests: the §3.2 lifecycle, the in-memory queue, the Postgres SQL.

No live Postgres here: the Postgres queue is covered by its SQL constants and
the lazy-import seam, exactly like ``tests/protocol/test_eventlog.py``. The
live-Postgres suite lives in ``test_postgres_live.py`` and skips without a
DSN.
"""

import sys

import pytest

from vdreamer import (
    LEASE_INTERVAL_S,
    MAX_ATTEMPTS,
    RUNS_DDL,
    ActiveRunConflict,
    InMemoryRunQueue,
    InvalidTransition,
    MissingPostgresDependencyError,
    PostgresRunQueue,
    RunQueue,
    RunRecord,
    RunState,
    queue,
)


class _Clock:
    """Deterministic clock: stands still until ``advance`` moves it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_run_state_members_match_prd() -> None:
    assert {state.value for state in RunState} == {
        "queued",
        "running",
        "awaiting_approval",
        "completed",
        "failed",
        "rejected",
        "expired",
        "cancelled",
    }
    assert RunState.QUEUED == "queued"
    assert RunState.AWAITING_APPROVAL == "awaiting_approval"


def test_runs_ddl_matches_prd() -> None:
    ddl = RUNS_DDL.lower()
    assert "create table if not exists runs" in ddl
    assert "primary key" in ddl
    assert "lease_expires_at" in ddl
    for column in (
        "id",
        "thread_id",
        "agent",
        "input",
        "status",
        "attempt",
        "claimed_by",
        "created_at",
        "updated_at",
    ):
        assert column in ddl
    # WP3's migrate reuses this constant: the index is created idempotently.
    assert "create index if not exists runs_active_thread_idx" in ddl


def test_runs_ddl_partial_index_guards_one_active_run_per_thread() -> None:
    ddl = RUNS_DDL.lower()
    # D12/A11: the partial index covers exactly the active statuses.
    assert "where status in ('queued', 'running', 'awaiting_approval')" in ddl


def test_allowed_transitions_match_prd() -> None:
    transitions = queue._ALLOWED_TRANSITIONS
    assert transitions[RunState.QUEUED] == frozenset({RunState.RUNNING, RunState.CANCELLED})
    assert transitions[RunState.RUNNING] == frozenset(
        {
            RunState.AWAITING_APPROVAL,
            RunState.COMPLETED,
            RunState.FAILED,
            RunState.REJECTED,
            RunState.EXPIRED,
            RunState.CANCELLED,
        }
    )
    assert transitions[RunState.AWAITING_APPROVAL] == frozenset(
        {RunState.QUEUED, RunState.EXPIRED, RunState.CANCELLED}
    )
    for terminal in queue.TERMINAL_STATES:
        assert transitions[terminal] == frozenset()


def test_lifecycle_constants_match_prd() -> None:
    assert LEASE_INTERVAL_S == 10.0  # the §3.2 lease heartbeat
    assert MAX_ATTEMPTS == 3  # recovery requeues at most this often


def test_queue_implementations_satisfy_the_abc() -> None:
    assert issubclass(InMemoryRunQueue, RunQueue)
    assert issubclass(PostgresRunQueue, RunQueue)


async def test_enqueue_creates_a_queued_run() -> None:
    clock = _Clock(start=42.0)
    q = InMemoryRunQueue(clock)
    run = await q.enqueue("sql-analyst", {"question": "hi"}, thread_id="t1", run_id=None)
    assert run.id.startswith("run_")
    assert run.thread_id == "t1"
    assert run.agent == "sql-analyst"
    assert run.input == {"question": "hi"}
    assert run.status is RunState.QUEUED
    assert run.attempt == 0
    assert run.claimed_by is None
    assert run.lease_expires_at is None
    assert (run.created_at, run.updated_at) == (42.0, 42.0)
    stored = await q.get(run.id)
    assert stored == run
    assert await q.get("run_missing") is None


async def test_enqueue_uses_the_callers_run_id() -> None:
    q = InMemoryRunQueue(_Clock())
    run = await q.enqueue("a", {}, thread_id="t1", run_id="run_custom")
    assert run.id == "run_custom"
    with pytest.raises(ValueError):
        await q.enqueue("a", {}, thread_id="t1", run_id="run_custom")


@pytest.mark.parametrize("first_state", ["queued", "running", "awaiting_approval"])
async def test_enqueue_conflicts_on_any_active_run(first_state: str) -> None:
    q = InMemoryRunQueue(_Clock())
    first = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    if first_state != "queued":
        await q.claim("w1")
        if first_state == "awaiting_approval":
            await q.suspend(first.id)
    with pytest.raises(ActiveRunConflict) as exc:
        await q.enqueue("a", {}, thread_id="t1", run_id=None)
    assert exc.value.active_run_id == first.id
    assert exc.value.reason == "thread_has_active_run"


async def test_wait_for_idle_queues_behind_running() -> None:
    q = InMemoryRunQueue(_Clock())
    first = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    await q.claim("w1")
    second = await q.enqueue("a", {}, thread_id="t1", run_id=None, wait_for_idle=True)
    assert second.status is RunState.QUEUED
    assert await q.claim("w2") is None  # still behind the running run
    await q.finish(first.id, RunState.COMPLETED)
    claimed = await q.claim("w2")
    assert claimed is not None and claimed.id == second.id


async def test_wait_for_idle_queues_behind_awaiting_approval() -> None:
    clock = _Clock()
    q = InMemoryRunQueue(clock)
    first = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    await q.claim("w1")
    await q.suspend(first.id)
    clock.advance(1.0)  # a later created_at, so "oldest" is unambiguous
    second = await q.enqueue("a", {}, thread_id="t1", run_id=None, wait_for_idle=True)
    assert second.status is RunState.QUEUED
    assert await q.claim("w2") is None
    await q.resume(first.id)
    claimed = await q.claim("w2")
    # Oldest queued run first: the resumed run predates the wait_for_idle one.
    assert claimed is not None and claimed.id == first.id


async def test_wait_for_idle_still_conflicts_behind_queued() -> None:
    q = InMemoryRunQueue(_Clock())
    first = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    with pytest.raises(ActiveRunConflict) as exc:
        await q.enqueue("a", {}, thread_id="t1", run_id=None, wait_for_idle=True)
    assert exc.value.active_run_id == first.id


async def test_enqueue_without_a_thread_never_conflicts() -> None:
    q = InMemoryRunQueue(_Clock())
    await q.enqueue("a", {}, thread_id=None, run_id=None)
    again = await q.enqueue("a", {}, thread_id=None, run_id=None)
    assert again.status is RunState.QUEUED


async def test_claim_returns_none_when_empty() -> None:
    assert await InMemoryRunQueue(_Clock()).claim("w1") is None


async def test_claim_picks_the_oldest_claimable_run() -> None:
    clock = _Clock()
    q = InMemoryRunQueue(clock)
    older = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    clock.advance(1.0)  # a later created_at, so "oldest" is unambiguous
    newer = await q.enqueue("a", {}, thread_id="t2", run_id=None)
    claimed = await q.claim("w1")
    assert claimed is not None and claimed.id == older.id
    claimed = await q.claim("w1")
    assert claimed is not None and claimed.id == newer.id
    assert await q.claim("w1") is None


async def test_claim_skips_a_thread_that_has_a_running_run() -> None:
    q = InMemoryRunQueue(_Clock())
    first = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    await q.claim("w1")  # t1 -> running
    behind = await q.enqueue("a", {}, thread_id="t1", run_id=None, wait_for_idle=True)
    free = await q.enqueue("a", {}, thread_id="t2", run_id=None)
    claimed = await q.claim("w2")
    assert claimed is not None and claimed.id == free.id  # t1 is busy
    await q.finish(first.id, RunState.COMPLETED)
    claimed = await q.claim("w2")
    assert claimed is not None and claimed.id == behind.id


async def test_claim_stamps_claimant_lease_and_attempt() -> None:
    clock = _Clock(start=500.0)
    q = InMemoryRunQueue(clock)
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    claimed = await q.claim("worker-1")
    assert claimed is not None
    assert claimed.id == run.id
    assert claimed.status is RunState.RUNNING
    assert claimed.attempt == 1
    assert claimed.claimed_by == "worker-1"
    assert claimed.lease_expires_at == 500.0 + LEASE_INTERVAL_S
    assert await q.claim("worker-1") is None  # nothing else claimable


async def test_heartbeat_extends_only_running_leases() -> None:
    clock = _Clock()
    q = InMemoryRunQueue(clock)
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    assert await q.heartbeat(run.id) is False  # queued: nothing to extend
    claimed = await q.claim("w1")
    assert claimed is not None and claimed.lease_expires_at == 1010.0
    clock.advance(4.0)
    assert await q.heartbeat(run.id) is True
    extended = await q.get(run.id)
    assert extended is not None and extended.lease_expires_at == 1014.0
    await q.finish(run.id, RunState.COMPLETED)
    assert await q.heartbeat(run.id) is False  # terminal: the worker must drain
    assert await q.heartbeat("run_missing") is False


async def test_suspend_and_resume_roundtrip() -> None:
    clock = _Clock()
    q = InMemoryRunQueue(clock)
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    await q.claim("w1")
    await q.suspend(run.id)
    suspended = await q.get(run.id)
    assert suspended is not None
    assert suspended.status is RunState.AWAITING_APPROVAL
    assert suspended.lease_expires_at is None  # approvals free the worker (§3.2)
    # The thread is still occupied: a plain enqueue conflicts, the claim waits.
    with pytest.raises(ActiveRunConflict):
        await q.enqueue("a", {}, thread_id="t1", run_id=None)
    assert await q.claim("w2") is None
    await q.resume(run.id)
    resumed = await q.get(run.id)
    assert resumed is not None
    assert resumed.status is RunState.QUEUED
    assert resumed.claimed_by is None  # the stale claim is gone
    reclaimed = await q.claim("w2")
    assert reclaimed is not None and reclaimed.attempt == 2


async def test_suspend_requires_running() -> None:
    q = InMemoryRunQueue(_Clock())
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    with pytest.raises(InvalidTransition):
        await q.suspend(run.id)


async def test_finish_clears_lease_and_claim() -> None:
    q = InMemoryRunQueue(_Clock())
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    await q.claim("w1")
    await q.finish(run.id, RunState.FAILED)
    final = await q.get(run.id)
    assert final is not None
    assert final.status is RunState.FAILED
    assert final.claimed_by is None
    assert final.lease_expires_at is None


async def test_finish_from_queued_is_invalid() -> None:
    q = InMemoryRunQueue(_Clock())
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    with pytest.raises(InvalidTransition) as exc:
        await q.finish(run.id, RunState.COMPLETED)
    assert exc.value.from_state is RunState.QUEUED
    assert exc.value.to_state is RunState.COMPLETED


async def test_terminal_states_are_final() -> None:
    q = InMemoryRunQueue(_Clock())
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    assert await q.cancel(run.id) is RunState.QUEUED
    with pytest.raises(InvalidTransition):
        await q.finish(run.id, RunState.COMPLETED)
    assert await q.cancel(run.id) is None


@pytest.mark.parametrize("setup_state", ["queued", "running", "awaiting_approval"])
async def test_cancel_from_active_states(setup_state: str) -> None:
    q = InMemoryRunQueue(_Clock())
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    if setup_state != "queued":
        await q.claim("w1")
        if setup_state == "awaiting_approval":
            await q.suspend(run.id)
    # cancel reports the prior status: it tells the caller who owes the final event.
    assert await q.cancel(run.id) is RunState(setup_state)
    final = await q.get(run.id)
    assert final is not None
    assert final.status is RunState.CANCELLED
    assert final.claimed_by is None
    assert final.lease_expires_at is None
    assert await q.cancel(run.id) is None  # terminal


async def test_cancel_false_from_terminal_and_unknown() -> None:
    q = InMemoryRunQueue(_Clock())
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    await q.claim("w1")
    await q.finish(run.id, RunState.COMPLETED)
    assert await q.cancel(run.id) is None
    assert await q.cancel("run_missing") is None


async def test_requeue_expired_requeues_and_preserves_attempt() -> None:
    clock = _Clock()
    q = InMemoryRunQueue(clock)
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    claimed = await q.claim("w1")
    assert claimed is not None and claimed.lease_expires_at == 1010.0
    clock.advance(LEASE_INTERVAL_S)  # exactly at the boundary: not yet expired
    assert await q.requeue_expired() == 0
    clock.advance(0.5)
    assert await q.requeue_expired() == 1
    requeued = await q.get(run.id)
    assert requeued is not None
    assert requeued.status is RunState.QUEUED
    assert requeued.attempt == 1  # unchanged: attempt increments on claim
    assert requeued.claimed_by is None
    assert requeued.lease_expires_at is None


async def test_requeue_expired_expires_after_max_attempts() -> None:
    clock = _Clock()
    q = InMemoryRunQueue(clock)
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        claimed = await q.claim("w1")
        assert claimed is not None and claimed.attempt == attempt
        clock.advance(LEASE_INTERVAL_S + 1.0)
        assert await q.requeue_expired() == 1
        state = await q.get(run.id)
        assert state is not None
        if attempt < MAX_ATTEMPTS:
            assert state.status is RunState.QUEUED
        else:
            assert state.status is RunState.EXPIRED
            assert state.attempt == MAX_ATTEMPTS
    assert await q.requeue_expired() == 0  # nothing left running


async def test_requeue_expired_ignores_live_leases() -> None:
    clock = _Clock()
    q = InMemoryRunQueue(clock)
    await q.enqueue("a", {}, thread_id="t1", run_id=None)
    await q.enqueue("a", {}, thread_id="t2", run_id=None)
    await q.claim("w1")
    await q.claim("w2")
    clock.advance(1.0)  # inside both leases
    assert await q.requeue_expired() == 0


def test_claim_sql_claims_with_skip_locked() -> None:
    sql = queue._CLAIM_SQL.lower()
    assert "for update skip locked" in sql
    assert "not exists" in sql  # the thread-active guard
    for state in ("'queued'", "'running'", "'awaiting_approval'"):
        assert state in sql
    assert "order by" in sql and "limit 1" in sql  # oldest first


def test_enqueue_check_sql_guards_the_thread() -> None:
    sql = queue._ENQUEUE_CHECK_SQL.lower()
    # No SKIP LOCKED: skipping an active row a concurrent claim holds would let
    # the one-active-run guard leak. The per-thread advisory lock serializes.
    assert "skip locked" not in sql
    assert "status in ('queued', 'running', 'awaiting_approval')" in sql
    assert "order by" in sql
    # The status column is the second projected column — the M3 code read the
    # thread id there, so wait_for_idle could never succeed on Postgres.
    assert sql.split("from")[0].split() == ["select", "id,", "status"]
    assert "pg_advisory_xact_lock" in queue._ENQUEUE_LOCK_SQL


def test_claim_sql_filters_by_agent() -> None:
    sql = queue._CLAIM_SQL.lower()
    assert "r.agent = %s" in sql


def test_recovery_sql_branches_on_attempt() -> None:
    expire = queue._EXPIRE_SQL.lower()
    requeue = queue._REQUEUE_SQL.lower()
    for sql in (expire, requeue):
        assert "status = 'running'" in sql
        assert "lease_expires_at < now()" in sql
    assert "attempt >= %s" in expire
    assert "attempt < %s" in requeue
    assert "status = 'expired'" in expire
    assert "status = 'queued'" in requeue


async def test_postgres_queue_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None in sys.modules makes `from psycopg import AsyncConnection` raise
    # ImportError, which the lazy-import seam converts to the friendly error.
    monkeypatch.setitem(sys.modules, "psycopg", None)
    q = PostgresRunQueue("postgresql://vd:vd@localhost/vd")
    with pytest.raises(MissingPostgresDependencyError):
        await q.enqueue("a", {}, thread_id="t1", run_id=None)


def test_run_record_is_frozen() -> None:
    record = RunRecord(
        id="run_1",
        thread_id="t1",
        agent="a",
        input={},
        status=RunState.QUEUED,
        attempt=0,
        claimed_by=None,
        lease_expires_at=None,
        created_at=1.0,
        updated_at=1.0,
    )
    with pytest.raises(AttributeError):
        record.status = RunState.RUNNING  # type: ignore[misc]
