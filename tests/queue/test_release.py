"""``RunQueue.release`` — the SIGTERM drain path (PRD §3.2, Milestone 3 WP2).

The one queue surface WP2 added: a drained run returns to ``queued`` with the
claim's attempt increment given back, so a later claim re-increments to the
same number ("drained and requeued without consuming an attempt"). In-memory
behavior here; the Postgres statement is pinned by its SQL constant (no live
Postgres in this suite — see ``test_postgres_live.py``).
"""

import pytest

from vdreamer import LEASE_INTERVAL_S, InMemoryRunQueue, InvalidTransition, RunState, queue


class _Clock:
    """Deterministic clock: stands still until ``advance`` moves it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def test_release_requeues_with_the_attempt_given_back() -> None:
    clock = _Clock()
    q = InMemoryRunQueue(clock)
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    claimed = await q.claim("w1")
    assert claimed is not None and claimed.attempt == 1
    assert claimed.lease_expires_at == 1000.0 + LEASE_INTERVAL_S
    clock.advance(1.0)
    await q.release(run.id)
    released = await q.get(run.id)
    assert released is not None
    assert released.status is RunState.QUEUED
    assert released.attempt == 0, "the claim's attempt increment was given back"
    assert released.claimed_by is None
    assert released.lease_expires_at is None
    assert released.updated_at == 1001.0
    # A later claim re-increments to the SAME attempt number: not consumed.
    reclaimed = await q.claim("w2")
    assert reclaimed is not None
    assert reclaimed.id == run.id
    assert reclaimed.attempt == 1
    assert reclaimed.claimed_by == "w2"


@pytest.mark.parametrize("setup_state", ["queued", "awaiting_approval"])
async def test_release_requires_running(setup_state: str) -> None:
    q = InMemoryRunQueue(_Clock())
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    if setup_state == "awaiting_approval":
        await q.claim("w1")
        await q.suspend(run.id)
    with pytest.raises(InvalidTransition) as exc:
        await q.release(run.id)
    assert exc.value.to_state is RunState.QUEUED
    assert exc.value.from_state is (
        RunState.QUEUED if setup_state == "queued" else RunState.AWAITING_APPROVAL
    )


async def test_release_rejects_missing_and_terminal_runs() -> None:
    q = InMemoryRunQueue(_Clock())
    with pytest.raises(InvalidTransition) as exc:
        await q.release("run_missing")
    assert exc.value.from_state is None
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    await q.claim("w1")
    await q.finish(run.id, RunState.COMPLETED)
    with pytest.raises(InvalidTransition):
        await q.release(run.id)


def test_release_sql_gives_the_attempt_back() -> None:
    sql = queue._RELEASE_SQL.lower()
    assert "status = 'queued'" in sql
    assert "attempt = runs.attempt - 1" in sql
    assert "where id = %s and status = 'running'" in sql
    assert "claimed_by = null" in sql
    assert "lease_expires_at = null" in sql
