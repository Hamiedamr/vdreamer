"""Live-Postgres ``PostgresRunQueue`` tests — skipped without a DSN.

Set ``VDREAMER_TEST_PG_DSN`` (e.g. ``postgresql://vd:vd@localhost:5432/vd``)
to run against a real Postgres. CI does not set it, so these skip offline and
``test_queue.py``'s SQL-constant tests carry the offline coverage. Every test
uses a unique thread id and cleans its rows up.
"""

import os
import uuid

import pytest

from vdreamer import (
    ActiveRunConflict,
    PostgresRunQueue,
    RunState,
)
from vdreamer.queue import _load_async_connection_class

pytestmark = pytest.mark.skipif(
    os.environ.get("VDREAMER_TEST_PG_DSN") is None,
    reason="VDREAMER_TEST_PG_DSN is not set — live Postgres coverage only",
)


def _dsn() -> str:
    return os.environ["VDREAMER_TEST_PG_DSN"]


async def _run_sql(dsn: str, sql: str, params: tuple[object, ...]) -> None:
    conn_cls = _load_async_connection_class()
    conn = await conn_cls.connect(dsn, autocommit=True)
    try:
        await conn.execute(sql, params)
    finally:
        await conn.close()


async def _cleanup(dsn: str, thread_id: str) -> None:
    await _run_sql(dsn, "DELETE FROM runs WHERE thread_id = %s", (thread_id,))


async def _backdate_lease(dsn: str, run_id: str) -> None:
    await _run_sql(
        dsn,
        "UPDATE runs SET lease_expires_at = now() - interval '1 second' WHERE id = %s",
        (run_id,),
    )


async def test_live_enqueue_claim_finish() -> None:
    dsn = _dsn()
    q = PostgresRunQueue(dsn)
    await q.setup()
    thread = f"t_{uuid.uuid4().hex}"
    try:
        run = await q.enqueue("live", {"n": 1}, thread_id=thread, run_id=None)
        assert run.status is RunState.QUEUED
        assert run.attempt == 0
        stored = await q.get(run.id)
        assert stored is not None and stored == run

        claimed = await q.claim("w1")
        assert claimed is not None and claimed.id == run.id
        assert claimed.status is RunState.RUNNING
        assert claimed.attempt == 1
        assert claimed.claimed_by == "w1"
        assert claimed.lease_expires_at is not None
        assert await q.heartbeat(run.id) is True
        assert await q.claim("w1") is None  # nothing else queued

        await q.finish(run.id, RunState.COMPLETED)
        final = await q.get(run.id)
        assert final is not None
        assert final.status is RunState.COMPLETED
        assert final.claimed_by is None
        assert final.lease_expires_at is None
        assert await q.heartbeat(run.id) is False
    finally:
        await _cleanup(dsn, thread)


async def test_live_conflict_and_wait_for_idle() -> None:
    dsn = _dsn()
    q = PostgresRunQueue(dsn)
    await q.setup()
    thread = f"t_{uuid.uuid4().hex}"
    try:
        first = await q.enqueue("live", {}, thread_id=thread, run_id=None)
        with pytest.raises(ActiveRunConflict) as exc:
            await q.enqueue("live", {}, thread_id=thread, run_id=None)
        assert exc.value.active_run_id == first.id

        await q.claim("w1")
        second = await q.enqueue("live", {}, thread_id=thread, run_id=None, wait_for_idle=True)
        assert second.status is RunState.QUEUED
        assert await q.claim("w2") is None  # the thread is busy
        await q.finish(first.id, RunState.COMPLETED)
        claimed = await q.claim("w2")
        assert claimed is not None and claimed.id == second.id
    finally:
        await _cleanup(dsn, thread)


async def test_live_suspend_resume_roundtrip() -> None:
    dsn = _dsn()
    q = PostgresRunQueue(dsn)
    await q.setup()
    thread = f"t_{uuid.uuid4().hex}"
    try:
        run = await q.enqueue("live", {}, thread_id=thread, run_id=None)
        await q.claim("w1")
        await q.suspend(run.id)
        suspended = await q.get(run.id)
        assert suspended is not None
        assert suspended.status is RunState.AWAITING_APPROVAL
        assert suspended.lease_expires_at is None
        await q.resume(run.id)
        resumed = await q.get(run.id)
        assert resumed is not None
        assert resumed.status is RunState.QUEUED
        reclaimed = await q.claim("w1")
        assert reclaimed is not None and reclaimed.attempt == 2
        await q.cancel(run.id)
    finally:
        await _cleanup(dsn, thread)


async def test_live_recovery_and_max_attempts() -> None:
    dsn = _dsn()
    q = PostgresRunQueue(dsn)
    await q.setup()
    thread = f"t_{uuid.uuid4().hex}"
    try:
        run = await q.enqueue("live", {}, thread_id=thread, run_id=None)
        claimed = await q.claim("w1")
        assert claimed is not None and claimed.id == run.id
        await _backdate_lease(dsn, run.id)
        assert await q.requeue_expired() == 1
        requeued = await q.get(run.id)
        assert requeued is not None
        assert requeued.status is RunState.QUEUED
        assert requeued.attempt == 1  # attempt increments on claim, not requeue

        # Two more expiry cycles exhaust MAX_ATTEMPTS; the last one expires.
        for attempt in (2, 3):
            claimed = await q.claim("w1")
            assert claimed is not None and claimed.attempt == attempt
            await _backdate_lease(dsn, run.id)
            assert await q.requeue_expired() == 1
        final = await q.get(run.id)
        assert final is not None
        assert final.status is RunState.EXPIRED
        assert final.attempt == 3
        assert await q.requeue_expired() == 0
    finally:
        await _cleanup(dsn, thread)
