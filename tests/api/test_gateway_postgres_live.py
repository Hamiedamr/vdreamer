"""Live-Postgres coverage of the Milestone-4 backends — skipped without a DSN.

Set ``VDREAMER_TEST_PG_DSN`` (e.g. ``postgresql://vd:vd@localhost:5432/vd``)
to run. Each test works on fresh ids and cleans its rows up. These exercise
what the offline fakes cannot: real ``jsonb`` adaptation, the advisory-lock
enqueue guard, ``LISTEN``/``NOTIFY`` delivery, and the CAS resolve.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import pytest

from vdreamer.eventlog import (
    EventHub,
    PostgresEventListener,
    PostgresEventLog,
    PostgresEventStore,
)
from vdreamer.events import EventEnvelope, EventType
from vdreamer.interrupts import (
    InterruptConflict,
    InterruptRecord,
    InterruptStatus,
    PostgresInterruptStore,
)
from vdreamer.queue import (
    ActiveRunConflict,
    PostgresRunQueue,
    RunState,
    _load_async_connection_class,
)
from vdreamer.threads import PostgresThreadStore

pytestmark = pytest.mark.skipif(
    os.environ.get("VDREAMER_TEST_PG_DSN") is None,
    reason="VDREAMER_TEST_PG_DSN is not set — live Postgres coverage only",
)


def _dsn() -> str:
    return os.environ["VDREAMER_TEST_PG_DSN"]


async def _sql(sql: str, params: tuple[object, ...] = ()) -> None:
    conn = await _load_async_connection_class().connect(_dsn(), autocommit=True)
    try:
        await conn.execute(sql, params)
    finally:
        await conn.close()


async def test_live_event_store_round_trip_and_notify() -> None:
    log = await PostgresEventLog.connect(_dsn())
    await log.setup()
    await log.close()
    run_id = f"run_{uuid.uuid4().hex}"
    store = PostgresEventStore(_dsn())
    hub = EventHub()
    listener = PostgresEventListener(_dsn(), hub)
    listener.start()
    try:
        for _ in range(200):
            if listener.connected:
                break
            await asyncio.sleep(0.01)
        assert listener.connected
        with hub.subscribe(run_id) as sub:
            envelope = EventEnvelope(
                type=EventType.TOOL_CALL,
                run_id=run_id,
                thread_id="t",
                agent="a",
                seq=0,
                ns=("child:1",),
                payload={"tool_call_id": "tc", "args": {"q": "SELECT 1"}},
            )
            await store.write(envelope)  # dict payload + ns list → jsonb
            assert envelope.seq == 1
            assert await sub.wait(2.0) is True  # NOTIFY delivered on commit
            await store.write(EventEnvelope(
                type=EventType.LOG, run_id=run_id, thread_id="t", agent="a", seq=0,
                payload={"level": "info", "message": "m"},
            ))
        events = await store.read(run_id, 0, agent="a", thread_id="t")
        assert [e.seq for e in events] == [1, 2]
        assert events[0].payload == {"tool_call_id": "tc", "args": {"q": "SELECT 1"}}
        assert events[0].ns == ("child:1",) and events[0].sse_id == f"{run_id}:1"
        assert [e.seq for e in await store.read(run_id, 1)] == [2]
    finally:
        await listener.stop()
        await _sql("DELETE FROM run_events WHERE run_id = %s", (run_id,))


async def test_live_queue_m4_columns_and_guards() -> None:
    queue = PostgresRunQueue(_dsn())
    await queue.setup()
    thread = f"t_{uuid.uuid4().hex}"
    agent = f"agent_{uuid.uuid4().hex[:8]}"
    try:
        run = await queue.enqueue(
            agent, {"x": 1}, thread_id=thread, run_id=None,
            user_id="alice", roles=["dba"], config={"configurable": {"k": "v"}},
        )
        assert (run.user_id, run.roles) == ("alice", ("dba",))
        assert run.config == {"configurable": {"k": "v"}}
        assert await queue.claim("w", agent="someone-else") is None
        claimed = await queue.claim("w", agent=agent)
        assert claimed is not None and claimed.id == run.id
        # wait_for_idle queues behind running (the M3 status-index bug made this conflict).
        waiting = await queue.enqueue(agent, {}, thread_id=thread, run_id=None, wait_for_idle=True)
        with pytest.raises(ActiveRunConflict):
            await queue.enqueue(agent, {}, thread_id=thread, run_id=None, wait_for_idle=True)
        assert [r.id for r in await queue.list_for_thread(thread)] == [run.id, waiting.id]
        await queue.suspend(run.id)
        await queue.resume(run.id, resume={"int_1": {"action": "approve"}})
        resumed = await queue.get(run.id)
        assert resumed is not None and resumed.resume == {"int_1": {"action": "approve"}}
        assert await queue.cancel(run.id) is RunState.QUEUED
        assert await queue.cancel(run.id) is None
        counts = await queue.count_by_status()
        assert counts.get(RunState.CANCELLED, 0) >= 1
        await queue.probe()
    finally:
        await _sql("DELETE FROM runs WHERE thread_id = %s", (thread,))


async def test_live_concurrent_enqueues_serialize_per_thread() -> None:
    queue = PostgresRunQueue(_dsn())
    await queue.setup()
    thread = f"t_{uuid.uuid4().hex}"
    try:
        results = await asyncio.gather(
            *(queue.enqueue("a", {}, thread_id=thread, run_id=None) for _ in range(5)),
            return_exceptions=True,
        )
        created = [r for r in results if not isinstance(r, BaseException)]
        conflicts = [r for r in results if isinstance(r, ActiveRunConflict)]
        assert len(created) == 1 and len(conflicts) == 4  # the advisory lock held
    finally:
        await _sql("DELETE FROM runs WHERE thread_id = %s", (thread,))


async def test_live_thread_store() -> None:
    store = PostgresThreadStore(_dsn())
    await store.setup()
    owner = f"user_{uuid.uuid4().hex}"
    try:
        first = await store.create("echo", owner, metadata={"topic": "x"})
        second = await store.create("form", owner)
        assert (await store.get(first.id)) == first
        assert [t.id for t in await store.list(owner)] == [second.id, first.id]
        assert [t.id for t in await store.list(owner, agent="echo")] == [first.id]
        assert await store.soft_delete(first.id) is True
        assert await store.soft_delete(first.id) is False
        assert await store.get(first.id) is None
    finally:
        await _sql("DELETE FROM threads WHERE owner_id = %s", (owner,))


async def test_live_interrupt_store_cas() -> None:
    store = PostgresInterruptStore(_dsn())
    await store.setup()
    run_id = f"run_{uuid.uuid4().hex}"
    record = InterruptRecord(
        interrupt_id=f"int_{uuid.uuid4().hex}",
        run_id=run_id,
        thread_id="t",
        agent="gate",
        kind="sql_review",
        actions=("approve", "edit"),
        payload={"sql": "SELECT 1"},
        approver_role="dba",
        edit_schema={"type": "object"},
        expires_at=time.time() + 60,
        initiator_id="alice",
    )
    try:
        created = await store.create(record)
        assert (await store.create(record)).created_at == created.created_at  # idempotent
        assert created.actions == ("approve", "edit") and created.edit_schema == {"type": "object"}
        assert created.expires_at is not None and abs(created.expires_at - record.expires_at) < 1
        with pytest.raises(InterruptConflict):
            await store.resolve(record.interrupt_id, expected_version=9, resolved_by="d",
                                resolution={})
        resolved = await store.resolve(
            record.interrupt_id, expected_version=1, resolved_by="dana",
            resolution={"action": "approve"},
        )
        assert resolved.status is InterruptStatus.RESOLVED and resolved.version == 2
        assert resolved.resolution == {"action": "approve"}
        with pytest.raises(InterruptConflict):  # the first resolver won (A12)
            await store.resolve(record.interrupt_id, expected_version=1, resolved_by="x",
                                resolution={})
        assert [r.interrupt_id for r in await store.list(status=None, run_id=run_id)] == [
            record.interrupt_id
        ]
        assert await store.cancel_for_run(run_id) == 0  # nothing pending
    finally:
        await _sql("DELETE FROM interrupts WHERE run_id = %s", (run_id,))
