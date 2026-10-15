"""Milestone-4 queue surface: identity/config/resume columns, agent-scoped claims,
cancel's prior status, thread listings, status counts — plus the Postgres SQL shape."""

from __future__ import annotations

from vdreamer import queue
from vdreamer.queue import InMemoryRunQueue, RunState


async def test_enqueue_stores_identity_and_config() -> None:
    q = InMemoryRunQueue()
    run = await q.enqueue(
        "a",
        {"x": 1},
        thread_id="t1",
        run_id=None,
        user_id="alice",
        roles=["dba", "analyst"],
        config={"configurable": {"k": "v"}},
    )
    assert run.user_id == "alice"
    assert run.roles == ("dba", "analyst")
    assert run.config == {"configurable": {"k": "v"}}
    assert run.resume is None


async def test_claim_filters_by_agent() -> None:
    q = InMemoryRunQueue()
    await q.enqueue("a", {}, thread_id="t1", run_id="r_a")
    await q.enqueue("b", {}, thread_id="t2", run_id="r_b")
    claimed = await q.claim("w", agent="b")
    assert claimed is not None and claimed.id == "r_b"
    assert await q.claim("w", agent="b") is None
    claimed = await q.claim("w")  # no filter: anyone's
    assert claimed is not None and claimed.id == "r_a"


async def test_resume_plants_and_keeps_the_resume_map() -> None:
    q = InMemoryRunQueue()
    run = await q.enqueue("a", {}, thread_id="t1", run_id=None)
    await q.claim("w")
    await q.suspend(run.id)
    await q.resume(run.id, resume={"int_1": {"action": "approve"}})
    resumed = await q.get(run.id)
    assert resumed is not None and resumed.status is RunState.QUEUED
    assert resumed.resume == {"int_1": {"action": "approve"}}
    await q.claim("w")
    await q.suspend(run.id)
    await q.resume(run.id)  # None keeps the row's map
    again = await q.get(run.id)
    assert again is not None and again.resume == {"int_1": {"action": "approve"}}


async def test_list_for_thread_and_counts() -> None:
    q = InMemoryRunQueue()
    first = await q.enqueue("a", {}, thread_id="t1", run_id="r1")
    await q.claim("w")
    await q.finish(first.id, RunState.COMPLETED)
    await q.enqueue("a", {}, thread_id="t1", run_id="r2")
    await q.enqueue("a", {}, thread_id="t2", run_id="r3")
    assert [run.id for run in await q.list_for_thread("t1")] == ["r1", "r2"]
    assert await q.list_for_thread("ghost") == []
    assert await q.count_by_status() == {RunState.COMPLETED: 1, RunState.QUEUED: 2}


def test_runs_ddl_upgrades_a_milestone_3_table_in_place() -> None:
    ddl = queue.RUNS_DDL.lower()
    for column in ("user_id text", "roles jsonb", "config jsonb", "resume jsonb"):
        assert f"alter table runs add column if not exists {column}" in ddl
    assert "runs_thread_created_idx" in ddl


def test_postgres_sql_shapes() -> None:
    assert "prior.status" in queue._CANCEL_SQL.lower()  # cancel reports the prior status
    assert "for update" in queue._CANCEL_SQL.lower()
    assert "coalesce(%s::jsonb, resume)" in queue._RESUME_SQL.lower()
    assert "group by status" in queue._COUNT_BY_STATUS_SQL.lower()
    assert queue._RUN_COLUMNS.split(", ")[-6:] == [
        "user_id", "roles", "config", "resume", "callback", "headless"
    ]
    # M9: a /trigger replay re-derives the run id; the insert must not raise on it.
    assert "on conflict (id) do nothing" in queue._ENQUEUE_SQL.lower()


def test_row_mapping_covers_the_new_columns() -> None:
    from datetime import UTC, datetime

    now = datetime(2026, 10, 14, tzinfo=UTC)
    row = (
        "r1", "t1", "a", {"x": 1}, "queued", 0, None, None, now, now,
        "alice", ["dba"], {"configurable": {}}, {"int_1": {"action": "approve"}},
        "https://hooks.internal/cb", True,
    )
    record = queue.PostgresRunQueue._row_to_record(row)
    assert record.user_id == "alice" and record.roles == ("dba",)
    assert record.config == {"configurable": {}}
    assert record.resume == {"int_1": {"action": "approve"}}
    assert record.callback == "https://hooks.internal/cb" and record.headless is True
