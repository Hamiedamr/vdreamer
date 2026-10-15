"""UC1 acceptance (PRD §7.3): SQL & Data Analysis, end to end through ``vdreamer run``.

Read path: approve, edit, feedback → replan, reject. Write path: a non-SELECT
goes to the ``dba`` gate scoped ``db:write:warehouse``; the requester (a dba
too) is refused by four-eyes (``HITL_STRICT_SCOPES=db:write:*``), a second dba
approves from the inbox, the requester reattaches and sees it done.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from _uc_kit import EXAMPLES, FakeNeo4j, ScriptedModel, scripted, serve, vdreamer_run
from sql_analyst import SqlAnalystAgent

REVENUE_SQL = (
    "```sql\nSELECT c.region, SUM(o.amount) AS revenue FROM orders o "
    "JOIN customers c ON o.customer_id = c.id GROUP BY c.region ORDER BY revenue DESC\n```"
)
ARCHIVE_SQL = "```sql\nUPDATE orders SET status = 'archived' WHERE status = 'cancelled'\n```"


def _seed(path: Path) -> Path:
    spec = importlib.util.spec_from_file_location(
        "uc1_seed", EXAMPLES / "uc1_sql_analyst" / "demo" / "seed.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return Path(module.seed_warehouse(path))


@pytest.fixture
def warehouse(tmp_path: Path) -> Path:
    return _seed(tmp_path / "warehouse.db")


def _stack(warehouse: Path, model: ScriptedModel, semantic: FakeNeo4j) -> Any:
    from _api import make_stack

    @asynccontextmanager
    async def backends() -> AsyncIterator[dict[str, Any]]:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(f"sqlite+aiosqlite:///{warehouse.as_posix()}")
        try:
            yield {"models": {"analyst": model},
                   "resources": {"semantic": semantic, "warehouse": engine}}
        finally:
            await engine.dispose()

    return make_stack(
        workers=True,
        extra_agents=[SqlAnalystAgent()],
        worker_agents=("sql-analyst",),
        poll_interval_s=0.05,
        runner_options={"sql-analyst": backends},
    )


@pytest.fixture
def uc1(warehouse: Path) -> Iterator[tuple[str, ScriptedModel, FakeNeo4j, Any]]:
    model = scripted(REVENUE_SQL)
    semantic = FakeNeo4j()
    stack = _stack(warehouse, model, semantic)
    try:
        with serve(stack) as url:
            yield url, model, semantic, stack
    finally:
        stack.close()


def _artifacts(stack: Any) -> dict[str, Any]:
    import asyncio

    async def rows() -> list[Any]:
        return list((await stack.artifacts.list_records()) if hasattr(
            stack.artifacts, "list_records") else [])

    return {"records": asyncio.run(rows())}


def test_a_question_becomes_a_reviewed_query_a_table_and_a_report(
    uc1: tuple[str, ScriptedModel, FakeNeo4j, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, model, semantic, _ = uc1
    result = vdreamer_run(
        url, monkeypatch, "approve", agent="sql-analyst",
        message="What is the revenue per region?", once=True,
    )
    assert result.code == 0, result.out
    out = result.out
    # schema_link went through the semantic layer, and the model saw only linked tables.
    assert semantic.queries and "orders(" in str(model.seen[0][0].content)
    assert "[interrupt sql_review] actions: approve, reject, edit, feedback" in out
    assert '"lineage"' in out and '"table": "orders"' in out and '"plan"' in out
    assert "[artifact] table 'result'" in out and "[artifact] html 'report'" in out
    assert "returned 3 row(s)" in out and '"region": "EMEA"' in out
    if sys_has_sandbox():
        assert "[artifact] image" in out  # the sandboxed matplotlib chart
    else:
        assert "chart skipped" in out
    assert out.rstrip().endswith("[run completed]")


def sys_has_sandbox() -> bool:
    import os

    return os.name == "posix"


def test_the_reviewer_edits_the_sql(
    uc1: tuple[str, ScriptedModel, FakeNeo4j, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, *_ = uc1
    edited = '{"sql": "SELECT name FROM customers WHERE region = \'EMEA\' ORDER BY name"}'
    result = vdreamer_run(
        url, monkeypatch, "edit", '{"sql": ""}', "edit", edited,
        agent="sql-analyst", message="Which customers do we have?", once=True,
    )
    assert result.code == 0, result.out
    assert "422 invalid_edit" in result.out  # the empty edit was refused by edit_schema
    assert "returned 3 row(s)" in result.out and "Atlas Retail" in result.out


def test_feedback_replans_and_reject_ends_the_turn(
    warehouse: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = scripted(
        REVENUE_SQL,
        "```sql\nSELECT c.region, COUNT(*) AS orders FROM orders o JOIN customers c "
        "ON o.customer_id = c.id GROUP BY c.region\n```",
    )
    stack = _stack(warehouse, model, FakeNeo4j())
    try:
        with serve(stack) as url:
            result = vdreamer_run(
                url, monkeypatch, "feedback", "count orders instead of revenue", "reject",
                "not needed anymore", agent="sql-analyst",
                message="Revenue per region?", once=True,
            )
    finally:
        stack.close()
    assert result.code == 0, result.out
    assert result.out.count("[interrupt sql_review]") == 2  # asked again after the replan
    assert "COUNT(*) AS orders" in result.out
    assert "Reviewer feedback: count orders instead of revenue" in str(model.seen[1])
    assert "the query was not run (not needed anymore)" in result.out


def test_a_write_needs_a_second_dba(warehouse: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = scripted(ARCHIVE_SQL)
    stack = _stack(warehouse, model, FakeNeo4j())
    try:
        with serve(stack) as url:
            # Alice (a dba) asks for the write and tries to approve it herself: four-eyes.
            asked = vdreamer_run(
                url, monkeypatch, "approve", agent="sql-analyst", roles=("dba",),
                message="Archive the cancelled orders", once=True,
            )
            assert asked.code == 3, asked.out
            assert "approver role dba" in asked.out
            assert "four_eyes" in asked.out or "forbidden_role" in asked.out
            thread_id = asked.thread_id()
            # A non-dba sees nothing to decide; Bob (dba) approves from his inbox.
            empty = vdreamer_run(url, monkeypatch, agent="sql-analyst", user="carol",
                                 inbox=True)
            assert "[no pending interrupts for sql-analyst]" in empty.out
            bob = vdreamer_run(url, monkeypatch, "approve", agent="sql-analyst", user="bob",
                               roles=("dba",), inbox=True)
            assert bob.code == 0, bob.out
            assert '"scope": "db:write:warehouse"' not in bob.out  # scope is on the record,
            # Alice reattaches to her thread and sees the write land.
            back = vdreamer_run(url, monkeypatch, agent="sql-analyst", roles=("dba",),
                                thread=thread_id, once=True)
    finally:
        stack.close()
    assert back.code == 0, back.out
    assert "[interrupt resolved: approve by bob]" in back.out
    assert "Done: 1 row(s) changed in orders" in back.out
    with sqlite3.connect(warehouse) as conn:
        assert conn.execute("SELECT status FROM orders WHERE id = 5").fetchone() == ("archived",)
