"""Milestone 8 primitives: ``require_approval``, ``ApprovalGate``, interrupt policies,
four-eyes matching, ``tool_review`` normalization (both drivers), expiry in the
stores, and the ``validate`` route checks (§4.1–§4.5, A16, A32, A44)."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

import pydantic
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from vdreamer import (
    ApprovalGate,
    BaseAgent,
    BuildContext,
    InterruptPolicy,
    Mode,
    VDreamerContext,
    bind_context,
    require_approval,
)
from vdreamer.approval import approval_gates
from vdreamer.eventlog import InMemoryEventSink
from vdreamer.interrupts import (
    InMemoryInterruptStore,
    InterruptRecord,
    InterruptStatus,
    apply_interrupt_policy,
    is_strict_scope,
)
from vdreamer.queue import RunState, _finish_sources
from vdreamer.runtime import AgentRunner
from vdreamer.sqlite_state import SqliteInterruptStore, open_state_db
from vdreamer.streaming import STREAM_DRIVER_ENV_VAR, normalize_interrupt
from vdreamer.validate import validate_agent


def _ctx(*, headless: bool = False) -> VDreamerContext:
    return VDreamerContext(run_id="r1", thread_id="t1", agent_name="a", headless=headless)


# -- require_approval ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"actions": []}, "non-empty"),
        ({"actions": ["approve", "explode"]}, r"unknown: \['explode'\]"),
        ({"ttl_hours": 0}, "positive"),
    ],
)
def test_require_approval_refuses_a_bad_policy(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        require_approval("k", {}, **kwargs)


def test_headless_require_approval_needs_an_approver_role() -> None:
    with (
        bind_context(_ctx(headless=True)),
        pytest.raises(ValueError, match="needs an approver_role"),
    ):
        require_approval("verdict", {})


class _Edit(pydantic.BaseModel):
    limit: int = pydantic.Field(ge=1)


class _S(TypedDict, total=False):
    decision: Any


def _gate_graph() -> Any:
    def gate(state: _S) -> dict[str, Any]:
        return {"decision": require_approval("k", {}, edit_schema=_Edit)}

    graph = StateGraph(_S)
    graph.add_node("gate", gate)
    graph.add_edge(START, "gate")
    graph.add_edge("gate", END)
    return graph.compile(checkpointer=InMemorySaver())


def test_edit_payloads_are_revalidated_and_dumped_in_the_node() -> None:
    async def go() -> None:
        graph = _gate_graph()
        config: Any = {"configurable": {"thread_id": "t"}}
        await graph.ainvoke({}, config)
        (pending,) = (await graph.aget_state(config)).interrupts
        assert pending.value["edit_schema"]["properties"]["limit"]["minimum"] == 1
        out = await graph.ainvoke(
            Command(resume={"action": "edit", "payload": {"limit": "7"}}), config
        )
        assert out["decision"] == {"action": "edit", "payload": {"limit": 7}}

        other: Any = {"configurable": {"thread_id": "u"}}
        await graph.ainvoke({}, other)
        with pytest.raises(pydantic.ValidationError):
            await graph.ainvoke(
                Command(resume={"action": "edit", "payload": {"limit": 0}}), other
            )

    asyncio.run(go())


# -- ApprovalGate --------------------------------------------------------------------------------


class _G(TypedDict, total=False):
    x: int
    approval: Any
    out: str


def _noop(state: _G) -> dict[str, Any]:
    return {}


def test_gate_constructor_rules() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ApprovalGate(_G, kind="k", payload=dict, execute=_noop, actions=["respond"])
    with pytest.raises(ValueError, match="decision_key"):
        ApprovalGate(_G, kind="k", payload=dict, execute=_noop, replan=_noop)


def test_gate_routes_and_unwired_actions() -> None:
    gate = ApprovalGate(_G, kind="k", payload=dict, execute=_noop)
    assert gate.routes() == {
        "approve": "execute",
        "reject": None,
        "edit": "execute",
        "feedback": None,
    }
    assert gate.unwired() == ["reject", "feedback"]
    assert gate.unwired(headless=True) == ["feedback"]  # a headless reject ends the run
    narrow = ApprovalGate(_G, kind="k", payload=dict, execute=_noop, actions=["approve"])
    assert narrow.unwired() == []


def test_gate_metadata_is_found_in_nested_subgraphs() -> None:
    gate = ApprovalGate(
        _G, kind="nested", payload=dict, execute=_noop, on_reject=_noop, approver_role="ops"
    )
    parent = StateGraph(_G)
    parent.add_node("review", gate.compile())
    parent.add_edge(START, "review")
    parent.add_edge("review", END)
    (found,) = approval_gates(parent.compile())
    assert found["kind"] == "nested" and found["approver_role"] == "ops"
    assert found["routes"]["feedback"] is None


def test_gate_routes_each_action() -> None:
    gate = ApprovalGate(
        _G,
        kind="k",
        payload=lambda state: {"x": state["x"]},
        prepare=lambda state: {"x": state["x"] + 1},
        execute=lambda state: {"out": f"exec {state['x']}"},
        on_reject=lambda state: {"out": "rejected"},
        replan=lambda state: {"x": state["x"] * 10},
        decision_key="approval",
    )
    graph = gate.compile(checkpointer=InMemorySaver())

    async def run(*resumes: dict[str, Any]) -> dict[str, Any]:
        config: Any = {"configurable": {"thread_id": str(resumes)}}
        await graph.ainvoke({"x": 1}, config)
        out: dict[str, Any] = {}
        for resume in resumes:
            out = await graph.ainvoke(Command(resume=resume), config)
        return out

    assert asyncio.run(run({"action": "approve"}))["out"] == "exec 2"
    assert asyncio.run(run({"action": "reject"}))["out"] == "rejected"
    assert asyncio.run(run({"action": "edit", "payload": {"x": 9}}))["out"] == "exec 9"
    looped = asyncio.run(run({"action": "feedback", "text": "bigger"}, {"action": "approve"}))
    assert looped["out"] == "exec 21"  # replan 2→20, then prepare again 20→21
    assert looped["approval"] == {"action": "approve"}


# -- interrupt_policies (§4.5) and four-eyes (A32) -------------------------------------------------

_REVIEW = {
    "interrupt_id": "i",
    "kind": "tool_review",
    "actions": ["approve"],
    "payload": {"action_requests": [{"name": "sql_db_query", "args": {}}], "review_configs": []},
    "expires_at": None,
}


def test_policy_tool_glob_beats_the_kind_per_field() -> None:
    policies = {
        "sql_db_*": InterruptPolicy(approver_role="dba"),
        "tool_review": InterruptPolicy(approver_role="ops", scope="db:read", ttl_hours=3),
    }
    now = 1_800_000_000.0
    applied = apply_interrupt_policy(_REVIEW, policies, now=now)
    assert applied["approver_role"] == "dba"  # the tool key wins
    assert applied["scope"] == "db:read"  # the kind fills the rest
    assert applied["expires_at"] == "2027-01-15T11:00:00+00:00"  # now + 3 h


def test_policy_leaves_require_approval_and_unmatched_sources_alone() -> None:
    own = {**_REVIEW, "kind": "sql_review", "approver_role": None}
    assert apply_interrupt_policy(own, {"sql_review": InterruptPolicy(approver_role="x")}) == own
    assert apply_interrupt_policy(_REVIEW, {"other_*": InterruptPolicy(scope="s")}) == _REVIEW
    elicit = {**_REVIEW, "kind": "mcp_elicitation", "payload": {"tool_name": "netinv_book"}}
    applied = apply_interrupt_policy(elicit, {"netinv_*": InterruptPolicy(approver_role="noc")})
    assert applied["approver_role"] == "noc"


def test_strict_scope_matching() -> None:
    patterns = ("db:write:*", "payments")
    assert is_strict_scope("db:write:prod", patterns)
    assert is_strict_scope("payments", patterns)
    assert not is_strict_scope("db:read", patterns)
    assert not is_strict_scope(None, patterns)
    assert not is_strict_scope("db:write:prod", ())


# -- tool_review normalization, on both drivers ----------------------------------------------------


class _FakeToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


@tool
def drop_table(name: str, token: str) -> str:
    """Drop a table."""
    return "dropped"


class _ReviewAgent(BaseAgent):
    name = "reviewing"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> Any:
        call = {"name": "drop_table", "args": {"name": "t", "token": "s3cret"}, "id": "c1"}
        model = _FakeToolCallingModel(
            responses=[AIMessage("", id="ai_1", tool_calls=[call]), AIMessage("ok", id="ai_2")]
        )
        kwargs = ctx.agent_kwargs()
        kwargs["middleware"] = [
            HumanInTheLoopMiddleware(interrupt_on={"drop_table": True}),
            *kwargs["middleware"],
        ]
        return create_agent(model=model, tools=[drop_table], **kwargs)


@pytest.fixture
def restore_driver() -> Iterator[None]:
    old = os.environ.get(STREAM_DRIVER_ENV_VAR)
    yield
    if old is None:
        os.environ.pop(STREAM_DRIVER_ENV_VAR, None)
    else:
        os.environ[STREAM_DRIVER_ENV_VAR] = old


@pytest.mark.parametrize("driver", ["v2", "v3"])
def test_hitl_middleware_normalizes_to_tool_review(driver: str, restore_driver: None) -> None:
    os.environ[STREAM_DRIVER_ENV_VAR] = driver

    async def go() -> dict[str, Any]:
        sink = InMemoryEventSink()
        await AgentRunner(_ReviewAgent(), checkpointer=InMemorySaver()).arun(
            {"messages": [HumanMessage("drop t")]},
            config={"configurable": {"thread_id": "t"}},
            context=_ctx(),
            sink=sink,
        )
        (event,) = [e for e in sink.events if e.type.value == "interrupt.requested"]
        return dict(event.payload)

    payload = asyncio.run(go())
    assert payload["kind"] == "tool_review"
    assert payload["actions"] == ["approve", "edit", "reject", "respond"]
    (request,) = payload["payload"]["action_requests"]
    assert request["name"] == "drop_table"
    assert request["args"] == {"name": "t", "token": "***"}  # A33 defaults
    assert payload["expires_at"] is not None


def test_non_review_values_are_unchanged_by_the_review_branch() -> None:
    assert normalize_interrupt({"action_requests": "x"}, "i").kind == "generic"
    assert normalize_interrupt("plain", "i").payload == {"value": "plain"}


# -- expiry in the stores --------------------------------------------------------------------------


def _record(interrupt_id: str, expires_at: float | None) -> InterruptRecord:
    return InterruptRecord(
        interrupt_id=interrupt_id,
        run_id=f"run_{interrupt_id}",
        thread_id="t",
        agent="a",
        kind="k",
        actions=("approve",),
        expires_at=expires_at,
        created_at=time.time(),
    )


async def _exercise_expiry(store: InMemoryInterruptStore) -> None:
    now = 1_000.0
    for interrupt_id, expires in (("late", 900.0), ("later", 950.0), ("ok", 2_000.0), ("no", None)):
        await store.create(_record(interrupt_id, expires))
    expired = await store.expire_overdue(now)
    assert [r.interrupt_id for r in expired] == ["late", "later"]
    assert all(r.status is InterruptStatus.EXPIRED and r.version == 2 for r in expired)
    assert await store.expire_overdue(now) == []
    pending = await store.list()
    assert sorted(r.interrupt_id for r in pending) == ["no", "ok"]
    assert [r.interrupt_id for r in await store.expire_overdue(5_000.0, limit=1)] == ["ok"]


def test_expire_overdue_in_memory() -> None:
    asyncio.run(_exercise_expiry(InMemoryInterruptStore()))


def test_expire_overdue_persists_in_sqlite(tmp_path: Path) -> None:
    async def go() -> None:
        db = await open_state_db(tmp_path / "state.db")
        try:
            await _exercise_expiry(SqliteInterruptStore(db))
            reloaded = SqliteInterruptStore(db)
            await reloaded.load()
            record = await reloaded.get("late")
            assert record is not None and record.status is InterruptStatus.EXPIRED
        finally:
            await db.close()

    asyncio.run(go())


def test_postgres_finish_guards_on_the_lifecycle_sources() -> None:
    assert _finish_sources(RunState.REJECTED) == ["awaiting_approval", "running"]
    assert _finish_sources(RunState.EXPIRED) == ["awaiting_approval", "running"]
    assert _finish_sources(RunState.COMPLETED) == ["running"]


# -- validate: approval routes (§2.3, A16) ---------------------------------------------------------


def _routes_check(agent: BaseAgent) -> Any:
    (check,) = [c for c in validate_agent(agent) if c.name == "approval.routes"]
    return check


class _GateAgent(BaseAgent):
    name = "gated"
    mode = Mode.CHAT
    on_reject: Any = None

    def build(self, ctx: BuildContext) -> Any:
        return ApprovalGate(
            _G,
            kind="sql_review",
            payload=dict,
            execute=_noop,
            on_reject=self.on_reject,
            actions=["approve", "reject"],
            approver_role=None,
        ).build()


def test_validate_flags_an_unwired_gate_route() -> None:
    check = _routes_check(_GateAgent())
    assert not check.ok and "unwired routes: ['reject']" in check.detail

    class Wired(_GateAgent):
        on_reject = staticmethod(_noop)

    check = _routes_check(Wired())
    assert check.ok and "ApprovalGate(sql_review)" in check.detail


def test_validate_headless_gates_need_a_role_but_no_reject_route() -> None:
    class Headless(_GateAgent):
        mode = Mode.HEADLESS

    check = _routes_check(Headless())
    assert not check.ok
    assert "without an approver_role" in check.detail and "unwired" not in check.detail


def test_validate_headless_hitl_middleware_needs_a_review_role() -> None:
    class HeadlessReview(_ReviewAgent):
        mode = Mode.HEADLESS

    check = _routes_check(HeadlessReview())
    assert not check.ok and "interrupt_policies['tool_review']" in check.detail

    class Covered(HeadlessReview):
        interrupt_policies = {"tool_review": InterruptPolicy(approver_role="dba")}

    assert _routes_check(Covered()).ok
