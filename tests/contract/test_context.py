"""vdreamer.context tests — run context, implicit helpers, require_approval.

Covers the contextvar round-trip and the implicit-helper promise (PRD §2.2
line 70), the buffered record shapes (§3.3), the artifact ref carriers (§3.4),
and the normalized approval interrupt value (§4.1, A44). Byte storage, MinIO
writes, and the ``artifact.created`` write order (D7) land with Milestone 6;
the worker pipeline that drains ``pending`` with Milestone 2.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from vdreamer.context import (
    DEFAULT_APPROVAL_ACTIONS,
    MAX_SNIPPET_CHARS,
    ArtifactKind,
    ArtifactRef,
    FileRef,
    VDreamerContext,
    bind_context,
    current_context,
    emit_artifact,
    emit_citation,
    emit_log,
    emit_progress,
    require_approval,
)


def _ctx(**overrides: Any) -> VDreamerContext:
    """A context for one test; helpers append into its ``pending`` buffer."""
    fields: dict[str, Any] = {
        "run_id": "run_test",
        "thread_id": "t_test",
        "agent_name": "test-agent",
        "user_id": "u1",
        "roles": ("netops",),
    }
    fields.update(overrides)
    return VDreamerContext(**fields)


def test_bind_context_roundtrip_and_reset() -> None:
    ctx = _ctx()
    with bind_context(ctx):
        assert current_context() is ctx
        inner = _ctx(run_id="run_inner")
        with bind_context(inner):
            assert current_context() is inner
        # nesting resets to the outer binding, not to unbound
        assert current_context() is ctx
    # and the CM resets to unbound on exit
    with pytest.raises(RuntimeError, match="no active vdreamer run context"):
        current_context()


def test_current_context_raises_when_unbound() -> None:
    with pytest.raises(RuntimeError, match="no active vdreamer run context"):
        current_context()


def test_emit_progress_appends_exact_record() -> None:
    ctx = _ctx()
    with bind_context(ctx):
        emit_progress("dry run", 25.0)
        emit_progress("step 2")
    assert ctx.pending == [
        {"type": "progress", "label": "dry run", "pct": 25.0},
        {"type": "progress", "label": "step 2", "pct": None},
    ]


def test_emit_log_appends_exact_record() -> None:
    ctx = _ctx()
    with bind_context(ctx):
        emit_log("loading schema")
        emit_log("slow query", level="warn")
    assert ctx.pending == [
        {"type": "log", "level": "info", "message": "loading schema"},
        {"type": "log", "level": "warn", "message": "slow query"},
    ]


def test_emit_helpers_raise_outside_a_run() -> None:
    with pytest.raises(RuntimeError, match="no active vdreamer run context"):
        emit_log("no run bound")


def test_emit_artifact_appends_record_and_returns_ref() -> None:
    ctx = _ctx()
    with bind_context(ctx):
        ref = emit_artifact(
            "verdict",
            ArtifactKind.JSON,
            {"ok": True},
            metadata={"source": "vision"},
            renderer="verdict.v1",
        )
    assert isinstance(ref, ArtifactRef)
    assert ref.artifact_id.startswith("art_")
    assert len(ref.artifact_id) == len("art_") + 32  # uuid4 hex
    record = ctx.pending[0]
    # M6: the record carries the encoded artifact's description, never its data;
    # no store is bound here, so nothing was kept (``stored`` false, no URLs).
    assert record == {
        "type": "artifact",
        "artifact_id": ref.artifact_id,
        "name": "verdict",
        "kind": "json",
        "mime_type": "application/json",
        "size_bytes": 12,
        "preview": {"body": {"ok": True}},
        "metadata": {"source": "vision"},
        "renderer": "verdict.v1",
        "stored": False,
    }
    assert ref.name == "verdict"
    assert ref.kind is ArtifactKind.JSON
    assert ref.mime_type == "application/json"
    assert ref.size_bytes == 12


def test_emit_artifact_rejects_bad_kind_and_none_data() -> None:
    ctx = _ctx()
    with bind_context(ctx):
        # a bare string is not an ArtifactKind member — no auto-detection (§3.4)
        with pytest.raises(TypeError, match="ArtifactKind"):
            emit_artifact("x", "json", {"data": 1})  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ArtifactKind"):
            emit_artifact("x", None, {"data": 1})  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must not be None"):
            emit_artifact("x", ArtifactKind.FILE, None)
    # nothing buffered when validation failed
    assert ctx.pending == []


def test_emit_citation_appends_record_truncates_and_returns_id() -> None:
    ctx = _ctx()
    long_snippet = "x" * (MAX_SNIPPET_CHARS + 200)
    with bind_context(ctx):
        citation_id = emit_citation(
            "msg_1",
            source={"uri": "doc://a", "title": "Doc A"},
            locator={"page": 3},
            snippet=long_snippet,
            score=0.87,
        )
        # defaults: locator/snippet/score all optional
        default_id = emit_citation("msg_2", {"uri": "doc://b"})
    assert citation_id.startswith("cit_")
    assert len(citation_id) == len("cit_") + 32
    assert ctx.pending == [
        {
            "type": "citation",
            "citation_id": citation_id,
            "message_id": "msg_1",
            "source": {"uri": "doc://a", "title": "Doc A"},
            "locator": {"page": 3},
            "snippet": "x" * MAX_SNIPPET_CHARS,
            "score": 0.87,
        },
        {
            "type": "citation",
            "citation_id": default_id,
            "message_id": "msg_2",
            "source": {"uri": "doc://b"},
            "locator": None,
            "snippet": "",
            "score": None,
        },
    ]
    assert ctx.pending[0]["citation_id"] != ctx.pending[1]["citation_id"]


def test_fileref_and_artifactref_round_trip() -> None:
    ref = FileRef(object_key="inputs/site.zip")
    assert FileRef.model_validate(ref.model_dump()) == ref

    artifact = ArtifactRef(
        artifact_id="art_abc",
        name="verdict",
        kind=ArtifactKind.JSON,
        mime_type="application/json",
        size_bytes=12,
    )
    dumped = artifact.model_dump()
    assert dumped["kind"] == "json"
    assert ArtifactRef.model_validate(dumped) == artifact
    # a bare "json" string coerces to the enum member
    coerced = ArtifactRef.model_validate(
        {
            "artifact_id": "art_abc",
            "name": "verdict",
            "kind": "json",
            "mime_type": "application/json",
            "size_bytes": 12,
        }
    )
    assert coerced.kind is ArtifactKind.JSON


def test_vdreamer_context_defaults_roles_tuple_and_round_trip() -> None:
    ctx = VDreamerContext(run_id="r1", thread_id="t1", agent_name="a1")
    assert ctx.user_id is None
    assert ctx.roles == ()
    assert ctx.idempotency_key is None
    assert ctx.attempt == 0
    assert ctx.pending == []
    # mutable defaults are per-instance, never shared
    ctx.pending.append({"type": "log", "level": "info", "message": "x"})
    assert VDreamerContext(run_id="r2", thread_id="t2", agent_name="a2").pending == []

    full = _ctx(
        run_id="r3",
        thread_id="t3",
        agent_name="a3",
        user_id="u9",
        roles=["netops", "dba"],  # coerced to a tuple
        idempotency_key="idem-1",
        attempt=2,
        pending=[{"type": "progress", "label": "l", "pct": 1.0}],
    )
    assert isinstance(full.roles, tuple)
    assert full.roles == ("netops", "dba")
    assert full.attempt == 2
    dumped = full.model_dump()
    # pydantic keeps the tuple on model_dump (a JSON round-trip would list it)
    assert dumped["roles"] == ("netops", "dba")
    assert VDreamerContext.model_validate(dumped) == full


class _SqlEdit(BaseModel):
    """The edit payload schema for the approval gate."""

    sql: str
    dry_run: bool = False


class _GateState(TypedDict):
    """Two-node approval graph: prepare → gate (interrupt) → end."""

    question: str
    sql: str
    decision: Any


def _prepare(state: _GateState) -> dict[str, Any]:
    return {"sql": "DELETE FROM orders"}


def _gate_explicit(state: _GateState) -> dict[str, Any]:
    decision = require_approval(
        kind="sql_review",
        payload={"sql": state["sql"], "lineage": ["semantic", "warehouse"]},
        actions=["approve", "reject", "edit"],
        approver_role="dba",
        scope="db:write:prod",
        edit_schema=_SqlEdit,
        ttl_hours=48,
    )
    return {"decision": decision}


def _gate_defaults(state: _GateState) -> dict[str, Any]:
    decision = require_approval(kind="generic_review", payload={"note": "see me"})
    return {"decision": decision}


async def test_require_approval_interrupts_then_resumes() -> None:
    graph = StateGraph(_GateState)
    graph.add_node("prepare", _prepare)
    graph.add_node("gate", _gate_explicit)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "gate")
    graph.add_edge("gate", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": "t-gate-explicit"}}

    # First run stops at the interrupt; ainvoke returns normally.
    await compiled.ainvoke({"question": "clean up"}, config, durability="sync")
    state = await compiled.aget_state(config)
    assert state.interrupts
    assert state.next == ("gate",)
    # The pending interrupt carries the normalized approval value (§4.1).
    assert state.interrupts[0].value == {
        "kind": "sql_review",
        "payload": {"sql": "DELETE FROM orders", "lineage": ["semantic", "warehouse"]},
        "actions": ["approve", "reject", "edit"],
        "approver_role": "dba",
        "scope": "db:write:prod",
        "edit_schema": _SqlEdit.model_json_schema(),
        "ttl_hours": 48,
    }

    # Resume (§4.3): the node restarts from the top and the resume value is
    # what require_approval returns.
    resumed = await compiled.ainvoke(
        Command(resume={"action": "approve", "text": "go"}), config, durability="sync"
    )
    assert resumed["decision"] == {"action": "approve", "text": "go"}
    final = await compiled.aget_state(config)
    assert not final.interrupts
    assert final.values["decision"] == {"action": "approve", "text": "go"}


async def test_require_approval_default_actions_and_options() -> None:
    graph = StateGraph(_GateState)
    graph.add_node("gate", _gate_defaults)
    graph.add_edge(START, "gate")
    graph.add_edge("gate", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": "t-gate-defaults"}}

    await compiled.ainvoke({"question": "q"}, config, durability="sync")
    state = await compiled.aget_state(config)
    assert state.interrupts
    assert state.interrupts[0].value == {
        "kind": "generic_review",
        "payload": {"note": "see me"},
        "actions": list(DEFAULT_APPROVAL_ACTIONS),
        "approver_role": None,
        "scope": None,
        "edit_schema": None,
        "ttl_hours": 24,
    }
    assert list(DEFAULT_APPROVAL_ACTIONS) == ["approve", "reject", "edit", "feedback"]


def test_require_approval_outside_a_node_raises() -> None:
    # interrupt() needs the LangGraph run config; outside a runnable context
    # it raises RuntimeError — the gate is only callable from inside a node.
    with pytest.raises(RuntimeError):
        require_approval("sql_review", {"sql": "SELECT 1"})


@pytest.mark.parametrize("pct", [-1, 100.5, 250])
def test_emit_progress_pct_is_a_percentage(pct: float) -> None:
    from vdreamer.context import emit_progress

    with pytest.raises(ValueError, match=r"percentage in \[0, 100\]"):
        emit_progress("step", pct)
