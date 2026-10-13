"""vdreamer.runtime tests — build-once, hooks, durability (A42), contextvar binding.

``AgentRunner`` is the in-process driver Milestone 1 tests use; the worker/
queue topology (Milestone 3) claims queued runs and calls into this same
runner. The test agents here stand in for the reference agents until
Milestone 12.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from vdreamer.agents import BaseAgent, BuildContext, Mode
from vdreamer.context import VDreamerContext, current_context, emit_log, emit_progress
from vdreamer.runtime import AgentRunner, RunInfo


class _State(TypedDict):
    """One node: message in, reply out."""

    message: str
    reply: str


def _reply(state: _State) -> dict[str, Any]:
    return {"reply": f"pong: {state['message']}"}


def _boom(state: _State) -> dict[str, Any]:
    raise ValueError("boom from node")


def _log_deep(state: _State) -> dict[str, Any]:
    """A plain node: the helpers take no parameters and resolve the contextvar."""

    def deep_inside() -> None:
        emit_log("from deep inside a node")
        emit_progress("half", 0.5)

    deep_inside()
    return {"reply": f"pong: {state['message']}"}


class _RecordingAgent(BaseAgent):
    """Counts build() calls and records every hook invocation."""

    name = "recording"
    mode = Mode.CHAT
    models: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    expose_thinking = True
    access_roles: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.build_calls = 0
        self.built_with: BuildContext | None = None
        self.hook_events: list[tuple[str, str]] = []
        self.starts: list[RunInfo] = []
        self.ends: list[tuple[RunInfo, str]] = []

    def build(self, ctx: BuildContext) -> StateGraph:
        self.build_calls += 1
        self.built_with = ctx
        graph = StateGraph(_State)
        graph.add_node("reply", _reply)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph

    async def on_run_start(self, run: Any) -> None:
        self.starts.append(run)
        self.hook_events.append(("start", run.status))

    async def on_run_end(self, run: Any, status: str) -> None:
        self.ends.append((run, status))
        self.hook_events.append(("end", status))


class _FailingAgent(_RecordingAgent):
    name = "failing"

    def build(self, ctx: BuildContext) -> StateGraph:
        self.build_calls += 1
        self.built_with = ctx
        graph = StateGraph(_State)
        graph.add_node("boom", _boom)
        graph.add_edge(START, "boom")
        graph.add_edge("boom", END)
        return graph


class _AsyncDurabilityAgent(_RecordingAgent):
    """A42 opt-in: the whole agent runs with durability='async'."""

    name = "async-durability"
    durability = "async"


class _BogusDurabilityAgent(_RecordingAgent):
    """An agent that misdeclares durability — the runner must reject it."""

    name = "bogus-durability"
    durability = "bogus"


class _LoggingAgent(_RecordingAgent):
    """A node that reaches for the implicit helpers with no parameters."""

    name = "logging-agent"

    def build(self, ctx: BuildContext) -> StateGraph:
        self.build_calls += 1
        self.built_with = ctx
        graph = StateGraph(_State)
        graph.add_node("reply", _log_deep)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph


class _CtxState(TypedDict):
    """A context_schema graph: the node reads the run context."""

    message: str
    reply: str


def _ctx_node(state: _CtxState, runtime: Runtime[VDreamerContext]) -> dict[str, Any]:
    return {"reply": f"{runtime.context.run_id}:{state['message']}"}


class _ContextSchemaAgent(_RecordingAgent):
    """§2.4: run context rides the LangGraph context= channel."""

    name = "ctx-schema"

    def build(self, ctx: BuildContext) -> StateGraph:
        self.build_calls += 1
        self.built_with = ctx
        graph = StateGraph(_CtxState, context_schema=VDreamerContext)
        graph.add_node("reply", _ctx_node)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph


def test_runinfo_is_frozen_with_hook_defaults() -> None:
    info = RunInfo(run_id="r", thread_id="t", agent_name="a")
    assert info.attempt == 0
    assert info.status == "running"
    with pytest.raises(FrozenInstanceError):
        info.status = "completed"


async def test_build_runs_once_and_is_cached() -> None:
    agent = _RecordingAgent()
    checkpointer = InMemorySaver()
    runner = AgentRunner(agent, checkpointer=checkpointer, settings={"k": "v"})
    first = runner.build()
    assert first is runner.build() is runner.compiled
    config: dict[str, Any] = {"configurable": {"thread_id": "t-once"}}
    await runner.arun({"message": "a"}, config=config)
    await runner.arun({"message": "b"}, config=config)
    # §2.2: "build() runs once per worker; graph cached"
    assert agent.build_calls == 1
    # the BuildContext the agent built with carries the runner wiring
    assert agent.built_with is not None
    assert agent.built_with.settings == {"k": "v"}
    assert agent.built_with.checkpointer is checkpointer
    assert agent.built_with.store is None


async def test_arun_returns_the_graph_result_through_the_checkpointer() -> None:
    agent = _RecordingAgent()
    runner = AgentRunner(agent, checkpointer=InMemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": "t-result"}}
    result = await runner.arun({"message": "hello"}, config=config)
    assert result == {"message": "hello", "reply": "pong: hello"}
    state = await runner.compiled.aget_state(config)
    assert state.values["reply"] == "pong: hello"


async def test_hooks_fire_around_the_run() -> None:
    agent = _RecordingAgent()
    runner = AgentRunner(agent, checkpointer=InMemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": "t-hooks"}}
    ctx = VDreamerContext(
        run_id="run_hooks", thread_id="t-hooks", agent_name=agent.name, user_id="u9", attempt=2
    )
    result = await runner.arun({"message": "hi"}, config=config, context=ctx)
    assert result == {"message": "hi", "reply": "pong: hi"}
    assert agent.hook_events == [("start", "running"), ("end", "completed")]
    start_info = agent.starts[0]
    assert (start_info.run_id, start_info.thread_id, start_info.agent_name) == (
        "run_hooks",
        "t-hooks",
        "recording",
    )
    assert start_info.attempt == 2
    assert start_info.status == "running"
    end_info, end_status = agent.ends[0]
    assert end_status == "completed"
    assert end_info.run_id == "run_hooks"


async def test_run_info_falls_back_to_config_then_generates() -> None:
    agent = _RecordingAgent()
    runner = AgentRunner(agent, checkpointer=InMemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": "t-cfg", "run_id": "r-cfg"}}
    await runner.arun({"message": "x"}, config=config)
    assert (agent.starts[0].run_id, agent.starts[0].thread_id) == ("r-cfg", "t-cfg")
    assert agent.starts[0].attempt == 0

    # nothing given: ids are generated, and one execution is one run_id (§3.1)
    # (a checkpointered graph needs a real thread_id in config — langgraph's
    # requirement — so the bare path is exercised without one)
    bare = _RecordingAgent()
    bare_runner = AgentRunner(bare)
    await bare_runner.arun({"message": "x1"})
    await bare_runner.arun({"message": "x2"})
    first, second = bare.starts[0], bare.starts[1]
    assert first.run_id.startswith("run_")
    assert first.thread_id.startswith("thread_")
    assert first.run_id != second.run_id


async def test_failing_node_fails_the_run_and_propagates() -> None:
    agent = _FailingAgent()
    runner = AgentRunner(agent, checkpointer=InMemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": "t-fail"}}
    with pytest.raises(ValueError, match="boom from node"):
        await runner.arun({"message": "x"}, config=config)
    assert agent.hook_events == [("start", "running"), ("end", "failed")]


async def test_durability_default_explicit_and_agent_opt_in() -> None:
    # A42: default "sync" — no attribute, no argument — threaded through and
    # the run completes through the checkpointer.
    agent = _RecordingAgent()
    runner = AgentRunner(agent, checkpointer=InMemorySaver())
    result = await runner.arun(
        {"message": "a"}, config={"configurable": {"thread_id": "t-d1"}}
    )
    assert result["reply"] == "pong: a"
    # explicit arguments override both the default and the agent attribute
    result = await runner.arun(
        {"message": "b"}, config={"configurable": {"thread_id": "t-d2"}}, durability="sync"
    )
    assert result["reply"] == "pong: b"
    result = await runner.arun(
        {"message": "c"}, config={"configurable": {"thread_id": "t-d3"}}, durability="async"
    )
    assert result["reply"] == "pong: c"
    # an agent may opt the whole agent in: durability = "async"
    async_agent = _AsyncDurabilityAgent()
    async_runner = AgentRunner(async_agent, checkpointer=InMemorySaver())
    result = await async_runner.arun(
        {"message": "d"}, config={"configurable": {"thread_id": "t-d4"}}
    )
    assert result["reply"] == "pong: d"


async def test_misdeclared_durability_is_rejected() -> None:
    agent = _BogusDurabilityAgent()
    runner = AgentRunner(agent, checkpointer=InMemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": "t-bad"}}
    with pytest.raises(ValueError, match="durability"):
        await runner.arun({"message": "x"}, config=config)
    # the explicit argument overrides the bogus attribute
    result = await runner.arun({"message": "x"}, config=config, durability="sync")
    assert result["reply"] == "pong: x"
    # and an invalid explicit value is rejected too
    with pytest.raises(ValueError, match="durability"):
        await runner.arun({"message": "x"}, config=config, durability="bogus")  # type: ignore[arg-type]


async def test_implicit_helpers_resolve_inside_nodes_via_arun_context() -> None:
    """The core §2.2-line-70 promise: a helper called deep inside a plain node
    with no parameters resolves the run bound by ``arun(context=...)``."""
    agent = _LoggingAgent()
    runner = AgentRunner(agent, checkpointer=InMemorySaver())
    ctx = VDreamerContext(
        run_id="run_ctx", thread_id="t-ctx", agent_name=agent.name, user_id="u1"
    )
    result = await runner.arun(
        {"message": "hello"}, config={"configurable": {"thread_id": "t-ctx"}}, context=ctx
    )
    assert result == {"message": "hello", "reply": "pong: hello"}
    assert ctx.pending == [
        {"type": "log", "level": "info", "message": "from deep inside a node"},
        {"type": "progress", "label": "half", "pct": 0.5},
    ]
    # the binding is gone once the run ends
    with pytest.raises(RuntimeError, match="no active vdreamer run context"):
        current_context()


async def test_arun_without_context_does_not_bind() -> None:
    agent = _LoggingAgent()
    runner = AgentRunner(agent, checkpointer=InMemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": "t-nobind"}}
    # nothing bound: the node's emit_log raises, which fails the run
    with pytest.raises(RuntimeError, match="no active vdreamer run context"):
        await runner.arun({"message": "x"}, config=config)
    assert agent.hook_events == [("start", "running"), ("end", "failed")]


async def test_context_kwarg_is_safe_on_a_graph_without_context_schema() -> None:
    """The mini-experiment behind ``AgentRunner._execute`` always forwarding
    ``context=``: verified on langgraph 1.2.15 that a graph declaring no
    ``context_schema`` runs unchanged with a non-None context passed through."""
    agent = _RecordingAgent()
    runner = AgentRunner(agent, checkpointer=InMemorySaver())
    ctx = VDreamerContext(run_id="run_plain", thread_id="t-plain", agent_name=agent.name)
    config: dict[str, Any] = {"configurable": {"thread_id": "t-plain"}}
    result = await runner.arun({"message": "x"}, config=config, context=ctx)
    assert result == {"message": "x", "reply": "pong: x"}


async def test_context_flows_into_context_schema_graphs() -> None:
    """§2.4: with ``context_schema=VDreamerContext`` the same ``context=``
    reaches nodes as ``runtime.context``."""
    agent = _ContextSchemaAgent()
    runner = AgentRunner(agent, checkpointer=InMemorySaver())
    ctx = VDreamerContext(run_id="run_schema", thread_id="t-schema", agent_name=agent.name)
    config: dict[str, Any] = {"configurable": {"thread_id": "t-schema"}}
    result = await runner.arun({"message": "m"}, config=config, context=ctx)
    assert result["reply"] == "run_schema:m"
