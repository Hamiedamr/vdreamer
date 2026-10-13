"""The Milestone-1 exit criterion (docs/tasks.md) as a real test.

``uv run pytest tests/contract`` is the documented exit command; this module
is the criterion: a trivial in-repo agent builds, compiles, and runs a single
node through the checkpointer, firing the lifecycle hooks, resolving the
implicit helpers through the runner-bound context, and carrying the §2.4
``agent_kwargs()`` wiring — then passes ``validate_agent`` with every check
green.

Deliberately *not* marked ``smoke``: ``pytest -m smoke`` stays the exact
floor/latest CI subset (A41), which is one test
(``tests/contract/test_contract.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode
from vdreamer.context import VDreamerContext, emit_log
from vdreamer.persistence import DEFAULT_DURABILITY
from vdreamer.runtime import AgentRunner
from vdreamer.tools import InterruptPolicy, Policy, VDreamerToolMiddleware, resolve_policy
from vdreamer.validate import validate_agent


class _ExitInput(BaseModel):
    """Input body; declared because mode is BOTH (D16, A16)."""

    message: str = ""


class _ExitState(TypedDict):
    """Graph state: one node."""

    message: str
    reply: str


@vdreamer.tool(redact=["message"], timeout_s=5)
def _echo(message: str) -> str:
    """Echo the message back."""
    return message


def _node(state: _ExitState) -> dict[str, Any]:
    # The implicit helper takes no parameters: it resolves the run through the
    # contextvar the runner binds (PRD §2.2 line 70).
    emit_log("exit criterion node ran")
    return {"reply": f"pong: {state['message']}"}


class _ExitAgent(BaseAgent):
    """A trivial agent exercising the Milestone-1 contract surface."""

    name = "exit-criterion"
    mode = Mode.BOTH
    models: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    input_schema = _ExitInput
    expose_thinking = True
    access_roles: tuple[str, ...] = ()
    # §2.7: keyed by tool name or glob; agent policy overrides the decorator's
    # per field (resolve_policy).
    tool_policies: Mapping[str, Policy] = {"_echo": Policy(timeout_s=7.0)}
    # §4.5: keyed by kind for non-vdreamer interrupt sources.
    interrupt_policies: Mapping[str, InterruptPolicy] = {"generic": InterruptPolicy(ttl_hours=48)}

    def __init__(self) -> None:
        self.build_calls = 0
        self.hook_events: list[tuple[str, str]] = []

    def build(self, ctx: BuildContext) -> StateGraph:
        self.build_calls += 1
        graph = StateGraph(_ExitState)
        graph.add_node("reply", _node)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph

    async def on_run_start(self, run: Any) -> None:
        self.hook_events.append(("start", run.status))

    async def on_run_end(self, run: Any, status: str) -> None:
        self.hook_events.append(("end", status))


async def test_milestone1_exit_criterion() -> None:
    agent = _ExitAgent()
    assert agent.durability == DEFAULT_DURABILITY == "sync"  # A42 default
    runner = AgentRunner(agent, checkpointer=InMemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": "t-exit"}}
    ctx = VDreamerContext(
        run_id="run_exit",
        thread_id="t-exit",
        agent_name=agent.name,
        user_id="u1",
        roles=("netops",),
    )

    # The criterion: a single node runs through the checkpointer.
    result = await runner.arun({"message": "milestone-1"}, config=config, context=ctx)
    assert result == {"message": "milestone-1", "reply": "pong: milestone-1"}

    # build() ran exactly once and the compiled graph is cached (§2.2).
    assert agent.build_calls == 1
    assert runner.compiled is not None

    # Checkpoint state is persisted for the thread.
    state = await runner.compiled.aget_state(config)
    assert state.values["reply"] == "pong: milestone-1"

    # Hooks fired around the run with the run identity.
    assert agent.hook_events == [("start", "running"), ("end", "completed")]

    # The emit_log() call inside the node resolved the contextvar-bound run.
    assert {"type": "log", "level": "info", "message": "exit criterion node ran"} in ctx.pending

    # §2.4 agent_kwargs wiring, with checkpointer/store passed through.
    checkpointer: Any = object()
    store: Any = object()
    kwargs = BuildContext(checkpointer=checkpointer, store=store).agent_kwargs()
    assert set(kwargs) == {"checkpointer", "store", "context_schema", "middleware"}
    assert kwargs["checkpointer"] is checkpointer
    assert kwargs["store"] is store
    assert kwargs["context_schema"] is VDreamerContext
    assert isinstance(kwargs["middleware"], list)
    assert any(isinstance(m, VDreamerToolMiddleware) for m in kwargs["middleware"])

    # §2.7: the decorated tool carries its policy in metadata; the agent
    # mapping overlays the decorator's per field (timeout overridden,
    # redaction kept), and the top-level decorator produced a BaseTool.
    assert isinstance(_echo, BaseTool)  # A37: vdreamer adds no tool base class
    decorated = Policy.from_metadata(_echo.metadata[vdreamer.VDREAMER_METADATA_KEY])
    resolved = resolve_policy("_echo", decorated, agent.tool_policies)
    assert resolved.redact == ("message",)  # kept from the decorator
    assert resolved.timeout_s == 7.0  # overridden by the agent mapping

    # The full contract passes: every check (M0 + M1) is green.
    checks = validate_agent(agent)
    assert [check.name for check in checks] == [
        "name",
        "mode",
        "input_schema",
        "models",
        "resources",
        "access_roles",
        "expose_thinking",
        "graph.build",
        "tool_policies",
        "interrupt_policies",
        "durability",
        "agent_kwargs",
    ]
    assert [(check.name, check.detail) for check in checks if not check.ok] == []
