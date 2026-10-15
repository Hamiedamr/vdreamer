"""The ``BaseAgent`` developer contract (PRD §2.2, D1, A5).

Complete as of Milestone 1 (Developer Contract Core):

- **Attributes** (§2.2): ``name``, ``mode``, ``models``, ``resources``,
  ``input_schema``, ``expose_thinking``, ``access_roles`` — plus the
  Milestone-1 additions ``tool_policies`` / ``interrupt_policies`` (§2.7,
  §4.5) and ``durability`` (§2.4, A42).
- **``build(ctx)``** returns any accepted graph (§2.4): a ``StateGraph``
  (vdreamer compiles it), an already-compiled graph, a functional-API
  ``@entrypoint``, or a ``create_agent`` / ``create_deep_agent`` prebuilt —
  prebuilts get their wiring from ``ctx.agent_kwargs()``.
- **Optional hooks** ``on_run_start`` / ``on_run_end``, fired around each
  run by ``vdreamer.runtime.AgentRunner``.

``vdreamer validate`` (validate.py) checks every piece of this contract.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pydantic

from vdreamer.context import VDreamerContext
from vdreamer.governance import ToolKit
from vdreamer.persistence import DEFAULT_DURABILITY, Durability
from vdreamer.tools import InterruptPolicy, Policy, VDreamerToolMiddleware


class Mode(enum.Enum):
    """How an agent is driven (PRD §2.2): CHAT | HEADLESS | BOTH."""

    CHAT = "chat"
    HEADLESS = "headless"
    BOTH = "both"


def runs_headless(agent: Any, run_headless: bool = False) -> bool:
    """Whether a run is headless (D15): approvals need an ``approver_role``
    (A16) and a ``reject`` ends the run ``rejected`` (§4.3).

    Every run of a ``HEADLESS`` agent is. A ``BOTH`` agent's runs started
    through the chat routes are not; its ``/trigger`` runs are, which the
    run row carries as ``run_headless`` (``RunRecord.headless``).
    """
    return run_headless or getattr(agent, "mode", None) is Mode.HEADLESS


@dataclass
class BuildContext:
    """What ``BaseAgent.build()`` receives (PRD §2.2).

    A worker populates this once per agent before ``build()`` runs, then
    caches the graph ("build() runs once per worker", §2.2). ``settings``
    carries deployment config; ``checkpointer`` / ``store`` are the runtime
    wiring ``agent_kwargs()`` forwards to prebuilt agents.

    ``models`` / ``resources`` hold the agent's declared aliases only
    (``BaseAgent.models`` / ``BaseAgent.resources``, §3.6–§3.7): a LangChain
    chat model or a native fastembed object per model alias, and the native
    pooled client per resource alias (A35), opened at worker start
    (``vdreamer.backends``). Looking up an undeclared alias is a ``KeyError``.

    ``tools`` is the :class:`~vdreamer.governance.ToolKit` (§2.7):
    ``ctx.tools.govern(tools)`` and ``ctx.tools.mcp(server, include=[...])``.
    ``tool_policies`` are the agent's own (``BaseAgent.tool_policies``), applied
    by both the toolkit and the ``agent_kwargs()`` middleware.
    """

    settings: Mapping[str, Any] = field(default_factory=dict)
    models: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    tools: ToolKit = None  # type: ignore[assignment]  # filled in __post_init__
    checkpointer: Any = None
    store: Any = None
    tool_policies: Mapping[str, Policy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tools is None:
            self.tools = ToolKit(self.tool_policies)

    def agent_kwargs(self) -> dict[str, Any]:
        """``create_agent`` / ``create_deep_agent`` kwargs (PRD §2.4).

        Supplies exactly ``{checkpointer, store, context_schema:
        VDreamerContext, middleware: [VDreamerToolMiddleware(...)]}`` — the
        run-context schema (nodes read the run through ``runtime.context``)
        plus the tool-governance middleware (§2.7) carrying the agent's
        ``tool_policies`` — so a prebuilt agent wires into vdreamer without
        touching internals.
        """
        return {
            "checkpointer": self.checkpointer,
            "store": self.store,
            "context_schema": VDreamerContext,
            "middleware": [VDreamerToolMiddleware(self.tool_policies)],
        }


class BaseAgent:
    """An agent served by vdreamer (PRD §2.2).

    Subclasses declare the attribute contract and implement ``build()``.
    Nothing magic: plain class attributes, a plain method.
    """

    name: str = ""
    mode: Mode = Mode.CHAT
    models: Sequence[str] = ()
    resources: Sequence[str] = ()
    input_schema: type[pydantic.BaseModel] | None = None
    expose_thinking: bool = True
    access_roles: Sequence[str] = ()

    tool_policies: Mapping[str, Policy] = {}
    """Tool governance policies keyed by tool name or ``fnmatch`` glob
    (PRD §2.7), e.g. ``tool_policies = {"sql_db_*": Policy(side_effects=True)}``.
    Class-level only: subclasses *replace* the whole mapping, so the shared
    empty default is never mutated; resolution against a decorator's own
    policy (per field, agent wins) is ``vdreamer.tools.resolve_policy``.
    """

    interrupt_policies: Mapping[str, InterruptPolicy] = {}
    """HITL overrides for non-vdreamer interrupt sources (PRD §4.5, A44),
    keyed by ``kind`` — ``"tool_review"`` | ``"mcp_elicitation"`` |
    ``"generic"`` — or by tool name. Class-level only, replaced wholesale
    like ``tool_policies``.
    """

    durability: Durability = DEFAULT_DURABILITY
    """Per-superstep invocation durability (PRD §2.4, A42): ``"sync"`` by
    default so every step is checkpointed and crash recovery (A6) resumes
    from the last checkpoint; set ``durability = "async"`` per agent to opt
    out. Resolved per run by ``vdreamer.runtime.AgentRunner``.
    """

    def build(self, ctx: BuildContext) -> Any:
        """Return this agent's LangGraph graph (PRD §2.4). Runs once per worker."""
        raise NotImplementedError("agents must implement build(ctx)")

    async def on_run_start(self, run: Any) -> None:
        """Optional lifecycle hook (PRD §2.2), fired by ``runtime.AgentRunner``.

        ``run`` is a ``RunInfo`` with ``status="running"``, before the graph
        executes. The worker (Milestone 3) maps interrupts onto
        ``awaiting_approval`` between these hooks.
        """

    async def on_run_end(self, run: Any, status: str) -> None:
        """Optional lifecycle hook (PRD §2.2), fired by ``runtime.AgentRunner``.

        ``status`` is ``"completed"``, or ``"failed"`` with the exception
        already re-raised to the caller; ``run`` is the same ``RunInfo``
        with matching ``status``.
        """
