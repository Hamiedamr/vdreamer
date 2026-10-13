"""The ``BaseAgent`` developer contract (PRD §2.2, D1, A5).

Milestone 0 ships the attribute contract plus the ``build()`` hook that
``vdreamer validate`` and the CI smoke tests exercise. The runtime half of the
contract — ``BuildContext`` population, ``VDreamerContext``, ``agent_kwargs()``,
run hooks, tool/interrupt policies — lands with Milestone 1 and later.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pydantic


class Mode(enum.Enum):
    """How an agent is driven (PRD §2.2): CHAT | HEADLESS | BOTH."""

    CHAT = "chat"
    HEADLESS = "headless"
    BOTH = "both"


@dataclass
class BuildContext:
    """What ``BaseAgent.build()`` receives (PRD §2.2).

    A worker populates this once per agent before ``build()`` runs, then caches
    the graph. Milestone 0 only needs an empty context for ``validate`` and the
    smoke tests; workers start populating it in Milestone 1 (models, resources,
    tools, checkpointer, store, settings), and ``agent_kwargs()`` arrives with
    the prebuilt-agent support of Milestone 1.
    """

    settings: Mapping[str, Any] = field(default_factory=dict)
    models: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    tools: Any = None
    checkpointer: Any = None
    store: Any = None

    def agent_kwargs(self) -> dict[str, Any]:
        """``create_agent`` / ``create_deep_agent`` kwargs (PRD §2.4)."""
        raise NotImplementedError("agent_kwargs() lands with Milestone 1")


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

    def build(self, ctx: BuildContext) -> Any:
        """Return this agent's LangGraph graph (PRD §2.4). Runs once per worker."""
        raise NotImplementedError("agents must implement build(ctx)")

    async def on_run_start(self, run: Any) -> None:
        """Optional lifecycle hook; wired into runs in Milestone 1."""

    async def on_run_end(self, run: Any, status: str) -> None:
        """Optional lifecycle hook; wired into runs in Milestone 1."""
