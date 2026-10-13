"""``vdreamer validate`` — developer-contract checks (PRD §2.3).

Milestone 0 scope: load the agent, check the attribute contract (PRD §2.2),
build the graph and compile it (PRD §2.4). Alias resolution and model-file
probes (Milestone 7), approval-route wiring (Milestone 8), and checkpointer /
governed-tool checks (Milestones 1, 5) extend this module as they land.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pydantic
from langgraph.graph import StateGraph

from vdreamer.agents import BaseAgent, BuildContext, Mode
from vdreamer.discovery import AgentLoadError, load_agent

#: Agent name == route segment (PRD §2.5): a lowercase slug.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


@dataclass(frozen=True)
class Check:
    """One validation result."""

    name: str
    ok: bool
    detail: str = ""


def ensure_compiled(graph: Any, checkpointer: Any = None) -> Any:
    """Compile a ``build()`` return value where needed (PRD §2.4).

    ``StateGraph`` instances compile here; already-runnable graphs (compiled
    graphs, functional-API ``@entrypoint`` graphs, ``create_agent`` /
    ``create_deep_agent`` prebuilts) pass through untouched. Anything else is
    rejected — no guessing, no duck-typing beyond "can it run".
    """
    if isinstance(graph, StateGraph):
        return graph.compile(checkpointer=checkpointer)
    if callable(getattr(graph, "ainvoke", None)):
        return graph
    raise TypeError(
        "build() must return a StateGraph or a runnable graph (compiled "
        "StateGraph, functional-API @entrypoint, create_agent, or "
        f"create_deep_agent); got {type(graph).__name__}"
    )


def _check_str_seq(label: str, value: Any) -> Check:
    """A declared sequence of str alias names."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return Check(label, False, f"expected a sequence of names, got {type(value).__name__}")
    bad = [entry for entry in value if not isinstance(entry, str)]
    if bad:
        return Check(label, False, f"non-string entries: {bad!r}")
    return Check(label, True, ", ".join(value) if value else "(none)")


def _check_input_schema(agent: BaseAgent) -> Check:
    """Headless/both agents must declare a pydantic input schema (D16, A16)."""
    needs_schema = agent.mode in (Mode.HEADLESS, Mode.BOTH)
    schema = agent.input_schema
    if schema is None:
        return Check(
            "input_schema",
            not needs_schema,
            "required for HEADLESS/BOTH" if needs_schema else "not required for CHAT",
        )
    if not (isinstance(schema, type) and issubclass(schema, pydantic.BaseModel)):
        return Check(
            "input_schema",
            False,
            f"expected a pydantic.BaseModel subclass, got {schema!r}",
        )
    try:
        schema.model_json_schema()
    except Exception as exc:  # surfaced as a failed check
        return Check("input_schema", False, f"model_json_schema() failed: {exc}")
    return Check("input_schema", True, schema.__name__)


def _check_graph(agent: BaseAgent) -> Check:
    """``build()`` must return something vdreamer can compile or run."""
    try:
        graph = agent.build(BuildContext())
        compiled = ensure_compiled(graph)
    except NotImplementedError:
        return Check("graph.build", False, "build(ctx) is not implemented")
    except Exception as exc:
        return Check("graph.build", False, f"{type(exc).__name__}: {exc}")
    return Check("graph.build", True, type(compiled).__module__ + "." + type(compiled).__qualname__)


def validate_agent(agent: BaseAgent) -> list[Check]:
    """Run the Milestone-0 contract checks against one agent."""
    name = getattr(agent, "name", "")
    checks = [
        Check(
            "name",
            isinstance(name, str) and bool(_NAME_RE.fullmatch(name)),
            repr(name) if not isinstance(name, str) or not _NAME_RE.fullmatch(name) else name,
        ),
        Check(
            "mode",
            isinstance(agent.mode, Mode),
            agent.mode.value if isinstance(agent.mode, Mode) else repr(agent.mode),
        ),
        _check_input_schema(agent),
        _check_str_seq("models", agent.models),
        _check_str_seq("resources", agent.resources),
        _check_str_seq("access_roles", agent.access_roles),
        Check(
            "expose_thinking",
            isinstance(agent.expose_thinking, bool),
            repr(agent.expose_thinking),
        ),
        _check_graph(agent),
    ]
    return checks


def run_validate(spec: str) -> int:
    """CLI entry point for ``vdreamer validate``. 0 = valid, 1 = invalid."""
    try:
        agent = load_agent(spec)
    except AgentLoadError as exc:
        print(f"validate: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # agent code raised while importing; surface it, don't crash
        print(f"validate: cannot load {spec!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    checks = validate_agent(agent)
    for check in checks:
        print(f"[{'ok' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    failed = [check for check in checks if not check.ok]
    label = "PASS" if not failed else f"FAIL ({len(failed)} check{'s' if len(failed) > 1 else ''})"
    print(f"validate {agent.__class__.__name__}: {label}")
    return 0 if not failed else 1
