"""``vdreamer validate`` — developer-contract checks (PRD §2.3).

Milestone 0 loaded the agent, checked the attribute contract (PRD §2.2), and
built and compiled the graph (PRD §2.4). Milestone 1 adds the new-contract
checks — ``tool_policies`` / ``interrupt_policies`` shape (§2.7, §4.5),
per-agent ``durability`` (§2.4, A42), and the ``agent_kwargs()`` wiring
proof — plus the D2 discovery flags on the CLI: ``--registry`` /
``--agents-dir`` resolve a bare name through
``vdreamer.discovery.resolve_agent``, while a ``pkg.module:agent`` spec stays
explicit (it short-circuits inside ``resolve_agent``). Alias resolution and
model-file probes (Milestone 7) and approval-route wiring (Milestone 8)
extend this module as they land.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydantic
from langgraph.graph import StateGraph

from vdreamer.agents import BaseAgent, BuildContext, Mode
from vdreamer.context import VDreamerContext
from vdreamer.discovery import AgentLoadError, resolve_agent
from vdreamer.persistence import DEFAULT_DURABILITY
from vdreamer.tools import InterruptPolicy, Policy, VDreamerToolMiddleware

#: Agent name == route segment (PRD §2.5): a lowercase slug.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")

#: The exact ``BuildContext.agent_kwargs()`` key set (PRD §2.4).
_AGENT_KWARGS_KEYS: tuple[str, ...] = ("checkpointer", "store", "context_schema", "middleware")


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


def _check_policy_mapping(
    label: str,
    value: Any,
    policy_cls: type[Policy] | type[InterruptPolicy],
) -> Check:
    """``tool_policies`` / ``interrupt_policies`` shape (PRD §2.7, §4.5).

    Class-level mappings keyed by tool name or ``fnmatch`` glob: keys must be
    non-empty strings (glob *syntax* is not pre-validated) and values
    ``policy_cls`` instances — the common mistake is a raw metadata dict
    where a policy object belongs, so the failure names the offender.
    """
    if not isinstance(value, Mapping):
        return Check(
            label,
            False,
            f"expected a mapping of name/glob to {policy_cls.__name__}, got {type(value).__name__}",
        )
    bad_keys = [key for key in value if not isinstance(key, str) or not key]
    if bad_keys:
        return Check(
            label,
            False,
            f"keys must be non-empty strings (name or fnmatch glob): {bad_keys!r}",
        )
    bad = [(key, pol) for key, pol in value.items() if not isinstance(pol, policy_cls)]
    if bad:
        key, offender = bad[0]
        return Check(
            label,
            False,
            f"{key!r} -> {offender!r}: values must be {policy_cls.__name__} instances",
        )
    return Check(label, True, ", ".join(str(key) for key in value) if value else "(none)")


def _check_durability(agent: BaseAgent) -> Check:
    """``durability`` must be ``"sync"`` or ``"async"`` (PRD §2.4, A42)."""
    value = getattr(agent, "durability", DEFAULT_DURABILITY)
    if value in ("sync", "async"):
        return Check("durability", True, str(value))
    return Check(
        "durability",
        False,
        "durability must be 'sync' or 'async' (A42: 'sync' by default so every step is "
        f"checkpointed; set durability = 'async' per agent to opt out), got {value!r}",
    )


def _check_agent_kwargs() -> Check:
    """Cheap wiring proof for ``BuildContext.agent_kwargs()`` (PRD §2.4).

    Calls it on an empty ``BuildContext`` and asserts the exact key set with
    ``context_schema=VDreamerContext`` and a ``VDreamerToolMiddleware`` in
    ``middleware`` — proving the prebuilt-agent path without importing
    ``create_agent`` or a model server.
    """
    try:
        kwargs = BuildContext().agent_kwargs()
    except Exception as exc:  # a broken override is a failed check, not a crash
        return Check("agent_kwargs", False, f"{type(exc).__name__}: {exc}")
    if not isinstance(kwargs, dict) or set(kwargs) != set(_AGENT_KWARGS_KEYS):
        shape = sorted(kwargs) if isinstance(kwargs, dict) else type(kwargs).__name__
        return Check(
            "agent_kwargs", False, f"expected exactly {sorted(_AGENT_KWARGS_KEYS)}, got {shape}"
        )
    if kwargs["context_schema"] is not VDreamerContext:
        return Check(
            "agent_kwargs",
            False,
            f"context_schema must be VDreamerContext, got {kwargs['context_schema']!r}",
        )
    middleware = kwargs["middleware"]
    has_middleware = isinstance(middleware, list) and any(
        isinstance(m, VDreamerToolMiddleware) for m in middleware
    )
    if not has_middleware:
        return Check(
            "agent_kwargs",
            False,
            "middleware must be a list containing a VDreamerToolMiddleware instance",
        )
    return Check("agent_kwargs", True, ", ".join(sorted(_AGENT_KWARGS_KEYS)))


def validate_agent(agent: BaseAgent) -> list[Check]:
    """Run the developer-contract checks (Milestones 0+1) against one agent."""
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
        _check_policy_mapping("tool_policies", agent.tool_policies, Policy),
        _check_policy_mapping("interrupt_policies", agent.interrupt_policies, InterruptPolicy),
        _check_durability(agent),
        _check_agent_kwargs(),
    ]
    return checks


def run_validate(
    spec: str,
    *,
    registry: str | Path | None = None,
    agents_dir: str | Path | None = None,
) -> int:
    """CLI entry point for ``vdreamer validate``. 0 = valid, 1 = invalid.

    ``spec`` is a ``pkg.module:agent`` spec (explicit — ``registry`` and
    ``agents_dir`` are ignored), or a bare name resolved D2-style through
    ``vdreamer.discovery.resolve_agent``: registry first, then agents dir.
    """
    try:
        agent = resolve_agent(spec, registry=registry, agents_dir=agents_dir)
    except AgentLoadError as exc:  # RegistryError subclasses AgentLoadError
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
