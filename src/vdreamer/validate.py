"""``vdreamer validate`` — developer-contract checks (PRD §2.3).

Milestone 0 loaded the agent, checked the attribute contract (PRD §2.2), and
built and compiled the graph (PRD §2.4). Milestone 1 adds the new-contract
checks — ``tool_policies`` / ``interrupt_policies`` shape (§2.7, §4.5),
per-agent ``durability`` (§2.4, A42), and the ``agent_kwargs()`` wiring
proof — plus the D2 discovery flags on the CLI: ``--registry`` /
``--agents-dir`` resolve a bare name through
``vdreamer.discovery.resolve_agent``, while a ``pkg.module:agent`` spec stays
explicit (it short-circuits inside ``resolve_agent``). Alias resolution and
model-file probes (Milestone 7) extend this module as they land.

Milestone 5 adds the tool checks (§2.3): **no ungoverned tools** — every
``ToolNode`` in the compiled graph (subgraphs included) either holds only
``GovernedTool`` proxies or runs under ``VDreamerToolMiddleware`` — and
**approval routes wired** for per-tool gates: a headless agent's approval
needs an ``approver_role`` (A16). Milestone 8 extends ``approval.routes`` to
every ``ApprovalGate`` (no offered action unwired) and to headless
``HumanInTheLoopMiddleware`` reviews. ``--resources`` lets ``build()`` call
``ctx.tools.mcp(...)``: the servers are declared but not contacted, so their
tool lists are empty here.

Milestone 7 adds the alias checks (§3.6, §3.7): every alias in
``BaseAgent.models`` must be in ``models.yaml`` (``--models`` or
``$VDREAMER_MODELS``) and every alias in ``BaseAgent.resources`` in
``resources.yaml``. Each declared alias is then probed — a chat endpoint
must answer and serve the configured model, a fastembed model's files must
be in the cache, a resource must connect and pass its probe — and
``--no-probe`` keeps validation off the network (model files and ``*_env``
variables are still checked). ``build()`` receives the real chat models and
fastembed objects (located, not loaded) and an unconnected placeholder per
resource.

Milestone 10 adds **vdreamer checkpointer used** (§2.3, §2.4): ``build()``
runs with a stand-in checkpointer and store on ``ctx``; a ``StateGraph`` is
compiled by vdreamer, and anything ``build()`` compiled itself (a compiled
graph, ``create_agent``, an ``@entrypoint``) must have been given exactly
``ctx.checkpointer`` (``**ctx.agent_kwargs()``) — none, or its own saver,
would lose crash recovery (A6) and resume.
"""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pydantic
from langgraph.graph import StateGraph

from vdreamer.agents import BaseAgent, BuildContext, Mode
from vdreamer.approval import approval_gates
from vdreamer.context import VDreamerContext
from vdreamer.discovery import AgentLoadError, resolve_agent
from vdreamer.governance import GovernedTool, ToolKit, effective_policy
from vdreamer.persistence import DEFAULT_DURABILITY
from vdreamer.resources import ResourcesError
from vdreamer.tools import InterruptPolicy, Policy, VDreamerToolMiddleware

if TYPE_CHECKING:
    from vdreamer.backends import BackendConfig
    from vdreamer.mcp import MCPRegistry
    from vdreamer.models import AliasMap

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


def ensure_compiled(graph: Any, checkpointer: Any = None, store: Any = None) -> Any:
    """Compile a ``build()`` return value where needed (PRD §2.4).

    ``StateGraph`` instances compile here; already-runnable graphs (compiled
    graphs, functional-API ``@entrypoint`` graphs, ``create_agent`` /
    ``create_deep_agent`` prebuilts) pass through untouched. Anything else is
    rejected — no guessing, no duck-typing beyond "can it run".
    """
    if isinstance(graph, StateGraph):
        return graph.compile(checkpointer=checkpointer, store=store)
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


def _build_context(
    agent: BaseAgent,
    mcp: MCPRegistry | None,
    models: Mapping[str, Any] | None = None,
    resources: Mapping[str, Any] | None = None,
) -> BuildContext:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    policies = agent.tool_policies if isinstance(agent.tool_policies, Mapping) else {}
    return BuildContext(
        tools=ToolKit(policies, mcp=mcp),
        tool_policies=policies,
        models=models if models is not None else {},
        resources=resources if resources is not None else {},
        # Stand-ins for the worker's savers: identity is what _check_checkpointer tests.
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )


def _check_checkpointer(graph: Any, compiled: Any, ctx: BuildContext) -> Check:
    """vdreamer's checkpointer (and store) reach the graph (§2.4, A6)."""
    if isinstance(graph, StateGraph):
        return Check("checkpointer", True, "StateGraph compiled by vdreamer")
    used = getattr(compiled, "checkpointer", None)
    hint = "pass ctx.checkpointer — e.g. create_agent(..., **ctx.agent_kwargs())"
    if used is None or used is False:
        return Check(
            "checkpointer", False, f"build() compiled the graph without a checkpointer: {hint}"
        )
    if used is not ctx.checkpointer:
        return Check(
            "checkpointer",
            False,
            f"build() compiled the graph with its own {type(used).__name__}: {hint}",
        )
    store = getattr(compiled, "store", None)
    if store is not None and store is not ctx.store:
        return Check(
            "checkpointer",
            False,
            f"build() compiled the graph with its own {type(store).__name__} store: "
            "pass ctx.store",
        )
    return Check("checkpointer", True, "ctx.checkpointer")


def _check_graph(
    agent: BaseAgent,
    mcp: MCPRegistry | None = None,
    models: Mapping[str, Any] | None = None,
    resources: Mapping[str, Any] | None = None,
) -> tuple[Check, Any, Check | None]:
    """``build()`` must return something vdreamer can compile or run.

    Also returns the checkpointer check (``None`` when the build failed).
    """
    ctx = _build_context(agent, mcp, models, resources)
    try:
        graph = agent.build(ctx)
        compiled = ensure_compiled(graph, checkpointer=ctx.checkpointer, store=ctx.store)
    except NotImplementedError:
        return Check("graph.build", False, "build(ctx) is not implemented"), None, None
    except Exception as exc:
        return Check("graph.build", False, f"{type(exc).__name__}: {exc}"), None, None
    kind = type(compiled).__module__ + "." + type(compiled).__qualname__
    return Check("graph.build", True, kind), compiled, _check_checkpointer(graph, compiled, ctx)


def _references(obj: Any, target: type, depth: int = 0, seen: set[int] | None = None) -> bool:
    """Whether a (composed, traced) wrapper function closes over a ``target`` instance."""
    seen = seen if seen is not None else set()
    if obj is None or id(obj) in seen or depth > 8:
        return False
    seen.add(id(obj))
    if isinstance(obj, target):
        return True
    if isinstance(obj, (list, tuple)):
        return any(_references(item, target, depth + 1, seen) for item in obj)
    for attr in ("__self__", "__wrapped__", "__func__"):
        if _references(getattr(obj, attr, None), target, depth + 1, seen):
            return True
    for cell in getattr(obj, "__closure__", None) or ():
        try:
            contents = cell.cell_contents
        except ValueError:  # an empty cell
            continue
        if _references(contents, target, depth + 1, seen):
            return True
    return False


def tool_nodes(compiled: Any) -> list[Any]:
    """Every LangGraph ``ToolNode`` in a compiled graph and its subgraphs."""
    from langgraph.prebuilt import ToolNode

    graphs = [compiled]
    try:
        graphs += [sub for _ns, sub in compiled.get_subgraphs(recurse=True)]
    except Exception:  # not a Pregel graph (e.g. a bare runnable)
        pass
    found: list[Any] = []
    for graph in graphs:
        for node in (getattr(graph, "nodes", None) or {}).values():
            bound = getattr(node, "bound", None)
            if isinstance(bound, ToolNode) and bound not in found:
                found.append(bound)
    return found


def _middleware_governed(node: Any) -> bool:
    return _references(getattr(node, "_awrap_tool_call", None), VDreamerToolMiddleware) or (
        _references(getattr(node, "_wrap_tool_call", None), VDreamerToolMiddleware)
    )


def _check_tools_governed(compiled: Any) -> Check:
    """§2.3: no ungoverned tools — each one is a proxy or runs under the middleware."""
    ungoverned: list[str] = []
    count = 0
    for node in tool_nodes(compiled):
        tools = list(node.tools_by_name.values())
        count += len(tools)
        if _middleware_governed(node):
            continue
        ungoverned += [tool.name for tool in tools if not isinstance(tool, GovernedTool)]
    if ungoverned:
        return Check(
            "tools.governed",
            False,
            f"ungoverned tools {sorted(ungoverned)}: wrap them with ctx.tools.govern(...) or "
            "build the agent with ctx.agent_kwargs() (VDreamerToolMiddleware)",
        )
    return Check("tools.governed", True, f"{count} tool(s)" if count else "(no tool nodes)")


def _check_approval_routes(agent: BaseAgent, compiled: Any) -> Check:
    """§2.3 approval routes wired, and A16's ``approver_role`` for headless agents.

    Covers per-tool gates (``Policy.approval``), every ``ApprovalGate`` (an
    offered action with no node behind it is unwired — a ``HEADLESS`` agent's
    ``reject`` needs none, the run ends ``rejected``), and
    ``HumanInTheLoopMiddleware`` reviews (role via ``interrupt_policies``).
    """
    headless = agent.mode in (Mode.HEADLESS, Mode.BOTH)
    policies = agent.tool_policies if isinstance(agent.tool_policies, Mapping) else {}
    gated: list[str] = []
    missing: list[str] = []
    for node in tool_nodes(compiled):
        for tool in node.tools_by_name.values():
            policy = tool.policy if isinstance(tool, GovernedTool) else effective_policy(
                tool, policies
            )
            if policy.approval is None:
                continue
            gated.append(tool.name)
            if headless and not policy.approval.approver_role:
                missing.append(tool.name)
    problems: list[str] = []
    if missing:
        problems.append(
            f"headless approval gates without an approver_role (A16): {sorted(missing)}"
        )
    for gate in approval_gates(compiled):
        kind = gate.get("kind")
        gated.append(f"ApprovalGate({kind})")
        routes = gate.get("routes") or {}
        unwired = [
            action
            for action, node in routes.items()
            if node is None and not (agent.mode is Mode.HEADLESS and action == "reject")
        ]
        if unwired:
            problems.append(f"ApprovalGate({kind!r}) has unwired routes: {unwired}")
        if headless and not gate.get("approver_role"):
            problems.append(f"headless ApprovalGate({kind!r}) without an approver_role (A16)")
    if headless and _has_hitl_middleware(compiled) and not _review_role(agent):
        problems.append(
            "headless HumanInTheLoopMiddleware without an approver_role: set "
            "interrupt_policies['tool_review'] = InterruptPolicy(approver_role=...) (A16)"
        )
    if problems:
        return Check("approval.routes", False, "; ".join(problems))
    return Check("approval.routes", True, ", ".join(sorted(gated)) if gated else "(no gates)")


def _has_hitl_middleware(compiled: Any) -> bool:
    """A ``create_agent`` graph with ``HumanInTheLoopMiddleware`` (its node names say so)."""
    graphs = [compiled]
    try:
        graphs += [sub for _ns, sub in compiled.get_subgraphs(recurse=True)]
    except Exception:
        pass
    return any(
        str(name).startswith("HumanInTheLoopMiddleware")
        for graph in graphs
        for name in (getattr(graph, "nodes", None) or {})
    )


def _review_role(agent: BaseAgent) -> bool:
    """Every tool review gets an approver_role from ``interrupt_policies`` (§4.5)."""
    policies = agent.interrupt_policies if isinstance(agent.interrupt_policies, Mapping) else {}
    policy = policies.get("tool_review")
    return isinstance(policy, InterruptPolicy) and bool(policy.approver_role)


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


def static_checks(agent: BaseAgent) -> list[Check]:
    """The contract checks that need no ``build()`` — declarations and policies.

    :func:`validate_agent` reports these alongside the graph checks; the
    process entrypoints (``dev``/``worker``/``serve``) run them at boot and
    refuse an agent that fails any, so nothing serves what ``validate`` rejects.
    """
    name = getattr(agent, "name", "")
    return [
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
        _check_policy_mapping("tool_policies", agent.tool_policies, Policy),
        _check_policy_mapping("interrupt_policies", agent.interrupt_policies, InterruptPolicy),
        _check_durability(agent),
    ]


def validate_agent(
    agent: BaseAgent,
    *,
    mcp: MCPRegistry | None = None,
    models: Mapping[str, Any] | None = None,
    resources: Mapping[str, Any] | None = None,
) -> list[Check]:
    """Run the developer-contract checks (Milestones 0, 1, 5) against one agent.

    ``models`` / ``resources`` are what ``build()`` sees as ``ctx.models`` /
    ``ctx.resources``; the alias checks themselves are :func:`check_aliases`.
    """
    graph_check, compiled, checkpointer_check = _check_graph(agent, mcp, models, resources)
    static = static_checks(agent)
    # validate's report order: the declarations, the graph, then the policies.
    checks = [*static[:7], graph_check, *static[7:], _check_agent_kwargs()]
    if checkpointer_check is not None:
        checks.append(checkpointer_check)
    if compiled is not None:
        checks += [_check_tools_governed(compiled), _check_approval_routes(agent, compiled)]
    return checks


def _declared(agent: BaseAgent, attr: str) -> list[str]:
    value = getattr(agent, attr, ())
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [alias for alias in value if isinstance(alias, str)]


async def check_aliases(
    agent: BaseAgent, config: BackendConfig, *, probe: bool = True
) -> tuple[list[Check], AliasMap, AliasMap]:
    """§3.6/§3.7: each declared alias resolves and is reachable.

    Returns the checks plus what ``build()`` gets: the chat models and the
    located fastembed objects, and an :class:`Unconnected` per resource.
    """
    from vdreamer.adapters import probe_resource
    from vdreamer.backends import Unconnected
    from vdreamer.models import AliasMap, build_models, probe_model

    checks: list[Check] = []
    models: dict[str, Any] = {}
    for alias in _declared(agent, "models"):
        if config.models is None or alias not in config.models.aliases:
            known = sorted(config.models.aliases) if config.models is not None else []
            where = "no models.yaml (--models / $VDREAMER_MODELS)" if config.models is None else (
                f"models.yaml has {known or 'none'}"
            )
            checks.append(Check(f"model:{alias}", False, f"unknown alias ({where})"))
            continue
        try:
            detail = await probe_model(
                config.models, alias, allowlist=config.allowlist, network=probe
            )
            built = await asyncio.to_thread(
                build_models, config.models, [alias], lazy_embeddings=True
            )
            models.update(built)
        except Exception as exc:  # ModelUnavailable, ModelsError, a client refusing its kwargs
            checks.append(Check(f"model:{alias}", False, str(exc)))
            continue
        checks.append(Check(f"model:{alias}", True, detail))
    resources: dict[str, Any] = {}
    for alias in _declared(agent, "resources"):
        if alias not in config.resources:
            where = "no resources.yaml (--resources / $VDREAMER_RESOURCES)" if not (
                config.resources
            ) else f"resources.yaml has {sorted(config.resources)}"
            checks.append(Check(f"resource:{alias}", False, f"unknown alias ({where})"))
            continue
        adapter = config.adapters.get(alias)
        if adapter is None:  # an MCP server: its tools come through ctx.tools.mcp(...)
            checks.append(Check(f"resource:{alias}", True, "mcp server (tools)"))
            continue
        resources[alias] = Unconnected(alias)
        if not probe:
            checks.append(Check(f"resource:{alias}", True, f"{adapter.type} (not contacted)"))
            continue
        try:
            detail = await probe_resource(adapter)
        except Exception as exc:
            checks.append(Check(f"resource:{alias}", False, str(exc)))
            continue
        checks.append(Check(f"resource:{alias}", True, detail))
    owner = f"agent {getattr(agent, 'name', '')!r}"
    return checks, AliasMap("model", owner, models), AliasMap("resource", owner, resources)


def run_validate(
    spec: str,
    *,
    registry: str | Path | None = None,
    agents_dir: str | Path | None = None,
    resources: str | Path | None = None,
    models: str | Path | None = None,
    probe: bool = True,
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

    from vdreamer.backends import load_backend_config
    from vdreamer.mcp import MCPRegistry, mcp_servers_from_resources
    from vdreamer.models import ModelsError
    from vdreamer.network import NetworkAllowlist

    allowlist = NetworkAllowlist.from_env()
    try:
        config = load_backend_config(models, resources, allowlist=allowlist)
        configs = mcp_servers_from_resources(config.resources, allowlist=allowlist)
    except (ResourcesError, ModelsError) as exc:
        print(f"validate: {exc}", file=sys.stderr)
        return 1
    # MCP servers are declared, never contacted: their tool lists stay empty here.
    mcp = MCPRegistry(configs) if config.resources else None

    alias_checks, built_models, placeholders = asyncio.run(
        check_aliases(agent, config, probe=probe)
    )
    checks = validate_agent(agent, mcp=mcp, models=built_models, resources=placeholders)
    checks += alias_checks
    for check in checks:
        print(f"[{'ok' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    failed = [check for check in checks if not check.ok]
    label = "PASS" if not failed else f"FAIL ({len(failed)} check{'s' if len(failed) > 1 else ''})"
    print(f"validate {agent.__class__.__name__}: {label}")
    return 0 if not failed else 1
