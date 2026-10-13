"""Governed-tool primitives (PRD §2.7, A33, A37–A40).

vdreamer adds no tool base class (A37): every tool — custom, ecosystem, or
MCP — stays a plain LangChain ``BaseTool``. Governance is *data*, a
:class:`Policy` stored under ``tool.metadata["vdreamer"]``, plus *wrappers*
(:class:`VDreamerToolMiddleware` hooks ``create_agent`` tool calls; M5's
``ctx.tools.govern`` proxies plain tools) that keep the tool's name and
schema because they wrap and never rewrite.

Milestone 1 ships the contract: the policy dataclasses, argument redaction
(A33), policy resolution, the ``@tool`` decorator, and a pass-through
middleware that proves the wiring. The seven-step governance pipeline of
§2.7 — authorize → emit ``tool.call`` → approval gate → execute under
timeout → normalize → contain errors → emit ``tool.result`` — lands in
Milestone 5 (the approval gate itself in Milestone 8) and consumes exactly
the Policy/metadata types defined here.
"""

from __future__ import annotations

import dataclasses
import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool as _langchain_tool
from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence

    from langchain_core.messages import ToolMessage
    from langchain_core.runnables import Runnable
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command


VDREAMER_METADATA_KEY: Final = "vdreamer"
"""Metadata dict key under which a tool's :class:`Policy` is stored (PRD §2.7)."""


@dataclass(frozen=True)
class ApprovalSpec:
    """Approval requirements for one tool (PRD §4.1, attached via ``Policy.approval``).

    Mirrors the fields a per-tool approval needs from the HITL SDK: ``kind``
    is the renderer hint, ``approver_role`` gates who may resolve (required in
    headless runs, A16), ``scope`` names what is being approved, and
    ``ttl_hours`` bounds how long the interrupt stays resolvable. Milestone 8
    turns this into a real gate via ``require_approval``; until then it is
    declarative data that travels with the tool.
    """

    kind: str
    approver_role: str | None = None
    scope: str | None = None
    ttl_hours: int = 24


@dataclass(frozen=True)
class Policy:
    """Governance policy for one tool (PRD §2.7).

    Stored JSON-safe in ``tool.metadata["vdreamer"]`` by :func:`tool` and
    rebuilt with :meth:`from_metadata`. Every field feeds one governance step
    (see :class:`VDreamerToolMiddleware` for the pipeline they will drive):

    - ``redact``: argument keys masked in ``tool.call`` events (plus
      ``DEFAULT_REDACT_KEYS`` always, A33).
    - ``timeout_s``: execution timeout (default 60).
    - ``side_effects``: run the call under ``ctx.idempotency_key`` for safe
      retries (A6).
    - ``approval``: optional per-tool approval gate (§4.1).
    - ``max_result_chars``: results over this become an artifact and the
      model receives a truncated body (default 8,000).
    - ``fail_run``: contain a tool failure as an error ``ToolMessage``
      instead of failing the run (default).
    - ``scope`` / ``requires_roles``: authorization inputs.
    """

    redact: tuple[str, ...] = ()
    timeout_s: float = 60.0
    side_effects: bool = False
    approval: ApprovalSpec | None = None
    max_result_chars: int = 8000
    fail_run: bool = False
    scope: str | None = None
    requires_roles: tuple[str, ...] = ()

    @classmethod
    def from_metadata(cls, meta: Mapping[str, Any]) -> Policy:
        """Rebuild a :class:`Policy` from its metadata dict.

        Inverse of :meth:`to_metadata`: unknown keys are ignored and missing
        keys fall back to the class defaults, so hand-written partial metadata
        (``{"redact": ["query"]}``) also round-trips.
        """
        approval_meta = meta.get("approval")
        approval = (
            ApprovalSpec(
                kind=approval_meta["kind"],
                approver_role=approval_meta.get("approver_role"),
                scope=approval_meta.get("scope"),
                ttl_hours=approval_meta.get("ttl_hours", 24),
            )
            if approval_meta
            else None
        )
        return cls(
            redact=tuple(meta.get("redact", ())),
            timeout_s=meta.get("timeout_s", 60.0),
            side_effects=meta.get("side_effects", False),
            approval=approval,
            max_result_chars=meta.get("max_result_chars", 8000),
            fail_run=meta.get("fail_run", False),
            scope=meta.get("scope"),
            requires_roles=tuple(meta.get("requires_roles", ())),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Serialize to the JSON-safe dict stored under ``metadata["vdreamer"]``.

        Tuples become lists and ``approval`` becomes a dict (or ``None``) so
        the value survives ``json.dumps`` and Pydantic ``model_dump`` paths.
        """
        approval = (
            None
            if self.approval is None
            else {
                "kind": self.approval.kind,
                "approver_role": self.approval.approver_role,
                "scope": self.approval.scope,
                "ttl_hours": self.approval.ttl_hours,
            }
        )
        return {
            "redact": list(self.redact),
            "timeout_s": self.timeout_s,
            "side_effects": self.side_effects,
            "approval": approval,
            "max_result_chars": self.max_result_chars,
            "fail_run": self.fail_run,
            "scope": self.scope,
            "requires_roles": list(self.requires_roles),
        }


@dataclass(frozen=True)
class InterruptPolicy:
    """HITL policy overrides for non-vdreamer interrupt sources (PRD §4.5, A44).

    ``BaseAgent.interrupt_policies`` maps a source — keyed by ``kind``
    (``"tool_review"``, ``"mcp_elicitation"``, ``"generic"``) or by tool
    name — to one of these, attaching ``approver_role``/``scope``/TTL to
    interrupts raised outside ``require_approval``. All fields optional:
    ``None`` means "inherit the source's own value".
    """

    approver_role: str | None = None
    scope: str | None = None
    ttl_hours: int | None = None


DEFAULT_REDACT_KEYS: Final = ("password", "token", "secret", "api_key", "authorization")
"""Argument keys always redacted from ``tool.call`` events (A33: redaction =
policy keys plus these defaults)."""

_REDACTED: Final = "***"
_POLICY_DEFAULTS: Final = Policy()


def _redact_value(value: Any, masked: frozenset[str]) -> Any:
    """Recursively mask sensitive keys in nested dicts/lists (A33)."""
    if isinstance(value, dict):
        return {
            key: _REDACTED
            if isinstance(key, str) and key.lower() in masked
            else _redact_value(item, masked)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, masked) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item, masked) for item in value)
    return value


def redact_args(args: Mapping[str, Any], keys: Iterable[str] | None = None) -> dict[str, Any]:
    """Copy ``args`` with every sensitive key masked, at any nesting depth.

    A dict entry (including dicts inside dicts and lists) whose key
    case-insensitively equals a key in ``keys`` **or** in
    :data:`DEFAULT_REDACT_KEYS` has its value replaced with ``"***"``. The
    input mapping is never mutated; scalars and unknown objects pass through
    unchanged (repr-safety of exotic objects is M5's concern).
    """
    masked = {key.lower() for key in DEFAULT_REDACT_KEYS}
    if keys is not None:
        masked |= {key.lower() for key in keys}
    return cast("dict[str, Any]", _redact_value(dict(args), frozenset(masked)))


def resolve_policy(
    name: str,
    decorated: Policy | None,
    policies: Mapping[str, Policy],
) -> Policy:
    """Resolve the effective :class:`Policy` for the tool called ``name``.

    Precedence: **agent tool_policies override decorator metadata per field.**
    The resolution starts from ``decorated`` (or the class defaults when the
    tool carries no policy — a plain ecosystem tool) and, for every key in
    ``policies`` whose glob matches ``name`` (``fnmatch`` patterns such as
    ``"sql_db_*"``, compared case-sensitively via ``fnmatchcase``), overlays
    that policy's *non-default* fields — a field left at its class default
    means "not specified here" and never clobbers the decorator's value, and
    ``approval`` overlays only when not ``None``. When several patterns match,
    they apply in mapping order and the last match wins per field.
    """
    resolved = decorated if decorated is not None else Policy()
    for pattern, agent_policy in policies.items():
        if not fnmatch.fnmatchcase(name, pattern):
            continue
        overrides: dict[str, Any] = {}
        for field in dataclasses.fields(Policy):
            value = getattr(agent_policy, field.name)
            if field.name == "approval":
                if value is not None:
                    overrides[field.name] = value
            elif value != getattr(_POLICY_DEFAULTS, field.name):
                overrides[field.name] = value
        if overrides:
            resolved = dataclasses.replace(resolved, **overrides)
    return resolved


def _govern_tool(candidate: Any, policy: Policy) -> BaseTool:
    """Attach ``policy`` to a built tool, preserving unrelated metadata.

    The candidate must already be a LangChain ``BaseTool`` — vdreamer never
    wraps tools in another class (A37); governance travels as metadata.
    """
    if not isinstance(candidate, BaseTool):
        msg = (
            "vdreamer.tool expected langchain to produce a BaseTool, "
            f"got {type(candidate).__name__}"
        )
        raise TypeError(msg)
    metadata = dict(candidate.metadata) if candidate.metadata else {}
    metadata[VDREAMER_METADATA_KEY] = policy.to_metadata()
    candidate.metadata = metadata
    return candidate


def _build_langchain_tool(
    target: Callable[..., Any],
    name: str | None,
    tool_kwargs: dict[str, Any],
) -> BaseTool:
    """Delegate construction to ``langchain.tools.tool`` for a plain function.

    Upstream takes the tool name only positionally; vdreamer also accepts it
    as the ``name=`` keyword so the configured decorator reads uniformly.
    """
    if name is None:
        produced = _langchain_tool(target, **tool_kwargs)
    else:
        factory = _langchain_tool(name, **tool_kwargs)
        if isinstance(factory, BaseTool):
            # `runnable=` builds the tool immediately, no factory step.
            return factory
        produced = factory(target)
    if not isinstance(produced, BaseTool):
        msg = (
            "langchain.tools.tool produced "
            f"{type(produced).__name__}, expected a BaseTool"
        )
        raise TypeError(msg)
    return produced


def tool(
    *args: Any,
    redact: Iterable[str] = (),
    timeout_s: float = 60.0,
    side_effects: bool = False,
    approval: ApprovalSpec | None = None,
    max_result_chars: int = 8000,
    fail_run: bool = False,
    scope: str | None = None,
    requires_roles: Iterable[str] = (),
    policy: Policy | None = None,
    **tool_kwargs: Any,
) -> BaseTool | Callable[[Callable[..., Any] | Runnable[Any, Any]], BaseTool]:
    """``langchain.tools.tool`` plus a vdreamer :class:`Policy` (PRD §2.7).

    A drop-in for ``@tool``: ``@vdreamer.tool`` (no parens) and the
    configured ``@vdreamer.tool(...)``, with ``name``/``description``/
    ``args_schema``/… forwarded to LangChain untouched. The only addition is
    that the produced tool carries ``metadata["vdreamer"] =
    policy.to_metadata()`` (merged, so unrelated metadata keys survive).

    Governance kwargs (defaults per §2.7): ``redact``, ``timeout_s`` (60),
    ``side_effects``, ``approval``, ``max_result_chars`` (8,000),
    ``fail_run``, ``scope``, ``requires_roles``.

    Escape hatch: pass ``policy=Policy(...)`` to use a ready-made policy
    object; it wins over the individual kwargs (they are ignored entirely).

    Examples:
        >>> @tool
        ... def lookup(key: str) -> str:
        ...     '''Look a key up.'''
        ...     return "value"

        >>> @tool(redact=["query"], timeout_s=30, side_effects=True)
        ... def run_query(query: str) -> str:
        ...     '''Run a read-only query.'''
    """
    if policy is not None:
        effective_policy = policy
    else:
        effective_policy = Policy(
            redact=tuple(redact),
            timeout_s=timeout_s,
            side_effects=side_effects,
            approval=approval,
            max_result_chars=max_result_chars,
            fail_run=fail_run,
            scope=scope,
            requires_roles=tuple(requires_roles),
        )

    # Normalize the call shape. Upstream takes the tool name positionally;
    # vdreamer additionally accepts `name=` (and mirrors upstream's positional
    # name form `@tool("renamed")`).
    name = tool_kwargs.pop("name", None)
    direct_target: Callable[..., Any] | None = None
    if args:
        first = args[0]
        if len(args) == 1 and callable(first) and hasattr(first, "__name__"):
            direct_target = first  # `@tool` on a function, or tool(func, ...)
        elif len(args) == 1 and isinstance(first, str):
            if name is not None:
                msg = "tool() got both a positional name and name=; pass only one."
                raise ValueError(msg)
            name = first
        else:
            msg = (
                "tool() takes at most one positional argument: the function to "
                "decorate (bare @tool) or its name. Use runnable= for Runnables."
            )
            raise ValueError(msg)

    if direct_target is not None:
        return _govern_tool(_build_langchain_tool(direct_target, name, tool_kwargs),
                            effective_policy)

    if "runnable" in tool_kwargs:
        # tool(name, runnable=...) constructs the BaseTool immediately.
        positionals = () if name is None else (name,)
        produced = _langchain_tool(*positionals, **tool_kwargs)
        return _govern_tool(produced, effective_policy)

    # Configured decorator: upstream's own partial handles Runnable targets
    # and name inference exactly like @tool(...) does.
    candidate = (
        _langchain_tool(name, **tool_kwargs) if name is not None else _langchain_tool(**tool_kwargs)
    )
    if isinstance(candidate, BaseTool):
        # Only reachable for runnable=..., handled above; kept defensive.
        return _govern_tool(candidate, effective_policy)

    def _decorator(target: Callable[..., Any] | Runnable[Any, Any]) -> BaseTool:
        return _govern_tool(candidate(target), effective_policy)

    return _decorator


class VDreamerToolMiddleware(AgentMiddleware[Any, Any, Any]):
    """Tool-governance middleware for ``create_agent`` graphs (PRD §2.7, A37).

    **Milestone 1: pure pass-through.** Both hooks simply run the handler —
    nothing is authorized, redacted, timed out, normalized, or contained yet.
    The full governance pipeline lands in Milestone 5 (approval gate:
    Milestone 8) and will run inside these hooks on every governed call,
    consuming the :class:`Policy` stored in ``tool.metadata["vdreamer"]``
    (§2.7 steps):

    1. **Authorize** — tool ``scope`` plus ``requires_roles`` against the
       initiator's roles.
    2. **Emit ``tool.call``** — args redacted per policy keys plus
       :data:`DEFAULT_REDACT_KEYS` (A33).
    3. **Approve** — optional ``Policy.approval`` gate via
       ``require_approval`` (M8).
    4. **Execute** — under ``Policy.timeout_s``, with ``ctx.idempotency_key``
       for ``side_effects=True``.
    5. **Normalize** — results over ``max_result_chars`` become artifacts.
    6. **Contain** — failures become error ``ToolMessage`` unless
       ``fail_run=True``.
    7. **Emit ``tool.result``** — with source and ``duration_ms``.

    A37: the tool's name and schema are preserved because this middleware
    *wraps* calls — it never rewrites the ``BaseTool``, its name, its args
    schema, or its result.

    ``awrap_tool_call`` is the hook vdreamer's async runtime exercises
    (``ainvoke``/``astream``); ``wrap_tool_call`` keeps plain synchronous
    ``invoke`` working. No other hooks are overridden.
    """

    tools: Sequence[BaseTool] = ()
    """No additional tools; this middleware only wraps existing ones."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Synchronous pass-through; the §2.7 pipeline arrives in M5."""
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Asynchronous pass-through; the §2.7 pipeline arrives in M5."""
        return await handler(request)
