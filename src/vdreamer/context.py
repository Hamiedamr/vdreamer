"""The run context and the implicit runtime helpers (PRD §2.2, §3.3, §3.4, §4.1, §4.5).

Milestone 1 scope:

- ``VDreamerContext`` — the run identity. It rides the LangGraph ``context=``
  channel (``create_agent`` graphs declare ``context_schema=VDreamerContext``)
  *and* a module-level ``ContextVar``, so the implicit helpers below resolve
  the current run in any node or tool without parameters (PRD §2.2 line 70).
- ``emit_progress`` / ``emit_log`` / ``emit_citation`` — buffer one protocol
  record (PRD §3.3 payload shapes) into ``VDreamerContext.pending`` and
  mirror it into the run's stream. ``emit_artifact`` lives in
  ``vdreamer.artifacts`` (it stores bytes first, D7) and is re-exported here.
- ``require_approval`` — the HITL gate (PRD §4.1): one normalized interrupt
  value shared by every approval source (A44).

Since Milestone 6 the context also carries ``ctx.files`` (inbound files,
``safe_extract``) and ``ctx.sandbox`` (D19); ``FileRef`` validates its
``inputs/`` prefix (A17).
"""

from __future__ import annotations

import contextlib
import contextvars
import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pydantic
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from vdreamer.artifacts import (
    Artifact,
    ArtifactKind,
    ArtifactRef,
    aemit_artifact,
    emit_artifact,
)

if TYPE_CHECKING:
    from vdreamer.files import Files
    from vdreamer.sandbox import Sandbox

__all__ = [
    "Artifact",
    "ArtifactKind",
    "ArtifactRef",
    "FileRef",
    "VDreamerContext",
    "aemit_artifact",
    "bind_context",
    "current_context",
    "current_context_or_none",
    "emit_artifact",
    "emit_citation",
    "emit_log",
    "emit_progress",
    "emit_record",
    "require_approval",
]

#: PRD §3.3: citation snippets are capped at 300 characters.
MAX_SNIPPET_CHARS = 300

#: PRD §4.1: the default action set for ``require_approval``.
DEFAULT_APPROVAL_ACTIONS: tuple[str, ...] = ("approve", "reject", "edit", "feedback")

#: PRD §4.3: every action a resolver may take.
APPROVAL_ACTIONS: frozenset[str] = frozenset({"approve", "reject", "edit", "feedback", "respond"})


class FileRef(pydantic.BaseModel):
    """An inbound file: a MinIO key under ``inputs/`` (PRD §2.2, §3.5, A17).

    Upstream systems upload to ``inputs/`` and pass the key in a run's input;
    validation refuses any other prefix and any key that could climb out of
    it (``..``, a leading ``/``, backslashes) — so the gateway, which
    validates every input against the agent's ``input_schema``, rejects a
    ref that points at another thread's artifacts with ``422`` (A17). Agents
    read the file with ``ctx.files.open(ref)``.
    """

    object_key: str

    @pydantic.field_validator("object_key")
    @classmethod
    def _inputs_prefix(cls, value: str) -> str:
        from vdreamer.objects import INPUTS_PREFIX, ObjectKeyError, check_key

        try:
            check_key(value)
        except ObjectKeyError:
            raise ValueError("object_key must be a clean relative key (A17)") from None
        if not value.startswith(INPUTS_PREFIX) or value == INPUTS_PREFIX:
            raise ValueError(f"object_key must be under {INPUTS_PREFIX!r} (A17)")
        return value


class VDreamerContext(pydantic.BaseModel):
    """The run identity (PRD §2.2).

    Carried via LangGraph's ``context=`` — ``create_agent``-built graphs
    declare ``context_schema=VDreamerContext`` and receive it as
    ``runtime.context`` in nodes — *and* held in a module-level ``ContextVar``
    so the implicit helpers (``emit_*``) resolve the current run from any node
    or tool without parameters (PRD §2.2 line 70).

    ``pending`` buffers the protocol records the helpers append; the worker
    event pipeline that drains it lands in Milestone 2.
    """

    run_id: str
    thread_id: str
    agent_name: str
    user_id: str | None = None
    roles: tuple[str, ...] = ()
    idempotency_key: str | None = None
    attempt: int = 0
    headless: bool = False
    """A headless run (D15): no user in the loop, so every approval must name
    an ``approver_role`` (A16) and a ``reject`` ends the run ``rejected``."""
    pending: list[dict[str, Any]] = pydantic.Field(default_factory=list)

    @property
    def files(self) -> Files:
        """Inbound files and ``safe_extract`` (§3.5): ``ctx.files.open(ref)``."""
        from vdreamer.files import Files

        return Files(self)

    @property
    def sandbox(self) -> Sandbox:
        """The code sandbox (§3.5, D19): ``ctx.sandbox.run(code, inputs)``."""
        from vdreamer.sandbox import Sandbox

        return Sandbox(self)


#: The active run, or ``None`` outside a vdreamer-driven invocation.
_RUN: contextvars.ContextVar[VDreamerContext | None] = contextvars.ContextVar(
    "vdreamer_run", default=None
)


@contextlib.contextmanager
def bind_context(ctx: VDreamerContext) -> Iterator[VDreamerContext]:
    """Bind ``ctx`` as the current vdreamer run for the duration of the block.

    ``AgentRunner.arun`` (vdreamer.runtime) wraps a whole invocation —
    lifecycle hooks included — in this, so the implicit helpers resolve the
    run from anywhere the graph reaches. The previous binding is restored in
    ``finally``, so nesting and exceptions behave.
    """
    token = _RUN.set(ctx)
    try:
        yield ctx
    finally:
        _RUN.reset(token)


def current_context() -> VDreamerContext:
    """The active ``VDreamerContext``.

    Raises:
        RuntimeError: if no vdreamer run is bound — the helpers only work
            inside a vdreamer run.
    """
    ctx = _RUN.get()
    if ctx is None:
        raise RuntimeError(
            "no active vdreamer run context; helpers only work inside a vdreamer run"
        )
    return ctx


def current_context_or_none() -> VDreamerContext | None:
    """The active ``VDreamerContext``, or ``None`` outside a vdreamer run."""
    return _RUN.get()


def _mirror_to_stream(record: dict[str, Any]) -> None:
    """Best-effort mirror of a buffered record into the LangGraph stream (WP2).

    Under the vdreamer stream drivers (``vdreamer.streaming``, A31), custom-mode
    stream data — not ``pending`` — is the live transport: every driver maps
    vdreamer-enveloped ``custom`` parts back onto the protocol events. So the
    helpers append to ``pending`` (the fallback, kept for non-streamed runs
    and drained by the worker) *and* write the same record through
    ``langgraph.config.get_stream_writer()`` when one is active.

    The mirror is strictly best-effort: no stream context (plain ``ainvoke``,
    tests, or a driver run that does not collect ``custom`` mode), no callable,
    or a failing writer must never break the run. Outside a LangGraph stream,
    ``get_stream_writer()`` raises and the record simply stays buffered.
    """
    try:
        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    try:
        writer(record)
    except Exception:
        pass


def emit_record(record: dict[str, Any]) -> None:
    """Buffer one protocol record for the current run and mirror it live.

    The shared tail of every ``emit_*`` helper — also how the governed tool
    pipeline (``vdreamer.governance``) and the artifact writer publish their
    ``tool.call``/``tool.result``/``artifact`` records.
    """
    current_context().pending.append(record)
    _mirror_to_stream(record)


def emit_progress(label: str, pct: float | None = None) -> None:
    """Emit a ``progress`` record (PRD §3.3): ``{label, pct?}``."""
    record = {"type": "progress", "label": label, "pct": pct}
    emit_record(record)


def emit_log(message: str, level: str = "info") -> None:
    """Emit a ``log`` record (PRD §3.3): ``{level, message}``."""
    record = {"type": "log", "level": level, "message": message}
    emit_record(record)


def emit_citation(
    message_id: str,
    source: Mapping[str, Any],
    locator: Mapping[str, Any] | None = None,
    snippet: str = "",
    score: float | None = None,
) -> str:
    """Emit a ``citation`` record (PRD §3.3) and return the ``citation_id``.

    ``source`` and ``locator`` are copied to plain dicts (the record must
    survive the run); ``snippet`` is truncated to ``MAX_SNIPPET_CHARS`` (300).
    """
    citation_id = f"cit_{uuid.uuid4().hex}"
    record = {
        "type": "citation",
        "citation_id": citation_id,
        "message_id": message_id,
        "source": dict(source),
        "locator": dict(locator) if locator is not None else None,
        "snippet": snippet[:MAX_SNIPPET_CHARS],
        "score": score,
    }
    emit_record(record)
    return citation_id


def require_approval(
    kind: str,
    payload: Any,
    actions: Sequence[str] = DEFAULT_APPROVAL_ACTIONS,
    *,
    approver_role: str | None = None,
    scope: str | None = None,
    edit_schema: type[pydantic.BaseModel] | None = None,
    ttl_hours: int = 24,
) -> Any:
    """Pause the run for a human decision (PRD §4.1, §4.5, A44).

    Wraps ``langgraph.types.interrupt`` with the one normalized approval value
    every interrupt source shares (A44):

    ``{kind, payload, actions, approver_role, scope, edit_schema, ttl_hours}``

    where ``edit_schema`` is serialized to its JSON Schema. The gateway
    enforces the serialized policy (A15): who may resolve (the initiator, or
    ``approver_role``; four-eyes for ``HITL_STRICT_SCOPES``, A32), the
    offered ``actions``, the expiry, and ``edit`` payloads against the schema
    (``422`` otherwise). In a headless run a ``reject`` ends the run
    ``rejected`` without resuming it (§4.3).

    Must be called from inside a LangGraph node — ``interrupt()`` needs the
    run config, and outside a runnable context it raises ``RuntimeError``. The
    first call surfaces as the run's pending interrupt
    (``aget_state(config).interrupts``); a resume restarts this node from the
    top, so pre-work belongs in a preceding node (PRD §4.1), and the resume
    value arrives as this function's return.

    Args:
        kind: renderer hint, e.g. ``"sql_review"``.
        payload: what the approver sees.
        actions: offered decisions, default ``("approve", "reject", "edit",
            "feedback")``; any of ``approve|reject|edit|feedback|respond``.
        approver_role: required role for resolution; required in headless runs
            (A16) — a headless call without one raises ``ValueError``.
        scope: capability scope, e.g. ``"db:write:prod"``.
        edit_schema: pydantic model validating ``edit`` payloads.
        ttl_hours: how long the interrupt stays resolvable before expiry.

    Returns:
        The resolution ``{"action": ..., "payload"?: ..., "text"?: ...}``. An
        ``edit`` payload has been validated against ``edit_schema`` (again,
        here) and is its JSON dump.

    Raises:
        ValueError: an unknown or empty action set, a non-positive TTL, or a
            headless run without ``approver_role``.
        pydantic.ValidationError: an ``edit`` payload that fails
            ``edit_schema`` (the gateway refuses those first).
    """
    offered = list(actions)
    unknown = sorted(set(offered) - APPROVAL_ACTIONS)
    if not offered or unknown:
        raise ValueError(
            f"require_approval actions must be a non-empty subset of {sorted(APPROVAL_ACTIONS)}"
            + (f"; unknown: {unknown}" if unknown else "")
        )
    if ttl_hours <= 0:
        raise ValueError("require_approval ttl_hours must be positive")
    ctx = current_context_or_none()
    if ctx is not None and ctx.headless and not approver_role:
        raise ValueError(
            f"require_approval({kind!r}) in a headless run needs an approver_role (A16)"
        )
    decision = interrupt(
        {
            "kind": kind,
            "payload": payload,
            "actions": offered,
            "approver_role": approver_role,
            "scope": scope,
            "edit_schema": edit_schema.model_json_schema() if edit_schema is not None else None,
            "ttl_hours": ttl_hours,
        }
    )
    if (
        edit_schema is not None
        and isinstance(decision, Mapping)
        and decision.get("action") == "edit"
    ):
        edited = edit_schema.model_validate(decision.get("payload"))
        decision = {**decision, "payload": edited.model_dump(mode="json")}
    return decision
