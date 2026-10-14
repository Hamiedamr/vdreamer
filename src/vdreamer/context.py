"""The run context and the implicit runtime helpers (PRD §2.2, §3.3, §3.4, §4.1, §4.5).

Milestone 1 scope:

- ``VDreamerContext`` — the run identity. It rides the LangGraph ``context=``
  channel (``create_agent`` graphs declare ``context_schema=VDreamerContext``)
  *and* a module-level ``ContextVar``, so the implicit helpers below resolve
  the current run in any node or tool without parameters (PRD §2.2 line 70).
- ``emit_progress`` / ``emit_log`` / ``emit_artifact`` / ``emit_citation`` —
  buffer one protocol record (PRD §3.3 payload shapes) into
  ``VDreamerContext.pending``; the worker event pipeline that drains,
  envelopes, and persists ``pending`` lands in Milestone 2.
- ``require_approval`` — the HITL gate (PRD §4.1): one normalized interrupt
  value shared by every approval source (A44).

Deferred: artifact byte storage, MinIO writes, metadata rows, and the
``artifact.created`` write order (D7) — Milestone 6. ``FileRef`` prefix
validation (``inputs/``, A17) is the gateway's, Milestone 5.
"""

from __future__ import annotations

import contextlib
import contextvars
import enum
import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pydantic
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

#: PRD §3.3: citation snippets are capped at 300 characters.
MAX_SNIPPET_CHARS = 300

#: PRD §4.1: the default action set for ``require_approval``.
DEFAULT_APPROVAL_ACTIONS: tuple[str, ...] = ("approve", "reject", "edit", "feedback")


class ArtifactKind(enum.StrEnum):
    """The five artifact kinds (PRD §3.4)."""

    TABLE = "table"
    IMAGE = "image"
    HTML = "html"
    JSON = "json"
    FILE = "file"


class FileRef(pydantic.BaseModel):
    """A MinIO key under ``inputs/`` (PRD §2.2, A17).

    The gateway validates the ``inputs/`` prefix later; the SDK model is a
    plain carrier for now.
    """

    object_key: str


class ArtifactRef(pydantic.BaseModel):
    """What ``emit_artifact`` returns: identity and display hints, never bytes."""

    artifact_id: str
    name: str
    kind: ArtifactKind
    mime_type: str | None = None
    size_bytes: int | None = None


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
    pending: list[dict[str, Any]] = pydantic.Field(default_factory=list)


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


def emit_artifact(
    name: str,
    kind: ArtifactKind,
    data: Any,
    metadata: dict[str, Any] | None = None,
    renderer: str | None = None,
) -> ArtifactRef:
    """Buffer an artifact and return its ref (PRD §3.4).

    Generates ``artifact_id`` and appends the full record — ``data`` included —
    to the current run's ``pending`` buffer; the returned ``ArtifactRef`` is
    what agent code hands to the model or writes into state.

    ``data`` is *not* validated per-kind yet; that strictness (Parquet tables,
    mime sniffing, size caps — PRD §3.4) lands with Milestone 6. Two things are
    enforced now: ``kind`` must be an ``ArtifactKind`` member and ``data`` must
    not be ``None`` — both ``TypeError``, matching the §3.4 rule that
    unsupported data is a caller error, never auto-detected.

    Byte storage, MinIO writes, the metadata row, and the ``artifact.created``
    event with its write order (D7) land in Milestone 6; Milestone 1 is the SDK
    surface. ``renderer`` is an optional UI hint (e.g. ``verdict.v1``).
    """
    if not isinstance(kind, ArtifactKind):
        raise TypeError(
            f"kind must be an ArtifactKind member (table/image/html/json/file), got {kind!r}"
        )
    if data is None:
        raise TypeError("artifact data must not be None")
    artifact_id = f"art_{uuid.uuid4().hex}"
    record = {
        "type": "artifact",
        "artifact_id": artifact_id,
        "name": name,
        "kind": kind,
        "data": data,
        "metadata": metadata,
        "renderer": renderer,
    }
    emit_record(record)
    return ArtifactRef(artifact_id=artifact_id, name=name, kind=kind)


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
    enforces the serialized policy (A15) and the resolver body (§4.3 —
    ``{action, payload?, text?}``, ``edit`` validated against ``edit_schema``);
    that lands with Milestone 5. Milestone 1 returns whatever the resume
    delivers, verbatim.

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
            "feedback")``.
        approver_role: required role for resolution; required in headless runs
            (A16) — enforced by the gateway, Milestone 5.
        scope: capability scope, e.g. ``"db:write:prod"``.
        edit_schema: pydantic model validating ``edit`` payloads.
        ttl_hours: how long the interrupt stays resolvable before expiry.

    Returns:
        Whatever the resume delivers — the gateway-built resolution.
    """
    return interrupt(
        {
            "kind": kind,
            "payload": payload,
            "actions": list(actions),
            "approver_role": approver_role,
            "scope": scope,
            "edit_schema": edit_schema.model_json_schema() if edit_schema is not None else None,
            "ttl_hours": ttl_hours,
        }
    )
