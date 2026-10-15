"""Protocol v1.0 events: constants, event types, the envelope, payload models.

PRD §3.3 (Event Log & Protocol) is the contract: every wire event is one
``EventEnvelope`` whose ``type`` selects one of the twelve protocol event types
and whose ``payload`` matches the per-type shapes in the §3.3 table. The
envelope field for the protocol version is literally ``"v"``
(``{"v":"1.0", ...}``); clients send ``X-Protocol-Version: 1`` (§6.1).
``PROTOCOL_VERSION`` and ``PROTOCOL_VERSION_HEADER`` are the single source of
both.

Milestone 2 scope (WP1): this module is the protocol *core* — constants, the
event types, the envelope, the payload models, and the delivery-tuning
constants the WP2 stream drivers consume (delta batching 50 ms / 32 tokens,
15 s heartbeat, ``X-Accel-Buffering: no``, A22). The stream drivers (A31) and
the SSE transport land in later work packages; the event sinks and the
Postgres ``run_events`` log live in ``vdreamer.eventlog``.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any, Final, Literal

import pydantic

#: The protocol version carried in every envelope as ``"v"`` and asserted by
#: ``X-Protocol-Version`` (PRD §3.3, §6.1).
PROTOCOL_VERSION: Final[str] = "1.0"

#: The header clients send with the protocol version (PRD §6.1).
PROTOCOL_VERSION_HEADER: Final[str] = "X-Protocol-Version"

#: Deltas batch at emission: flush at most every 50 ms or every 32 tokens
#: (PRD §3.3 line 211). Consumed by the WP2 stream drivers.
DELTA_BATCH_INTERVAL_S: Final[float] = 0.05

#: The token count that forces a delta batch flush (PRD §3.3 line 211).
DELTA_BATCH_MAX_TOKENS: Final[int] = 32

#: SSE heartbeat cadence (PRD §3.3 line 211); idle proxy timeouts must be
#: more than 2x this (§6.2).
HEARTBEAT_INTERVAL_S: Final[float] = 15.0

#: Disable proxy response buffering for SSE (A22).
SSE_BUFFERING_HEADER: Final[str] = "X-Accel-Buffering"

#: The value that disables buffering (A22).
SSE_BUFFERING_VALUE: Final[str] = "no"


class EventType(enum.StrEnum):
    """The twelve protocol event types (PRD §3.3 table, docs/tasks.md line 92)."""

    RUN_STATUS = "run.status"
    TOKEN_DELTA = "token.delta"
    THINKING_DELTA = "thinking.delta"
    MESSAGE_COMPLETED = "message.completed"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    ARTIFACT_CREATED = "artifact.created"
    CITATION = "citation"
    INTERRUPT_REQUESTED = "interrupt.requested"
    INTERRUPT_RESOLVED = "interrupt.resolved"
    PROGRESS = "progress"
    LOG = "log"


class _Payload(pydantic.BaseModel):
    """Base for payload models: ``event_payload()`` drops ``None`` optionals.

    The PRD payload tables mark optional fields with ``?``; absent optionals
    are omitted from the wire payload, not serialized as ``null``. Emitters
    (the WP2 stream drivers) call ``event_payload()`` — the name avoids
    shadowing: ``InterruptRequestedPayload`` has a *field* named ``payload``
    (PRD §4.2). Tests spot-check the dumps against the §3.3 shapes.
    """

    model_config = pydantic.ConfigDict(populate_by_name=True)

    def event_payload(self) -> dict[str, Any]:
        """Return the wire payload: this model dumped without ``None`` optionals."""
        return self.model_dump(exclude_none=True)


class ErrorInfo(_Payload):
    """The ``{code, message}`` error block in ``run.status`` and ``tool.result``."""

    code: str
    message: str


class RunStatusPayload(_Payload):
    """``run.status``: ``{status, attempt, error?: {code, message}}`` (§3.3)."""

    status: str
    attempt: int = 0
    error: ErrorInfo | None = None


class TokenDeltaPayload(_Payload):
    """``token.delta`` / ``thinking.delta``: ``{message_id, text}`` (§3.3).

    The same shape serves both delta types; ``thinking.delta`` is suppressed
    entirely when ``expose_thinking=false`` — that gate is the driver's, WP2.
    """

    message_id: str
    text: str


class MessageCompletedPayload(_Payload):
    """``message.completed``: ``{message_id, role, text, thinking?, citation_ids[],
    usage?}`` (§3.3)."""

    message_id: str
    role: str
    text: str
    thinking: str | None = None
    citation_ids: list[str] = pydantic.Field(default_factory=list)
    usage: dict[str, Any] | None = None


class ToolCallPayload(_Payload):
    """``tool.call``: ``{tool_call_id, name, source, args}`` (§3.3, A33).

    ``source`` is ``custom|langchain|mcp:{server}``. ``args`` must already be
    redacted when it reaches here — the governance pipeline (§2.7) redacts
    policy keys plus the defaults (password, token, secret, api_key,
    authorization) before the event is emitted; this model does not redact.
    """

    tool_call_id: str
    name: str
    source: str
    args: dict[str, Any]


class ToolResultPayload(_Payload):
    """``tool.result``: ``{tool_call_id, status, error?, summary, artifact_ids[],
    duration_ms}`` (§3.3).

    Full tool output is *never* streamed — oversized results become artifacts
    (§2.7) and only ``summary`` plus ``artifact_ids`` ride the event.
    """

    tool_call_id: str
    status: Literal["ok", "error"]
    error: ErrorInfo | None = None
    summary: str
    artifact_ids: list[str] = pydantic.Field(default_factory=list)
    duration_ms: int


class CitationSource(_Payload):
    """The ``{artifact_id?, uri?, title}`` source block of a ``citation`` (§3.3)."""

    artifact_id: str | None = None
    uri: str | None = None
    title: str


class CitationPayload(_Payload):
    """``citation``: ``{citation_id, message_id, source, locator?, snippet, score?}``
    (§3.3).

    ``snippet`` is capped at 300 characters (``MAX_SNIPPET_CHARS``,
    ``vdreamer.context``) by the emitters — ``emit_citation`` truncates — so
    this carrier does not re-truncate.
    """

    citation_id: str
    message_id: str
    source: CitationSource
    locator: dict[str, Any] | None = None
    snippet: str
    score: float | None = None


class ProgressPayload(_Payload):
    """``progress``: ``{label, pct?}`` (§3.3)."""

    label: str
    pct: float | None = None


class LogPayload(_Payload):
    """``log``: ``{level, message}`` (§3.3)."""

    level: str
    message: str


class InterruptRequestedPayload(_Payload):
    """``interrupt.requested``: ``{interrupt_id, kind, scope?, approver_role?,
    actions[], payload, edit_schema?, expires_at?}`` (PRD §4.2).

    ``edit_schema`` is the JSON Schema of the ``edit`` payload (§4.1);
    ``expires_at`` is when the interrupt expires (§4.4).
    """

    interrupt_id: str
    kind: str
    scope: str | None = None
    approver_role: str | None = None
    actions: list[str]
    payload: dict[str, Any]
    edit_schema: dict[str, Any] | None = None
    expires_at: str | None = None


class InterruptResolvedPayload(_Payload):
    """``interrupt.resolved``: ``{interrupt_id, action, resolved_by}`` (PRD §4.2)."""

    interrupt_id: str
    action: str
    resolved_by: str


class ArtifactCreatedPayload(_Payload):
    """``artifact.created`` (PRD §3.4).

    ``{artifact_id, question_id, session_id, kind, renderer?, name, mime_type,
    size_bytes?, preview, view_url, download_url, metadata}``. ``question_id``
    is the ``run_id`` and ``session_id`` the ``thread_id`` (§3.1). ``preview``
    is the SSE preview block per kind (§3.4 table); ``view_url`` /
    ``download_url`` are the gateway artifact routes (§2.5).
    """

    artifact_id: str
    question_id: str
    session_id: str
    kind: str
    renderer: str | None = None
    name: str
    mime_type: str | None = None
    size_bytes: int | None = None
    preview: dict[str, Any] | None = None
    view_url: str | None = None
    download_url: str | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class EventEnvelope(pydantic.BaseModel):
    """The one wire envelope every event rides (PRD §3.3).

    Field order is the contract — the JSON object starts with ``"v"``:

    ``{"v":"1.0","type":"tool.call","run_id":"r_…","thread_id":"t_…",
    "agent":"sql-analyst","seq":42,"ts":"…","ns":[],"payload":{}}``

    ``ns`` is the subgraph namespace path, ``[]`` at the graph root (A31
    drivers preserve it). ``ts`` is UTC ISO-8601; ``seq`` is per-run,
    monotonically increasing from 1 — allocated by ``SeqCounter`` in memory
    or by the Postgres log (``vdreamer.eventlog``), never guessed here.
    """

    model_config = pydantic.ConfigDict(populate_by_name=True)

    v: str = PROTOCOL_VERSION
    type: EventType
    run_id: str
    thread_id: str
    agent: str
    seq: int
    ts: str = pydantic.Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    ns: tuple[str, ...] = ()
    payload: dict[str, Any] = pydantic.Field(default_factory=dict)

    @property
    def sse_id(self) -> str:
        """The SSE ``id:`` value — ``"{run_id}:{seq}"`` (PRD §3.3, A3)."""
        return f"{self.run_id}:{self.seq}"
