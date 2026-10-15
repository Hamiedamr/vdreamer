"""The one stream adapter module (PRD §2.4 "Event adapter", A31, A43).

A31: **all** LangGraph streaming specifics live in this module and only here.
Two drivers emit identical protocol events (:class:`EventEnvelope`,
`vdreamer.events`) given the same graph execution, selected by the
``VDREAMER_STREAM_DRIVER`` environment variable:

- **v2** (:class:`V2StreamDriver`, default, stable) —
  ``astream(stream_mode=["messages","updates","custom"], subgraphs=True,
  version="v2")``. v2 reports interrupts on ``values`` parts, which this
  driver does not subscribe to, so pending interrupts are read from
  ``aget_state(config).interrupts`` after the stream ends (PRD §2.4).
- **v3** (:class:`V3StreamDriver`, opt-in, beta) —
  ``astream_events(version="v3")``, verified against the installed
  langgraph 1.2.15 (see the class docstring for the concrete API facts).

Everything driver-specific is normalized *before* envelope construction by
module-private mappers (:class:`_RunMapper` and friends), so the two drivers
cannot drift: each driver only translates its native stream parts into
``mapper`` calls, and one shared code path turns those into protocol events.

Event mapping (PRD §3.3, §4.2, §4.5/A44, D18/A20):

- ``run.status`` — ``running`` at stream start, then ``completed`` /
  ``failed`` (with ``error {code, message}``) / ``awaiting_approval`` when
  pending interrupts exist.
- ``token.delta`` / ``thinking.delta`` — per-message deltas, coalesced by
  :class:`DeltaBatcher` (50 ms / 32 tokens at emission, §3.3).
  ``thinking.delta`` is suppressed entirely when ``expose_thinking`` is
  false (the spec is the driver's, not the payload model's).
- ``message.completed`` — on message completion (§3.3 payload).
- ``tool.call`` / ``tool.result`` — governed tools (§2.7,
  ``vdreamer.governance``) are the single emitter: their records arrive as
  custom data while the call runs, and the mapper turns them into the
  events. A call *nobody* governs falls back to the ungoverned mapping: its
  AIMessage ``tool_call`` is held, then emitted as ``tool.call`` with default
  redaction (A33) right before its ToolMessage's ``tool.result`` — or at run
  end, if the call never ran (a review interrupt). Holding the call until it
  executes is what lets a governed record replace it: the AIMessage arrives
  before the mapper can know whether a policy will redact more.
- custom data — vdreamer-enveloped records (the exact shapes
  ``vdreamer.context.emit_*`` buffer/stream) become ``progress`` / ``log`` /
  ``artifact.created`` / ``citation`` events; any other custom data becomes
  a ``log`` event. Subgraph ``ns`` is preserved (root → ``[]``).
- ``interrupt.requested`` — one per pending interrupt, normalized per §4.5
  (A44): a dict interrupt value carrying its own ``kind`` (every
  ``require_approval`` value does) keeps it; a dict without ``kind`` or a
  non-dict ``interrupt(value)`` becomes kind ``"generic"`` with actions
  ``["respond"]``.

No SSE transport lives here (that is the gateway's, M4); drivers are
usable in-process against any :class:`vdreamer.eventlog.EventSink` —
tests and ``vdreamer dev`` use :class:`InMemoryEventSink`.
"""

from __future__ import annotations

import os
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from vdreamer.artifacts import artifact_urls
from vdreamer.context import DEFAULT_APPROVAL_ACTIONS
from vdreamer.eventlog import SeqCounter
from vdreamer.events import (
    DELTA_BATCH_INTERVAL_S,
    DELTA_BATCH_MAX_TOKENS,
    ArtifactCreatedPayload,
    CitationPayload,
    CitationSource,
    ErrorInfo,
    EventEnvelope,
    EventType,
    InterruptRequestedPayload,
    LogPayload,
    MessageCompletedPayload,
    ProgressPayload,
    RunStatusPayload,
    TokenDeltaPayload,
    ToolCallPayload,
    ToolResultPayload,
)

# The §4.5 kinds live with the record; re-exported here, where interrupts normalize.
from vdreamer.interrupts import MCP_ELICITATION_KIND as MCP_ELICITATION_KIND
from vdreamer.interrupts import TOOL_REVIEW_KIND as TOOL_REVIEW_KIND
from vdreamer.tools import redact_args

#: Selects the stream driver (PRD §2.4, A43): ``v2`` (default, stable) or ``v3`` (beta).
STREAM_DRIVER_ENV_VAR: Final[str] = "VDREAMER_STREAM_DRIVER"

#: The valid :data:`STREAM_DRIVER_ENV_VAR` values (A43).
VALID_STREAM_DRIVERS: Final[tuple[str, ...]] = ("v2", "v3")

#: The vLLM/DeepSeek-style think format tags (PRD §3.3 line 212, D18). Built by
#: concatenation so the literal tags never appear verbatim in this source file.
_THINK_OPEN: Final[str] = "<" + "think" + ">"
_THINK_CLOSE: Final[str] = "<" + "/" + "think" + ">"

V2_STREAM_MODES: Final[tuple[str, ...]] = ("messages", "updates", "custom")

#: Truncation for the ``log`` fallback of unshaped custom data (§3.3 has no
#: cap for logs; 500 chars keeps SSE frames small while staying readable).
_LOG_FALLBACK_MAX_CHARS: Final[int] = 500

#: ``tool.result`` summaries are capped (§2.7 step 5 uses 8,000 for governed
#: results; the M2 ungoverned mapping applies the same cap to ToolMessages).
_TOOL_SUMMARY_MAX_CHARS: Final[int] = 8000


def stream_driver_from_env(
    env: Mapping[str, str] | None = None,
) -> Literal["v2", "v3"]:
    """Resolve :data:`STREAM_DRIVER_ENV_VAR` to ``"v2" | "v3"`` (A43).

    Default ``"v2"`` — the stable stream is the default driver (PRD §2.4).
    Any other value is a :exc:`ValueError` naming the valid values. The
    check runs at run start (when a driver is made), never at import.
    """
    source: Mapping[str, str] = env if env is not None else os.environ
    raw = (source.get(STREAM_DRIVER_ENV_VAR) or "").strip() or "v2"
    if raw not in VALID_STREAM_DRIVERS:
        raise ValueError(
            f"{STREAM_DRIVER_ENV_VAR} must be one of {list(VALID_STREAM_DRIVERS)}, got {raw!r}"
        )
    return cast(Literal["v2", "v3"], raw)


def make_stream_driver(name: Literal["v2", "v3"] | str = "v2") -> StreamDriver:
    """Build the named driver; unknown names raise :exc:`ValueError`."""
    if name == "v2":
        return V2StreamDriver()
    if name == "v3":
        return V3StreamDriver()
    raise ValueError(
        f"stream driver must be one of {list(VALID_STREAM_DRIVERS)}, got {name!r}"
    )


# ---------------------------------------------------------------------------
# Delta batching (§3.3: "Deltas batched at emission (50 ms / 32 tokens)")
# ---------------------------------------------------------------------------


@dataclass
class _Batch:
    """Accumulated deltas for one ``(kind, message_id)`` batcher key."""

    start: float
    parts: list[str] = field(default_factory=list)


class DeltaBatcher:
    """Coalesce consecutive deltas per message (PRD §3.3, WP2 drivers' shared core).

    Both drivers feed the same batcher: ``add`` buffers text per
    ``(kind, message_id)`` key (``kind`` discriminates ``token`` from
    ``thinking`` — different protocol types batch independently) and flushes
    a key when its buffered delta count reaches ``max_deltas`` (the "32
    tokens" threshold — deltas are counted, an approximation of tokens) or
    when ``interval_s`` elapsed since the batch opened (checked by
    :meth:`flush_ready`, which the drivers call after every stream part).
    ``flush_all`` drains everything at stream end, before the final
    ``run.status``. The clock is injectable for tests.

    Timing thresholds default to :data:`DELTA_BATCH_INTERVAL_S` /
    :data:`DELTA_BATCH_MAX_TOKENS` (§3.3 line 211).
    """

    def __init__(
        self,
        *,
        interval_s: float = DELTA_BATCH_INTERVAL_S,
        max_deltas: int = DELTA_BATCH_MAX_TOKENS,
        clock: Any = None,
    ) -> None:
        self._interval_s = interval_s
        self._max_deltas = max_deltas
        self._clock = clock if clock is not None else time.monotonic
        self._batches: dict[tuple[str, str], _Batch] = {}

    def add(self, key: tuple[str, str], text: str) -> list[tuple[tuple[str, str], str]]:
        """Buffer one delta; returns batches this add flushed (count threshold)."""
        batch = self._batches.get(key)
        if batch is None:
            batch = _Batch(start=self._clock())
            self._batches[key] = batch
        batch.parts.append(text)
        if len(batch.parts) >= self._max_deltas:
            return [(key, self._flush(key))]
        return []

    def flush_ready(self) -> list[tuple[tuple[str, str], str]]:
        """Flush every batch open longer than ``interval_s`` (time threshold)."""
        now = self._clock()
        out: list[tuple[tuple[str, str], str]] = []
        for key in list(self._batches):
            if now - self._batches[key].start >= self._interval_s:
                out.append((key, self._flush(key)))
        return out

    def flush_all(self) -> list[tuple[tuple[str, str], str]]:
        """Drain every open batch (stream end)."""
        return [(key, self._flush(key)) for key in list(self._batches)]

    def _flush(self, key: tuple[str, str]) -> str:
        batch = self._batches.pop(key)
        return "".join(batch.parts)


# ---------------------------------------------------------------------------
# Thinking extraction (§3.3 line 212, D18/A20)
# ---------------------------------------------------------------------------


class ThinkTagSplitter:
    """Stateful think-tag splitter (PRD §3.3 line 212: "tags may span chunks").

    The third thinking-extraction fallback, after standard ``reasoning``
    content blocks and the vLLM ``reasoning_content`` field: content
    arriving before the closing tag routes as thinking, content
    after it routes as text. The splitter buffers a trailing partial tag
    (``"…</"`` → hold ``"</"``, ``"thi"`` → hold the partial tag) so a tag split
    across chunk boundaries is never emitted as content. Multiple chunks feed
    the same instance; one splitter lives per message (the driver keys them
    by ``message_id``).

    **Wiring:** a *per-alias* fallback (PRD §3.3 line 212). The drivers do
    not route text through it: since Milestone 7 a ``thinking: think_tags``
    alias parses its tags at the model layer (``vdreamer.models``), so the
    chunks arrive here as standard ``reasoning`` blocks — on both drivers,
    in the final message, and in the history sent back to the model. This
    splitter stays a public primitive for agents that parse tags themselves.

    If the stream ends without any closing tag (a model that turned out not
    to use the think format), :meth:`close` routes the held-back tail as
    *text* — a judgment call the PRD does not cover: treating it as thinking
    would mislabel every token of a non-think model.
    """

    def __init__(self) -> None:
        self._tail = ""
        self._saw_close = False

    def feed(self, text: str) -> tuple[str, str]:
        """Feed one chunk; returns ``(thinking_delta, text_delta)``."""
        data = self._tail + text
        self._tail = ""
        if self._saw_close:
            return "", data
        out_thinking: list[str] = []
        index = data.find(_THINK_CLOSE)
        if index >= 0:
            out_thinking.append(data[:index])
            data = data[index + len(_THINK_CLOSE) :]
            self._saw_close = True
            return "".join(out_thinking), data
        # No closing tag in this chunk: hold back the longest suffix of the
        # buffered data that could be the start of the tag, emit the rest as
        # thinking.
        hold = min(len(data), len(_THINK_CLOSE) - 1)
        for size in range(hold, 0, -1):
            suffix = data[-size:]
            if _THINK_CLOSE.startswith(suffix):
                self._tail = suffix
                data = data[:-size]
                break
        out_thinking.append(data)
        return "".join(out_thinking), ""

    def close(self) -> tuple[str, str]:
        """Finish the message; returns any held-back remainder."""
        tail, self._tail = self._tail, ""
        if not self._saw_close:
            # No closing tag was ever seen: the model did not use the format.
            return "", tail
        return tail, ""


def chunk_text_and_thinking(chunk: BaseMessage) -> tuple[str, str]:
    """One streaming chunk → ``(text_delta, thinking_delta)`` (D18/A20 order).

    1. Standard ``reasoning``/``thinking`` content blocks on the chunk — any
       reasoning text found wins for the chunk.
    2. The vLLM-style ``reasoning_content`` additional field.

    3. Think tags, the *per-alias* fallback (PRD §3.3 line 212: "per-alias
       fallbacks apply"), are **not parsed here**: the ``models.yaml``
       alias's ``thinking`` mode (``vdreamer.models``, Milestone 7) turns
       ``reasoning_content`` fields and ``<think>`` tags into standard
       ``reasoning`` blocks inside the chat model, so both drivers read
       rule 1 and stay at parity.
    """
    content = chunk.content
    text = ""
    reasoning = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, Sequence):
        for block in content:
            if isinstance(block, str):
                text += block
            elif isinstance(block, Mapping):
                block_type = block.get("type")
                if block_type in ("reasoning", "thinking"):
                    reasoning += str(
                        block.get("reasoning")
                        or block.get("thinking")
                        or block.get("text")
                        or ""
                    )
                elif block_type == "text":
                    text += str(block.get("text") or "")
                # Other blocks (tool_call, image, …) carry no delta text.
    if reasoning:
        return text, reasoning
    extra = (
        chunk.additional_kwargs.get("reasoning_content") if chunk.additional_kwargs else None
    )
    if isinstance(extra, str) and extra:
        return text, extra
    return text, ""


# ---------------------------------------------------------------------------
# Interrupt normalization (§4.2 record, §4.5/A44)
# ---------------------------------------------------------------------------


def normalize_interrupt(value: Any, interrupt_id: str) -> InterruptRequestedPayload:
    """One interrupt value → the ``interrupt.requested`` payload (§4.5, A44).

    A dict value carrying its own ``kind`` (every ``require_approval``
    value does) declares its own ``actions``/``approver_role``/``scope``/
    ``edit_schema``/TTL. A dict without ``kind`` or a raw non-dict
    ``interrupt(value)`` normalizes to kind ``"generic"`` with the single
    action ``"respond"`` (§4.5 table); the resolver payload rides the
    ``payload`` field verbatim.
    """
    if (
        isinstance(value, Mapping)
        and isinstance(value.get("action_requests"), Sequence)
        and isinstance(value.get("review_configs"), Sequence)
    ):
        return _normalize_tool_review(value, interrupt_id)
    if isinstance(value, Mapping) and value.get("type") == MCP_ELICITATION_KIND:
        # langchain.mcp elicitation (§4.5): answered with edit (the form
        # answers) or reject (decline); the gateway builds {"responses": ...}.
        requests = value.get("requests")
        return InterruptRequestedPayload(
            interrupt_id=interrupt_id,
            kind=MCP_ELICITATION_KIND,
            actions=["edit", "reject"],
            payload={
                "tool_name": str(value.get("tool_name") or ""),
                "requests": list(requests) if isinstance(requests, Sequence) else [],
            },
            expires_at=(datetime.now(UTC) + timedelta(hours=24)).isoformat(),
        )
    if isinstance(value, Mapping) and "kind" in value:
        actions = value.get("actions")
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)) or not actions:
            actions = DEFAULT_APPROVAL_ACTIONS
        payload = value.get("payload")
        edit_schema = value.get("edit_schema")
        ttl_hours = value.get("ttl_hours", 24)
        try:
            ttl_hours = int(ttl_hours)
        except (TypeError, ValueError):
            ttl_hours = 24
        return InterruptRequestedPayload(
            interrupt_id=interrupt_id,
            kind=str(value["kind"]),
            scope=value.get("scope") if isinstance(value.get("scope"), str) else None,
            approver_role=(
                value.get("approver_role") if isinstance(value.get("approver_role"), str) else None
            ),
            actions=[str(action) for action in actions],
            payload=dict(payload) if isinstance(payload, Mapping) else {},
            edit_schema=dict(edit_schema) if isinstance(edit_schema, Mapping) else None,
            expires_at=(datetime.now(UTC) + timedelta(hours=ttl_hours)).isoformat(),
        )
    if isinstance(value, Mapping):
        # Raw ``interrupt(dict)`` — generic; the dict is the responder payload
        # verbatim (§4.5 "resolver payload, verbatim").
        return InterruptRequestedPayload(
            interrupt_id=interrupt_id,
            kind="generic",
            actions=["respond"],
            payload=dict(value),
        )
    return InterruptRequestedPayload(
        interrupt_id=interrupt_id,
        kind="generic",
        actions=["respond"],
        payload={"value": value},
    )


def _normalize_tool_review(
    value: Mapping[str, Any], interrupt_id: str
) -> InterruptRequestedPayload:
    """``HumanInTheLoopMiddleware``'s ``HITLRequest`` → kind ``tool_review`` (§4.5).

    The payload keeps the request's shape — ``action_requests[]`` (each call's
    ``name``/``args``/``description``, args redacted with the A33 defaults: the
    approver judges the call, an ``approve`` runs the model's own) and
    ``review_configs[]`` (each tool's ``allowed_decisions``, ``args_schema``).
    ``actions`` is the union of the allowed decisions, in first-seen order; the
    gateway checks each decision against its own request's list and builds
    ``{"decisions": [...]}`` in request order.
    """
    requests = [
        {
            **dict(request),
            "args": redact_args(dict(request.get("args") or {})),
        }
        for request in value["action_requests"]
        if isinstance(request, Mapping)
    ]
    configs = [dict(config) for config in value["review_configs"] if isinstance(config, Mapping)]
    allowed = {
        str(config.get("action_name")): config.get("allowed_decisions") for config in configs
    }
    actions: list[str] = []
    for request in requests:
        for decision in allowed.get(str(request.get("name"))) or ():
            if str(decision) not in actions:
                actions.append(str(decision))
    return InterruptRequestedPayload(
        interrupt_id=interrupt_id,
        kind=TOOL_REVIEW_KIND,
        actions=actions or ["approve", "reject"],
        payload={"action_requests": requests, "review_configs": configs},
        expires_at=(datetime.now(UTC) + timedelta(hours=24)).isoformat(),
    )


# ---------------------------------------------------------------------------
# Ungoverned tool-event mapping (calls no governance pipeline emitted)
# ---------------------------------------------------------------------------


def _tool_call_envelope(
    envelope: EnvelopeFn, ns: Sequence[str], tool_call: Mapping[str, Any]
) -> EventEnvelope:
    """One AIMessage ``tool_call`` dict → ``tool.call`` with default redaction (A33).

    ``source`` is ``"custom"``: an ungoverned call carries no tool object to
    classify (§2.4: ungoverned tool messages are mapped with default
    redaction); governed calls report their real source.
    """
    args = tool_call.get("args")
    return envelope(
        EventType.TOOL_CALL,
        ToolCallPayload(
            tool_call_id=str(tool_call.get("id") or ""),
            name=str(tool_call.get("name") or ""),
            source="custom",
            args=redact_args(args) if isinstance(args, Mapping) else {},
        ).event_payload(),
        ns,
    )


def _tool_result_envelope(
    envelope: EnvelopeFn, ns: Sequence[str], message: ToolMessage
) -> EventEnvelope:
    """One ToolMessage → ``tool.result`` (ungoverned mapping).

    ``duration_ms`` is 0 and ``artifact_ids`` empty — timing and overflow
    artifacts are the governance pipeline's (§2.7 steps 5/7). Full output is
    never streamed (§3.3): the content text rides ``summary`` capped at 8,000.
    """
    status = str(getattr(message, "status", None) or "ok")
    content = message.text
    error = (
        ErrorInfo(code="tool_error", message=content[:_TOOL_SUMMARY_MAX_CHARS])
        if status == "error"
        else None
    )
    return envelope(
        EventType.TOOL_RESULT,
        ToolResultPayload(
            tool_call_id=str(message.tool_call_id),
            status=cast("Literal['ok', 'error']", status if status in ("ok", "error") else "ok"),
            error=error,
            summary=content[:_TOOL_SUMMARY_MAX_CHARS],
            artifact_ids=[],
            duration_ms=0,
        ).event_payload(),
        ns,
    )


#: The mapper's envelope constructor signature (``type, payload, ns``).
EnvelopeFn = Callable[[EventType, Mapping[str, Any], Sequence[str]], EventEnvelope]


# ---------------------------------------------------------------------------
# Shared per-run mapper (one code path → both drivers cannot drift)
# ---------------------------------------------------------------------------

class _RunMapper:
    """Turns normalized driver events into protocol envelopes (shared by both drivers).

    Owns all per-run envelope state: seq allocation (a fresh
    :class:`SeqCounter` per run), delta batching, per-message thinking
    accumulation, citation bookkeeping, and completed-message dedup. The
    drivers translate their native stream parts into calls and yield what
    comes back; no envelope logic lives in either driver.
    """

    def __init__(self, spec: StreamSpec) -> None:
        self._spec = spec
        self._seq = SeqCounter()
        self._batcher = DeltaBatcher()
        self._completed: set[str] = set()
        self._thinking: dict[str, list[str]] = {}
        self._citations: dict[str, list[str]] = {}
        #: Ungoverned AIMessage tool calls waiting for their ToolMessage.
        self._held_calls: dict[str, tuple[tuple[str, ...], Mapping[str, Any]]] = {}
        #: Call ids a governance pipeline emitted for (never mapped again).
        self._governed: set[str] = set()

    def envelope(
        self, type_: EventType, payload: Mapping[str, Any], ns: Sequence[str] = ()
    ) -> EventEnvelope:
        """Build one envelope with the next ``seq`` (§3.3 field order)."""
        return EventEnvelope(
            type=type_,
            run_id=self._spec.run_id,
            thread_id=self._spec.thread_id,
            agent=self._spec.agent,
            seq=self._seq.next_seq(self._spec.run_id),
            ns=tuple(ns),
            payload=dict(payload),
        )

    # -- run.status --------------------------------------------------------

    def start(self) -> EventEnvelope:
        """``run.status`` ``running`` at stream start (§3.3)."""
        return self.envelope(
            EventType.RUN_STATUS,
            RunStatusPayload(status="running", attempt=self._spec.attempt).event_payload(),
        )

    def finish(
        self,
        *,
        error: BaseException | None = None,
        interrupts: Sequence[Any] = (),
    ) -> list[EventEnvelope]:
        """End-of-run events, in order: flush deltas → interrupts → final status.

        The final ``run.status`` is ``failed`` (with the ``error`` block)
        on an exception, ``awaiting_approval`` when pending interrupts
        exist, else ``completed`` (§3.2/§3.3).
        """
        out: list[EventEnvelope] = []
        for (kind, message_id), text in self._batcher.flush_all():
            type_ = EventType.TOKEN_DELTA if kind == "token" else EventType.THINKING_DELTA
            out.append(
                self.envelope(
                    type_, TokenDeltaPayload(message_id=message_id, text=text).event_payload()
                )
            )
        # Calls that never ran this attempt (e.g. held for a review interrupt).
        for call_id in list(self._held_calls):
            out.extend(self._release_call(call_id))
        seen_interrupt_ids: set[str] = set()
        for interrupt in interrupts:
            interrupt_id = str(getattr(interrupt, "id", None) or f"int_{uuid.uuid4().hex}")
            if interrupt_id in seen_interrupt_ids:
                continue
            seen_interrupt_ids.add(interrupt_id)
            value = getattr(interrupt, "value", interrupt)
            out.append(
                self.envelope(
                    EventType.INTERRUPT_REQUESTED,
                    normalize_interrupt(value, interrupt_id).event_payload(),
                )
            )
        if error is not None:
            payload = RunStatusPayload(
                status="failed",
                attempt=self._spec.attempt,
                # NodeTimeoutError/NodeError get stable codes in M3 (§2.4);
                # M2 carries the generic code.
                error=ErrorInfo(code="node_error", message=str(error)),
            ).event_payload()
        else:
            status = "awaiting_approval" if interrupts else "completed"
            payload = RunStatusPayload(
                status=status, attempt=self._spec.attempt
            ).event_payload()
        out.append(self.envelope(EventType.RUN_STATUS, payload))
        return out

    # -- deltas ------------------------------------------------------------

    def chunk(
        self, ns: Sequence[str], message_id: str, text: str, thinking: str
    ) -> list[EventEnvelope]:
        """One message-stream chunk → batched ``token.delta`` / ``thinking.delta``.

        Thinking is suppressed entirely when ``expose_thinking`` is false
        (§3.3) — no ``thinking.delta`` events and no ``thinking`` on
        ``message.completed``.
        """
        out: list[EventEnvelope] = []
        if text:
            for (_kind, mid), batch in self._batcher.add(("token", message_id), text):
                out.append(
                    self.envelope(
                        EventType.TOKEN_DELTA,
                        TokenDeltaPayload(message_id=mid, text=batch).event_payload(),
                        ns,
                    )
                )
        if thinking:
            if self._spec.expose_thinking:
                self._thinking.setdefault(message_id, []).append(thinking)
                for (_kind, mid), batch in self._batcher.add(("thinking", message_id), thinking):
                    out.append(
                        self.envelope(
                            EventType.THINKING_DELTA,
                            TokenDeltaPayload(message_id=mid, text=batch).event_payload(),
                            ns,
                        )
                    )
        return out

    def pump(self) -> list[EventEnvelope]:
        """Emit any batches the 50 ms interval flushed (drivers call per stream part)."""
        out: list[EventEnvelope] = []
        for (kind, message_id), text in self._batcher.flush_ready():
            type_ = EventType.TOKEN_DELTA if kind == "token" else EventType.THINKING_DELTA
            out.append(
                self.envelope(
                    type_, TokenDeltaPayload(message_id=message_id, text=text).event_payload()
                )
            )
        return out

    # -- custom data -------------------------------------------------------

    def custom(self, ns: Sequence[str], data: Any) -> list[EventEnvelope]:
        """Custom stream data → protocol events (§2.4 "Both").

        Vdreamer-enveloped records (the shapes ``emit_progress`` /
        ``emit_log`` / ``emit_artifact`` / ``emit_citation`` produce) become
        their protocol events; anything else becomes a ``log`` fallback with
        a repr capped at 500 chars.
        """
        if isinstance(data, Mapping) and data.get("type") in ("tool.call", "tool.result"):
            return self._governed_tool_event(ns, data)
        vdreamer_types = ("progress", "log", "artifact", "citation")
        if isinstance(data, Mapping) and data.get("type") in vdreamer_types:
            ctype = str(data["type"])
            if ctype == "progress":
                pct = data.get("pct")
                return [
                    self.envelope(
                        EventType.PROGRESS,
                        ProgressPayload(
                            label=str(data.get("label") or ""),
                            pct=float(pct) if isinstance(pct, (int, float)) else None,
                        ).event_payload(),
                        ns,
                    )
                ]
            if ctype == "log":
                return [
                    self.envelope(
                        EventType.LOG,
                        LogPayload(
                            level=str(data.get("level") or "info"),
                            message=str(data.get("message") or ""),
                        ).event_payload(),
                        ns,
                    )
                ]
            if ctype == "artifact":
                return [self._artifact_event(ns, data)]
            # citation
            message_id = str(data.get("message_id") or "")
            citation_id = str(data.get("citation_id") or f"cit_{uuid.uuid4().hex}")
            source = data.get("source")
            if not isinstance(source, Mapping):
                source = {}
            snippet = str(data.get("snippet") or "")
            score = data.get("score")
            self._citations.setdefault(message_id, []).append(citation_id)
            return [
                self.envelope(
                    EventType.CITATION,
                    CitationPayload(
                        citation_id=citation_id,
                        message_id=message_id,
                        source=CitationSource(
                            artifact_id=(
                                str(source["artifact_id"])
                                if source.get("artifact_id") is not None
                                else None
                            ),
                            uri=str(source["uri"]) if source.get("uri") is not None else None,
                            title=str(source.get("title") or ""),
                        ),
                        locator=(
                            dict(data["locator"])
                            if isinstance(data.get("locator"), Mapping)
                            else None
                        ),
                        snippet=snippet,
                        score=float(score) if isinstance(score, (int, float)) else None,
                    ).event_payload(),
                    ns,
                )
            ]
        message = repr(data)
        return [
            self.envelope(
                EventType.LOG,
                LogPayload(level="info", message=message[:_LOG_FALLBACK_MAX_CHARS]).event_payload(),
                ns,
            )
        ]

    def _artifact_event(self, ns: Sequence[str], data: Mapping[str, Any]) -> EventEnvelope:
        """An ``artifact`` record → ``artifact.created`` (§3.4).

        Every record carries its mime type, size, and preview; one the store
        kept (``stored``: bytes and metadata row already written, D7) also
        carries the gateway URLs. With no store bound (tests, a runner driven
        without a worker) nothing is servable, so no URLs.
        """
        artifact_id = str(data.get("artifact_id") or f"art_{uuid.uuid4().hex}")
        size = data.get("size_bytes")
        preview = data.get("preview")
        mime = data.get("mime_type")
        view_url = download_url = None
        if data.get("stored"):
            view_url, download_url = artifact_urls(artifact_id)
        return self.envelope(
            EventType.ARTIFACT_CREATED,
            ArtifactCreatedPayload(
                artifact_id=artifact_id,
                question_id=self._spec.run_id,
                session_id=self._spec.thread_id,
                kind=str(data.get("kind") or ""),
                renderer=str(data["renderer"]) if isinstance(data.get("renderer"), str) else None,
                name=str(data.get("name") or ""),
                mime_type=mime if isinstance(mime, str) else None,
                size_bytes=size if isinstance(size, int) else None,
                preview=dict(preview) if isinstance(preview, Mapping) else None,
                view_url=view_url,
                download_url=download_url,
                metadata=(
                    dict(data["metadata"]) if isinstance(data.get("metadata"), Mapping) else {}
                ),
            ).event_payload(),
            ns,
        )

    # -- tool events ---------------------------------------------------------

    def _governed_tool_event(
        self, ns: Sequence[str], data: Mapping[str, Any]
    ) -> list[EventEnvelope]:
        """A governance record → its ``tool.call`` / ``tool.result`` (§2.7 steps 2/7)."""
        call_id = str(data.get("tool_call_id") or "")
        self._governed.add(call_id)
        self._held_calls.pop(call_id, None)  # the governed call replaces the fallback
        if data.get("type") == "tool.call":
            args = data.get("args")
            call_payload = ToolCallPayload(
                tool_call_id=call_id,
                name=str(data.get("name") or ""),
                source=str(data.get("source") or "custom"),
                args=dict(args) if isinstance(args, Mapping) else {},
            ).event_payload()
            return [self.envelope(EventType.TOOL_CALL, call_payload, ns)]
        error = data.get("error")
        duration = data.get("duration_ms")
        artifact_ids = data.get("artifact_ids")
        result_payload = ToolResultPayload(
            tool_call_id=call_id,
            status="error" if data.get("status") == "error" else "ok",
            error=(
                ErrorInfo(
                    code=str(error.get("code") or "tool_error"),
                    message=str(error.get("message") or ""),
                )
                if isinstance(error, Mapping)
                else None
            ),
            summary=str(data.get("summary") or "")[:_TOOL_SUMMARY_MAX_CHARS],
            artifact_ids=(
                [str(item) for item in artifact_ids]
                if isinstance(artifact_ids, Sequence) and not isinstance(artifact_ids, str)
                else []
            ),
            duration_ms=duration if isinstance(duration, int) else 0,
        ).event_payload()
        return [self.envelope(EventType.TOOL_RESULT, result_payload, ns)]

    def _release_call(self, call_id: str) -> list[EventEnvelope]:
        """Emit a held ungoverned ``tool.call`` (default redaction)."""
        held = self._held_calls.pop(call_id, None)
        if held is None:
            return []
        ns, tool_call = held
        return [_tool_call_envelope(self.envelope, ns, tool_call)]

    # -- completed messages --------------------------------------------------

    def message(self, ns: Sequence[str], message: BaseMessage) -> list[EventEnvelope]:
        """A completed message → ``tool.call``/``tool.result``/``message.completed``.

        Only AI messages emit ``message.completed`` — token deltas only come
        from model calls, and human/tool activity reaches the client through
        its own turn and ``tool.result`` (a judgment call the PRD leaves to
        the drivers; it also keeps v2 (updates parts) and v3 (values parts)
        at parity, since input messages otherwise appear only on the values
        channel). Dedup is by message id: v3 values parts re-deliver the
        whole message list every superstep.
        """
        message_id = str(message.id) if message.id else f"msg_{uuid.uuid4().hex}"
        if message_id in self._completed:
            return []
        self._completed.add(message_id)
        out: list[EventEnvelope] = []
        if isinstance(message, ToolMessage):
            call_id = str(message.tool_call_id)
            if call_id in self._governed:
                return out  # the pipeline already emitted its tool.result
            out.extend(self._release_call(call_id))
            out.append(_tool_result_envelope(self.envelope, ns, message))
            return out
        if not isinstance(message, AIMessage):
            return out
        for tool_call in message.tool_calls or []:
            call_id = str(tool_call.get("id") or "")
            if not call_id:
                out.append(_tool_call_envelope(self.envelope, ns, tool_call))
            elif call_id not in self._governed:
                self._held_calls[call_id] = (tuple(ns), tool_call)
        # ``usage_metadata`` is a ``UsageMetadata`` TypedDict (a plain dict).
        usage = getattr(message, "usage_metadata", None)
        usage_payload = dict(usage) if usage else None
        thinking = "".join(self._thinking.get(message_id, [])) or None
        out.append(
            self.envelope(
                EventType.MESSAGE_COMPLETED,
                MessageCompletedPayload(
                    message_id=message_id,
                    role=str(message.type),
                    text=message.text,
                    thinking=thinking if self._spec.expose_thinking else None,
                    citation_ids=list(self._citations.get(message_id, [])),
                    usage=usage_payload,
                ).event_payload(),
                ns,
            )
        )
        return out


# ---------------------------------------------------------------------------
# The drivers (A31)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamSpec:
    """What the drivers need to build envelopes for one run.

    The runner (``AgentRunner.arun(sink=...)``) derives this from its
    :class:`~vdreamer.runtime.RunInfo` plus the agent's ``expose_thinking``
    and whether it holds a checkpointer (which gates the ``durability=``
    kwarg exactly like ``runtime.AgentRunner._execute`` does).
    """

    run_id: str
    thread_id: str
    agent: str
    attempt: int = 0
    expose_thinking: bool = True
    has_checkpointer: bool = False


class StreamDriver(ABC):
    """Base for the protocol-event adapters (A31): one per LangGraph streaming API.

    ``astream`` is an async generator of :class:`EventEnvelope` usable
    without any live Postgres (tests run compiled graphs on
    ``InMemorySaver``). After the generator is exhausted, ``result`` holds
    the run's final value (the same thing ``ainvoke`` would have returned,
    i.e. the final state) — the runner returns that from ``arun``.
    """

    result: Any = None
    """The final graph value of the last :meth:`astream` call (final state)."""

    @abstractmethod
    def astream(
        self,
        graph: Any,
        input: Any,
        *,
        config: RunnableConfig | None = None,
        context: Any = None,
        durability: str | None = None,
        spec: StreamSpec,
    ) -> AsyncIterator[EventEnvelope]:
        """Stream one run of ``graph`` as protocol events."""


class V2StreamDriver(StreamDriver):
    """The default, stable driver (PRD §2.4/A43): ``astream(version="v2")``.

    Verified against langgraph 1.2.15: with ``version="v2"`` and a
    ``stream_mode`` sequence, ``astream`` yields dicts
    ``{"type": <mode>, "ns": tuple[str, ...], "data": <mode payload>}`` —
    ``messages`` parts carry ``(AIMessageChunk, metadata)`` tuples,
    ``updates`` parts carry ``{node: update}`` (or ``{"__interrupt__": (Interrupt, ...)}``),
    ``custom`` parts carry the writer's data verbatim.

    Pending interrupts are read from ``aget_state(config).interrupts`` after
    the stream ends (v2 reports interrupts on ``values`` parts, which this
    driver does not subscribe to — PRD §2.4). When the graph has no
    checkpointer (``aget_state`` raises ``ValueError: No checkpointer set``),
    the driver falls back to the ``__interrupt__`` ``updates`` parts
    collected during the stream, and reconstructs the final value by
    shallow-merging root updates — channel reducers are not re-applied, a
    documented gap only reachable without a checkpointer.
    """

    async def astream(
        self,
        graph: Any,
        input: Any,
        *,
        config: RunnableConfig | None = None,
        context: Any = None,
        durability: str | None = None,
        spec: StreamSpec,
    ) -> AsyncIterator[EventEnvelope]:
        if spec is None:  # pragma: no cover - defensive; spec is required
            raise ValueError("spec is required")
        mapper = _RunMapper(spec)
        collected_interrupts: list[Any] = []
        merged: dict[str, Any] = {}
        yield mapper.start()
        try:
            stream_kwargs: dict[str, Any] = {}
            # Same durability/context rule as runtime.AgentRunner._execute:
            # durability only means anything with a checkpointer present
            # (langgraph 1.2.15 raises on durability="sync" without one).
            if spec.has_checkpointer and durability is not None:
                stream_kwargs["durability"] = durability
            if context is not None:
                stream_kwargs["context"] = context
            async for part in graph.astream(
                input,
                config,
                stream_mode=V2_STREAM_MODES,
                subgraphs=True,
                version="v2",
                **stream_kwargs,
            ):
                for envelope in self._part(mapper, part, collected_interrupts, merged):
                    yield envelope
                for envelope in mapper.pump():
                    yield envelope
        except Exception as error:
            self.result = None
            for envelope in mapper.finish(error=error):
                yield envelope
            raise
        pending: Sequence[Any] = collected_interrupts
        final: Any = merged
        if spec.has_checkpointer:
            snapshot = await graph.aget_state(config)
            final = snapshot.values
            pending = list(snapshot.interrupts or ()) or collected_interrupts
        self.result = final
        for envelope in mapper.finish(error=None, interrupts=pending):
            yield envelope

    def _part(
        self,
        mapper: _RunMapper,
        part: Any,
        collected_interrupts: list[Any],
        merged: dict[str, Any],
    ) -> list[EventEnvelope]:
        """One v2 stream part → mapper calls."""
        if not isinstance(part, Mapping):
            return []
        mode = part.get("type")
        ns = tuple(part.get("ns") or ())
        data = part.get("data")
        if mode == "custom":
            return mapper.custom(ns, data)
        if mode == "messages":
            if isinstance(data, Sequence) and not isinstance(data, (str, bytes)) and len(data) == 2:
                chunk = data[0]
                # Only AI-message chunks map to token deltas: tool output rides
                # ``tool.result`` (v3 streams ToolMessages as tool-output deltas,
                # never text deltas — this keeps the drivers at parity).
                if isinstance(chunk, AIMessage):
                    message_id = str(chunk.id) if chunk.id else f"msg_{uuid.uuid4().hex}"
                    text, thinking = chunk_text_and_thinking(chunk)
                    return mapper.chunk(ns, message_id, text, thinking)
            return []
        if mode == "updates" and isinstance(data, Mapping):
            if "__interrupt__" in data:
                collected_interrupts.extend(data["__interrupt__"])
                return []
            out: list[EventEnvelope] = []
            for update in data.values():
                if not isinstance(update, Mapping):
                    continue
                if not ns:
                    merged.update(update)
                for value in update.values():
                    messages = value if isinstance(value, Sequence) else [value]
                    for message in messages:
                        if isinstance(message, BaseMessage):
                            out.extend(mapper.message(ns, message))
            return out
        return []


class V3StreamDriver(StreamDriver):
    """The opt-in beta driver (PRD §2.4/A43): ``astream_events(version="v3")``.

    API facts verified against the installed langgraph 1.2.15 (do not
    invent event names — v3 here is *not* langchain-core's
    ``on_chat_model_stream`` dict protocol):

    - ``await graph.astream_events(input, config, version="v3", ...)`` returns
      an ``AsyncGraphRunStream`` (``langgraph.stream.run_stream``); async
      iteration over the run yields ``ProtocolEvent`` dicts
      ``{"method": <stream mode>, "params": {"namespace": list[str],
      "data": ..., "interrupts": tuple, "timestamp": ...}}`` and drives the
      graph (caller-pumped).
    - ``messages`` events carry ``(payload, metadata)`` where ``payload`` is
      a `langchain_protocol` lifecycle dict keyed by
      ``payload["event"]``: ``message-start`` (``id``, ``role``),
      ``content-block-start``/``-delta``/``-finish`` (``index``, content /
      ``delta``), ``message-finish`` (``usage``). Deltas are
      ``{"type": "text-delta", "text"}`` / ``{"type": "reasoning-delta",
      "reasoning"}`` — ``data-delta``/``block-delta`` carry no message text
      and are skipped in M2 (recorded gap; the WP3 golden suite gates v3).
    - ``values`` events carry the full state plus
      ``params["interrupts"]`` (Interrupt objects); ``custom`` events carry
      the stream writer's data verbatim.
    - There is no ``on_interrupt`` event in 1.2.15; interrupts come from the
      values parts (the same source ``run.interrupts()`` aggregates), which
      the driver prefers, falling back to ``aget_state`` when a checkpointer
      exists and the stream yielded none.
    - ``custom`` stream data is only delivered when a transformer collecting
      the ``custom`` mode is registered, so the driver passes the
      ``CustomTransformer`` **class** (``langgraph.stream.transformers``;
      transformers must be scope-aware callables, not pre-built instances);
      the default mux registers only values/messages/lifecycle/subgraph.
    - Execution kwargs (``context=``, ``durability=``) are forwarded to the
      underlying ``astream`` per the ``astream_events`` docstring.
    """

    async def astream(
        self,
        graph: Any,
        input: Any,
        *,
        config: RunnableConfig | None = None,
        context: Any = None,
        durability: str | None = None,
        spec: StreamSpec,
    ) -> AsyncIterator[EventEnvelope]:
        if spec is None:  # pragma: no cover - defensive; spec is required
            raise ValueError("spec is required")
        from langgraph.stream.transformers import CustomTransformer

        mapper = _RunMapper(spec)
        current_message_ids: dict[tuple[str, ...], str] = {}
        interrupts: list[Any] = []
        seen_interrupt_ids: set[str] = set()
        yield mapper.start()
        stream_kwargs: dict[str, Any] = {}
        if spec.has_checkpointer and durability is not None:
            stream_kwargs["durability"] = durability
        if context is not None:
            stream_kwargs["context"] = context
        run = await graph.astream_events(
            input,
            config,
            version="v3",
            # The class, not an instance: transformers must be scope-aware
            # callables ("got pre-built instance" otherwise, 1.2.15).
            transformers=[CustomTransformer],
            **stream_kwargs,
        )
        try:
            async for event in run:
                for envelope in self._event(
                    mapper, event, current_message_ids, interrupts, seen_interrupt_ids
                ):
                    yield envelope
                for envelope in mapper.pump():
                    yield envelope
        except Exception as error:
            self.result = None
            for envelope in mapper.finish(error=error):
                yield envelope
            raise
        finally:
            # Idempotent: no-op once the run is exhausted, cancels on error paths.
            await run.abort()
        if not interrupts and spec.has_checkpointer:
            # Fallback per §2.4: read the run's pending interrupts directly.
            try:
                snapshot = await graph.aget_state(config)
            except Exception:  # noqa: S110 - best-effort fallback only
                snapshot = None
            if snapshot is not None:
                for interrupt in snapshot.interrupts or ():
                    interrupt_id = str(getattr(interrupt, "id", None) or "")
                    if interrupt_id not in seen_interrupt_ids:
                        interrupts.append(interrupt)
                        seen_interrupt_ids.add(interrupt_id)
        self.result = await run.output()
        for envelope in mapper.finish(error=None, interrupts=interrupts):
            yield envelope

    def _event(
        self,
        mapper: _RunMapper,
        event: Mapping[str, Any],
        current_message_ids: dict[tuple[str, ...], str],
        interrupts: list[Any],
        seen_interrupt_ids: set[str],
    ) -> list[EventEnvelope]:
        """One v3 protocol event → mapper calls."""
        method = event.get("method")
        params = event.get("params")
        if not isinstance(params, Mapping):
            return []
        ns = tuple(params.get("namespace") or ())
        data = params.get("data")
        if method == "custom":
            return mapper.custom(ns, data)
        if method == "messages":
            if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
                return []
            if len(data) != 2:
                return []
            payload = data[0]
            if not isinstance(payload, Mapping):
                return []
            payload_type = payload.get("event")
            if payload_type == "message-start":
                current_message_ids[ns] = str(payload.get("id") or "")
                return []
            if payload_type == "content-block-delta":
                delta = payload.get("delta")
                if not isinstance(delta, Mapping):
                    return []
                message_id = current_message_ids.get(ns) or f"msg_{uuid.uuid4().hex}"
                delta_type = delta.get("type")
                if delta_type == "text-delta":
                    return mapper.chunk(ns, message_id, str(delta.get("text") or ""), "")
                if delta_type == "reasoning-delta":
                    return mapper.chunk(ns, message_id, "", str(delta.get("reasoning") or ""))
                return []  # data-delta / block-delta: M2 gap, see class docstring
            return []
        if method == "values":
            for interrupt in params.get("interrupts") or ():
                interrupt_id = str(getattr(interrupt, "id", None) or "")
                if interrupt_id not in seen_interrupt_ids:
                    interrupts.append(interrupt)
                    seen_interrupt_ids.add(interrupt_id)
            out: list[EventEnvelope] = []
            values = params.get("data")
            if isinstance(values, Mapping):
                for value in values.values():
                    items = value if isinstance(value, Sequence) else [value]
                    for item in items:
                        if isinstance(item, BaseMessage):
                            out.extend(mapper.message(ns, item))
            return out
        return []
