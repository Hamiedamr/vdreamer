"""Golden expectations and shared runners for the WP3 golden-event suite.

Suite contract (PRD §2.4 A31/A43): for every scenario the *same* graph runs
through both drivers — ``VDREAMER_STREAM_DRIVER=v2`` and ``=v3`` via the kit's
``driver_env`` — and the suite asserts

1. **parity** — the normalized envelope sequences are identical between the
   drivers (``_kit.normalized`` drops only per-run allocations: ``seq``/``ts``,
   ``expires_at``, run-unique uuids, provider-assigned message ids); and
2. **golden match** — each sequence equals the golden expectation recorded
   here as data: the ordered event types (:data:`GOLDEN_SEQUENCES`, recorded
   from the drivers' actual output on langgraph 1.2.15 — re-record
   *deliberately* when a driver change is intended, never silently) plus the
   §3.3 payload-shape table (:data:`PAYLOAD_REQUIRED_KEYS` +
   :func:`assert_event_shapes`) applied to *every* event, so drift is caught
   against the protocol, not just between drivers.

No live Postgres anywhere: runs go through ``AgentRunner(sink=...)`` on
``InMemorySaver`` into an ``InMemoryEventSink``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Final

import _kit
import pytest

from vdreamer.eventlog import EventEnvelope, InMemoryEventSink
from vdreamer.events import PROTOCOL_VERSION
from vdreamer.runtime import AgentRunner
from vdreamer.streaming import _LOG_FALLBACK_MAX_CHARS, _TOOL_SUMMARY_MAX_CHARS

#: The golden event-type sequences, one per scenario (PRD §3.3). Recorded from
#: the drivers' actual output; each scenario test asserts both drivers against
#: its entry. Terminal ``run.status`` payloads distinguish
#: completed/awaiting_approval/failed — the type sequence alone cannot.
GOLDEN_SEQUENCES: Final[dict[str, tuple[str, ...]]] = {
    # ChattyAgent: custom records stream while the node runs; the message's
    # open delta batch (fake models finish inside the 50 ms window) flushes
    # right before its ``message.completed`` — never trailing it.
    "chat_custom": (
        "run.status",
        "progress",
        "log",
        "artifact.created",
        "citation",
        "log",
        "token.delta",
        "message.completed",
        "run.status",
    ),
    # ToolAgent: the AIMessage carrying the tool call also completes as an
    # (empty-text) message. The call is ungoverned (a raw ToolMessage), so the
    # mapper holds its ``tool.call`` until the call runs — right before the
    # ``tool.result`` — because only then can it know no governance pipeline
    # emitted (and redacted) it (M5).
    "tool_roundtrip": (
        "run.status",
        "message.completed",
        "tool.call",
        "tool.result",
        "run.status",
    ),
    # ApprovalAgent: chat node streams and completes, then require_approval
    # interrupts; the run ends awaiting_approval (never completed).
    "approval_gate": (
        "run.status",
        "token.delta",
        "message.completed",
        "interrupt.requested",
        "run.status",
    ),
    # RawInterruptAgent: interrupt(value) with a non-dict value (§4.5).
    "raw_interrupt": (
        "run.status",
        "interrupt.requested",
        "run.status",
    ),
    # BoomyAgent: nothing streams before the node raises; the failed status
    # carries the §3.3 error block, then the exception re-raises to the caller.
    "failure": (
        "run.status",
        "run.status",
    ),
    # ThinkingAgent with expose_thinking=True: reasoning_content routes as
    # thinking (D18/A20 chain) — its delta flushed before the completion, like
    # token deltas.
    "thinking_exposed": (
        "run.status",
        "thinking.delta",
        "message.completed",
        "run.status",
    ),
    # The same graph with expose_thinking=False: no thinking.delta and no
    # ``thinking`` field on message.completed (§3.3 suppression).
    "thinking_suppressed": (
        "run.status",
        "message.completed",
        "run.status",
    ),
}

#: §3.3 payload table — the keys every event of a type must carry. Optional
#: (``?``) keys are absent when unset: emitters omit ``None`` optionals rather
#: than serializing ``null`` (``events._Payload.event_payload``).
PAYLOAD_REQUIRED_KEYS: Final[dict[str, frozenset[str]]] = {
    "run.status": frozenset({"status", "attempt"}),
    "token.delta": frozenset({"message_id", "text"}),
    "thinking.delta": frozenset({"message_id", "text"}),
    "message.completed": frozenset({"message_id", "role", "text", "citation_ids"}),
    "tool.call": frozenset({"tool_call_id", "name", "source", "args"}),
    "tool.result": frozenset(
        {"tool_call_id", "status", "summary", "artifact_ids", "duration_ms"}
    ),
    "artifact.created": frozenset(
        {"artifact_id", "question_id", "session_id", "kind", "name", "metadata"}
    ),
    "citation": frozenset({"citation_id", "message_id", "source", "snippet"}),
    "interrupt.requested": frozenset({"interrupt_id", "kind", "actions", "payload"}),
    "progress": frozenset({"label"}),
    "log": frozenset({"level", "message"}),
}

#: §3.3 line 206: the citation snippet is capped at 300 characters.
MAX_SNIPPET_CHARS: Final[int] = 300

#: §2.7 step 5: summaries cap at 8,000 (the M2 ungoverned tool mapping
#: applies the same cap to ToolMessages — ``streaming._TOOL_SUMMARY_MAX_CHARS``).
MAX_TOOL_SUMMARY_CHARS: Final[int] = _TOOL_SUMMARY_MAX_CHARS

#: §2.4: untyped custom data falls back to a log whose repr is capped
#: (``streaming._LOG_FALLBACK_MAX_CHARS``).
MAX_LOG_FALLBACK_CHARS: Final[int] = _LOG_FALLBACK_MAX_CHARS


def type_sequence(events: Iterable[EventEnvelope]) -> list[str]:
    """The ordered event types of one run — the golden-sequence view."""
    return [envelope.type.value for envelope in events]


def single(events: Iterable[EventEnvelope], type_value: str) -> EventEnvelope:
    """The one envelope of ``type_value``; asserts exactly one exists."""
    found = [envelope for envelope in events if envelope.type.value == type_value]
    assert len(found) == 1, f"expected exactly one {type_value} event, got {len(found)}"
    return found[0]


def statuses(events: Iterable[EventEnvelope]) -> list[str]:
    """The ordered ``run.status`` status values of one run."""
    return [
        envelope.payload["status"]
        for envelope in events
        if envelope.type.value == "run.status"
    ]


def no_leak(events: Iterable[EventEnvelope], *secrets: str) -> None:
    """Assert none of ``secrets`` appears anywhere in the emitted payloads."""
    for envelope in events:
        blob = repr(envelope.payload)
        for secret in secrets:
            assert secret not in blob, f"secret {secret!r} leaked into {envelope.type.value}"


def assert_event_shapes(events: Iterable[EventEnvelope]) -> None:
    """Every event matches its §3.3 payload shape (the protocol-side check)."""
    for envelope in events:
        payload = envelope.payload
        # Envelope contract (§3.3): "v" first-class, identity fields set,
        # per-run seq from 1, UTC ts.
        assert envelope.v == PROTOCOL_VERSION
        assert envelope.run_id and envelope.thread_id and envelope.agent
        assert envelope.seq >= 1
        assert envelope.ts

        required = PAYLOAD_REQUIRED_KEYS.get(envelope.type.value)
        assert required is not None, f"no §3.3 shape recorded for {envelope.type.value}"
        missing = required - payload.keys()
        assert not missing, (
            f"{envelope.type.value}: missing §3.3 payload keys {sorted(missing)}"
        )

        kind = envelope.type.value
        if kind == "run.status":
            assert payload["status"] in ("running", "completed", "failed", "awaiting_approval")
            if payload["status"] == "failed":
                assert set(payload["error"]) == {"code", "message"}
                assert payload["error"]["code"] and payload["error"]["message"]
            else:
                assert "error" not in payload, "absent optionals are omitted"
        elif kind in ("token.delta", "thinking.delta"):
            assert payload["message_id"] and payload["text"]
        elif kind == "message.completed":
            assert payload["role"] and isinstance(payload["text"], str)
            assert all(isinstance(item, str) for item in payload["citation_ids"])
        elif kind == "tool.call":
            # M2 ungoverned mapping (§2.4): source is "custom", args redacted.
            assert payload["source"] == "custom"
            assert isinstance(payload["args"], dict)
        elif kind == "tool.result":
            assert payload["status"] in ("ok", "error")
            assert len(payload["summary"]) <= _TOOL_SUMMARY_MAX_CHARS
            assert payload["artifact_ids"] == [], "M2: governed artifacts land with M5"
            assert isinstance(payload["duration_ms"], int) and payload["duration_ms"] >= 0
        elif kind == "artifact.created":
            # §3.1/§3.4: question_id is the run, session_id the thread.
            assert payload["question_id"] == envelope.run_id
            assert payload["session_id"] == envelope.thread_id
            assert payload["artifact_id"] and payload["kind"] and payload["name"]
        elif kind == "citation":
            assert len(payload["snippet"]) <= MAX_SNIPPET_CHARS
            assert isinstance(payload["source"], dict) and "title" in payload["source"]
        elif kind == "interrupt.requested":
            assert payload["kind"]
            assert isinstance(payload["actions"], list) and payload["actions"]
            assert isinstance(payload["payload"], dict)
        elif kind == "progress":
            assert payload["label"]
            assert "pct" not in payload or isinstance(payload["pct"], (int, float))
        elif kind == "log":
            # The 500-char cap is the *untyped-fallback* repr's (§2.4), not a
            # §3.3 rule for node-emitted logs — pinned in the scenario test.
            assert payload["level"] and isinstance(payload["message"], str)


def assert_parity(runs: Mapping[str, Iterable[EventEnvelope]]) -> None:
    """A31: both drivers emit identical *normalized* sequences for one graph.

    Accepts any driver → events mapping (``StreamedRun`` maps collapsed to
    their ``events``, or the bare sink tuples a failed run leaves behind).
    """
    v2 = _kit.normalized(runs["v2"])
    v3 = _kit.normalized(runs["v3"])
    assert v2 == v3, "A31: the v2 and v3 drivers must emit identical protocol events"


async def golden_runs(
    agent: _kit.BaseAgent, *, context: _kit.VDreamerContext | None = None
) -> dict[str, _kit.StreamedRun]:
    """Run one scenario graph through both drivers, asserting the suite contract.

    Asserts parity (a), the §3.3 shape table on every event of both runs (b),
    and that no ``interrupt.resolved`` ever appears (resolution is the
    worker's, M3 — §4.3). The scenario tests then pin their golden type
    sequence and payload specifics.
    """
    runs = await _kit.run_graph_all_drivers(
        agent,
        _kit.input_for(agent),
        context=context if context is not None else _kit.make_context(agent),
    )
    assert_parity({driver: run.events for driver, run in runs.items()})
    for run in runs.values():
        assert_event_shapes(run.events)
        assert not [e for e in run.events if e.type.value == "interrupt.resolved"]
    return runs


async def failing_golden_runs(
    agent: _kit.BaseAgent,
) -> dict[str, tuple[EventEnvelope, ...]]:
    """The failure scenario: the sink survives the re-raised exception.

    ``_kit.run_graph_all_drivers`` loses the sink when the run raises, so
    each driver runs on its own runner here; the exception must still
    propagate exactly as the non-streaming path would (``arun(sink=...)``).
    """
    out: dict[str, tuple[EventEnvelope, ...]] = {}
    for driver in ("v2", "v3"):
        runner = AgentRunner(agent, checkpointer=_kit.InMemorySaver())
        sink = InMemoryEventSink()
        with (
            _kit.driver_env(driver),
            pytest.raises(ValueError, match="boom from streaming node"),
        ):
            await runner.arun(
                _kit.input_for(agent),
                config={"configurable": {"thread_id": f"t-{driver}"}},
                context=_kit.make_context(agent),
                sink=sink,
            )
        out[driver] = sink.events
        assert_event_shapes(sink.events)
    assert_parity(out)
    return out


class ThinkingAgent(_kit.BaseTestAgent):
    """A ``reasoning_content`` chunk — the §3.3 extraction chain's middle alias."""

    name = "thinker"

    def build(self, ctx: Any) -> Any:
        def chat(state: _kit.ChatState) -> dict[str, Any]:
            model = _kit._fake_model(
                _kit._text_chunk("", "m1", additional_kwargs={"reasoning_content": "step one"}),
                _kit._text_chunk("Answer", "m1"),
            )
            return {"messages": [model.invoke(state["messages"])]}

        graph = _kit.StateGraph(_kit.ChatState)
        graph.add_node("chat", chat)
        graph.add_edge(_kit.START, "chat")
        graph.add_edge("chat", _kit.END)
        return graph


class PasswordToolAgent(_kit.BaseTestAgent):
    """A tool whose args carry a ``password`` key — A33 default redaction."""

    name = "pwtool"

    def build(self, ctx: Any) -> Any:
        def call_tool(state: _kit.ChatState) -> dict[str, Any]:
            return {
                "messages": [
                    _kit.AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "login",
                                "args": {"user": "ops", "password": "s3cret"},
                                "id": "call_pw",
                            }
                        ],
                        id="m_pw",
                    )
                ]
            }

        def tool_result(state: _kit.ChatState) -> dict[str, Any]:
            return {
                "messages": [
                    _kit.ToolMessage(content="ok", tool_call_id="call_pw", id="tm_pw")
                ]
            }

        graph = _kit.StateGraph(_kit.ChatState)
        graph.add_node("call", call_tool)
        graph.add_node("tool", tool_result)
        graph.add_edge(_kit.START, "call")
        graph.add_edge("call", "tool")
        graph.add_edge("tool", _kit.END)
        return graph
