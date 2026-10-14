"""The v2 driver end to end on small StateGraph agents (PRD §2.4, §3.3, §4).

No live Postgres anywhere: every run goes through ``AgentRunner(sink=...)``
on ``InMemorySaver``. The same assertions the v2 driver must satisfy are
mirrored by the driver-parity test (WP3's golden suite builds on the same
kit); this file pins the concrete protocol payloads per event type.
"""

from __future__ import annotations

from typing import Any

import _kit
import pytest

from vdreamer.eventlog import InMemoryEventSink
from vdreamer.events import PROTOCOL_VERSION, EventType
from vdreamer.runtime import AgentRunner


def _events_by_type(events: Any) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for envelope in events:
        out.setdefault(envelope.type.value, []).append(envelope)
    return out


async def test_chatty_run_emits_the_full_event_sequence() -> None:
    agent = _kit.ChattyAgent()
    run = await _kit.run_streamed_ctx(agent, "v2")

    types = [envelope.type.value for envelope in run.events]
    # §3.3 sequence: running → custom records → message.completed → (flushed)
    # deltas → terminal status. Deltas batch at emission (50 ms / 32 tokens)
    # and this run finishes faster than both, so the batcher flushes once at
    # stream end, before the terminal run.status.
    assert types[0] == "run.status"
    assert types[-1] == "run.status"
    assert run.events[0].payload == {"status": "running", "attempt": 0}
    assert run.events[-1].payload == {"status": "completed", "attempt": 0}
    assert "progress" in types
    assert "log" in types
    assert "artifact.created" in types
    assert "citation" in types
    assert "message.completed" in types
    assert "token.delta" in types


async def test_token_deltas_are_batched_per_message() -> None:
    agent = _kit.ChattyAgent()
    run = await _kit.run_streamed_ctx(agent, "v2")
    deltas = [e for e in run.events if e.type == EventType.TOKEN_DELTA]
    # All chunks of one message coalesce into one flushed batch (§3.3).
    assert len(deltas) == 1
    assert deltas[0].payload["message_id"] == "m1"
    assert deltas[0].payload["text"] == "Hel"
    assert deltas[0].ns == ()


async def test_message_completed_payload_shape() -> None:
    agent = _kit.ChattyAgent()
    run = await _kit.run_streamed_ctx(agent, "v2")
    completed = _events_by_type(run.events)["message.completed"]
    assert len(completed) == 1
    payload = completed[0].payload
    assert payload["message_id"] == "m1"
    assert payload["role"] == "ai"
    assert payload["text"] == "Hel"
    assert payload["citation_ids"], "the citation bound to the answer binds to its message"
    assert payload.get("usage") is None


async def test_citation_payload_shape() -> None:
    agent = _kit.ChattyAgent()
    ctx = _kit.make_context(agent)
    run = await _kit.run_streamed(agent, _kit.input_for(agent), driver="v2", context=ctx)
    citation = _events_by_type(run.events)["citation"][0]
    assert citation.payload["message_id"] == "m1"
    assert citation.payload["source"] == {"uri": "doc://a", "title": "Doc A"}
    assert citation.payload["locator"] == {"page": 3}
    assert citation.payload["snippet"] == "quote"


async def test_artifact_created_payload_shape() -> None:
    agent = _kit.ChattyAgent()
    ctx = _kit.make_context(agent)
    run = await _kit.run_streamed(agent, _kit.input_for(agent), driver="v2", context=ctx)
    artifact = _events_by_type(run.events)["artifact.created"][0]
    payload = artifact.payload
    # §3.4: question_id is the run_id, session_id the thread_id.
    assert payload["question_id"] == ctx.run_id
    assert payload["session_id"] == ctx.thread_id
    assert payload["kind"] == "json"
    assert payload["renderer"] == "verdict.v1"
    assert payload["name"] == "verdict"
    assert payload["artifact_id"].startswith("art_")
    # M6: the encoded preview rides the event; URLs only once a store kept the bytes.
    assert payload["preview"] == {"body": {"ok": True}}
    assert "view_url" not in payload


async def test_untyped_custom_data_becomes_a_truncated_log() -> None:
    agent = _kit.ChattyAgent()
    run = await _kit.run_streamed_ctx(agent, "v2")
    logs = [
        e.payload
        for e in run.events
        if e.type == EventType.LOG and "untyped" in str(e.payload.get("message", ""))
    ]
    assert logs, "arbitrary custom data maps to a log event (§2.4)"
    assert logs[0]["level"] == "info"
    assert "untyped" in logs[0]["message"]


async def test_tool_call_result_with_default_redaction() -> None:
    agent = _kit.ToolAgent()
    run = await _kit.run_streamed_ctx(agent, "v2")

    calls = _events_by_type(run.events)["tool.call"]
    assert len(calls) == 1
    call = calls[0].payload
    # A33: policy keys plus defaults (password/token/secret/api_key/authorization).
    assert call["tool_call_id"] == "call_1"
    assert call["name"] == "run_query"
    assert call["source"] == "custom"
    assert call["args"]["query"] == "SELECT 1"
    assert call["args"]["api_key"] == "***"

    results = _events_by_type(run.events)["tool.result"]
    assert len(results) == 1
    result = results[0].payload
    assert result["tool_call_id"] == "call_1"
    assert result["status"] == "ok"
    assert result["summary"] == "rows: 1"
    assert result["artifact_ids"] == []
    # M5 seam: governed duration lands with the governance pipeline.
    assert result["duration_ms"] == 0


async def test_tool_result_error_status() -> None:
    from langchain_core.messages import ToolMessage

    from vdreamer.agents import BuildContext

    class _FailingToolAgent(_kit.BaseTestAgent):
        name = "failtool"

        def build(self, ctx: BuildContext) -> Any:
            def tool(state: _kit.ChatState) -> dict[str, Any]:
                return {
                    "messages": [
                        ToolMessage(
                            content="boom: bad sql",
                            tool_call_id="call_9",
                            status="error",
                            id="tm_9",
                        )
                    ]
                }

            graph = _kit.StateGraph(_kit.ChatState)
            graph.add_node("tool", tool)
            graph.add_node("edge_end", lambda state: {"flag": "done"})  # type: ignore[arg-type]
            graph.add_edge(_kit.START, "tool")
            graph.add_edge("tool", "edge_end")
            graph.add_edge("edge_end", _kit.END)
            return graph

    agent = _FailingToolAgent()
    run = await _kit.run_streamed_ctx(agent, "v2")
    result = _events_by_type(run.events)["tool.result"][0].payload
    assert result["status"] == "error"
    assert result["error"] == {"code": "tool_error", "message": "boom: bad sql"}


async def test_subgraph_namespace_is_preserved() -> None:
    agent = _kit.SubgraphAgent()
    run = await _kit.run_streamed_ctx(agent, "v2")
    namespaced = [
        e
        for e in run.events
        if e.type in (EventType.PROGRESS, EventType.MESSAGE_COMPLETED) and e.ns
    ]
    assert namespaced, "events emitted inside a subgraph carry a non-root ns"
    for envelope in namespaced:
        assert envelope.ns[0].startswith("child:")
    # Root-level events (status, deltas) stay at ns=[].
    assert all(e.ns == () for e in run.events if e.type == EventType.TOKEN_DELTA)
    assert run.events[0].ns == ()


async def test_pending_interrupts_are_read_after_the_stream() -> None:
    """§2.4: v2 does not subscribe to ``values``, so interrupts come from
    ``aget_state`` after the stream ends."""
    agent = _kit.ApprovalAgent()
    run = await _kit.run_streamed_ctx(agent, "v2")

    requested = _events_by_type(run.events)["interrupt.requested"]
    assert len(requested) == 1
    payload = requested[0].payload
    assert payload["kind"] == "sql_review"
    assert payload["scope"] == "db:write"
    assert payload["approver_role"] == "dba"
    assert payload["actions"] == ["approve", "reject", "edit"]
    assert payload["payload"] == {"sql": "UPDATE t SET x = 1"}
    assert "edit_schema" not in payload, "absent optionals are omitted"
    assert payload["expires_at"]
    assert payload["interrupt_id"]

    assert run.events[-1].payload["status"] == "awaiting_approval"


async def test_raw_interrupt_normalizes_to_generic_respond() -> None:
    agent = _kit.RawInterruptAgent()
    run = await _kit.run_streamed_ctx(agent, "v2")
    payload = _events_by_type(run.events)["interrupt.requested"][0].payload
    # §4.5: raw ``interrupt(value)`` → kind "generic", actions ["respond"],
    # the resolver payload rides verbatim.
    assert payload["kind"] == "generic"
    assert payload["actions"] == ["respond"]
    assert payload["payload"] == {"value": "just answer this"}
    assert run.events[-1].payload["status"] == "awaiting_approval"


async def test_raw_dict_interrupt_normalizes_to_generic() -> None:
    from langgraph.types import interrupt

    from vdreamer.agents import BuildContext

    class _DictInterruptAgent(_kit.BaseTestAgent):
        name = "dictgate"

        def build(self, ctx: BuildContext) -> Any:
            def gate(state: _kit.ChatState) -> dict[str, Any]:
                return {"flag": str(interrupt({"question": "proceed?"}))}

            graph = _kit.StateGraph(_kit.ChatState)
            graph.add_node("gate", gate)
            graph.add_edge(_kit.START, "gate")
            graph.add_edge("gate", _kit.END)
            return graph

    agent = _DictInterruptAgent()
    run = await _kit.run_streamed_ctx(agent, "v2")
    payload = _events_by_type(run.events)["interrupt.requested"][0].payload
    assert payload["kind"] == "generic"
    assert payload["actions"] == ["respond"]
    assert payload["payload"] == {"question": "proceed?"}


async def test_failed_run_emits_failed_status_and_reraises() -> None:
    agent = _kit.BoomyAgent()
    with pytest.raises(ValueError, match="boom from streaming node"):
        await _kit.run_streamed_ctx(agent, "v2")
    # run_streamed loses the sink on raise — rerun manually to inspect events.
    runner = AgentRunner(agent, checkpointer=_kit.InMemorySaver())
    sink = InMemoryEventSink()
    with pytest.raises(ValueError):
        await runner.arun(
            _kit.input_for(agent),
            config={"configurable": {"thread_id": "t-fail"}},
            context=_kit.make_context(agent),
            sink=sink,
        )
    statuses = [e.payload for e in sink.events if e.type == EventType.RUN_STATUS]
    assert statuses == [
        {"status": "running", "attempt": 0},
        {
            "status": "failed",
            "attempt": 0,
            "error": {"code": "node_error", "message": "boom from streaming node"},
        },
    ]


async def test_attempt_rides_the_run_status_payload() -> None:
    agent = _kit.ChattyAgent()
    ctx = _kit.make_context(agent).model_copy(update={"attempt": 3})
    run = await _kit.run_streamed(agent, _kit.input_for(agent), driver="v2", context=ctx)
    assert run.events[0].payload == {"status": "running", "attempt": 3}
    assert run.events[-1].payload["attempt"] == 3


async def test_thinking_extraction_reasoning_content_and_suppression() -> None:
    """The vLLM ``reasoning_content`` alias routes as thinking; agents with
    ``expose_thinking = False`` suppress it entirely (§3.3)."""

    class _ReasoningAgent(_kit.BaseTestAgent):
        name = "reasoner"
        expose_thinking = False

        def build(self, ctx: Any) -> Any:
            def chat(state: _kit.ChatState) -> dict[str, Any]:

                model = _kit._fake_model(
                    _kit._text_chunk(
                        "", "m1", additional_kwargs={"reasoning_content": "hidden thought"}
                    )
                )
                return {"messages": [model.invoke(state["messages"])]}

            graph = _kit.StateGraph(_kit.ChatState)
            graph.add_node("chat", chat)
            graph.add_edge(_kit.START, "chat")
            graph.add_edge("chat", _kit.END)
            return graph

    agent = _ReasoningAgent()
    run = await _kit.run_streamed_ctx(agent, "v2")
    assert [e.type.value for e in run.events if e.type == EventType.THINKING_DELTA] == []
    completed = _events_by_type(run.events)["message.completed"][0].payload
    assert "thinking" not in completed  # suppressed: no thinking key at all

    # Same agent exposing thinking → thinking.delta flows and completed carries it.
    agent_exposing = _ReasoningAgent()
    agent_exposing.expose_thinking = True
    run2 = await _kit.run_streamed_ctx(agent_exposing, "v2")
    deltas = _events_by_type(run2.events)["thinking.delta"]
    assert deltas and deltas[0].payload["text"] == "hidden thought"
    completed2 = _events_by_type(run2.events)["message.completed"][0].payload
    assert completed2["thinking"] == "hidden thought"


async def test_envelope_protocol_fields() -> None:
    agent = _kit.ChattyAgent()
    ctx = _kit.make_context(agent)
    run = await _kit.run_streamed(agent, _kit.input_for(agent), driver="v2", context=ctx)
    for envelope in run.events:
        assert envelope.v == PROTOCOL_VERSION
        assert envelope.agent == agent.name
        assert envelope.run_id == ctx.run_id
        assert envelope.thread_id == ctx.thread_id
        assert envelope.ts
    seqs = [envelope.seq for envelope in run.events]
    assert seqs == list(range(1, len(seqs) + 1)), "per-run seq is monotonic from 1"
    assert run.events[0].sse_id == f"{ctx.run_id}:1"


def test_kit_normalize_is_importable_without_running() -> None:
    """The parity helper is pure — WP3 reuses it for the golden suite."""
    from vdreamer.events import EventEnvelope, EventType

    envelope = EventEnvelope(
        type=EventType.RUN_STATUS,
        run_id="r",
        thread_id="t",
        agent="a",
        seq=1,
        payload={
            "status": "running",
            "message_id": "lc_run--01a136ef-1ac9-70c3-9e99-85275ec189f9",
        },
    )
    rows = _kit.normalized([envelope])
    assert rows[0][0] == "run.status"
    assert rows[0][2]["message_id"].startswith("<id")
