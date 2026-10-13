"""Protocol v1.0 core tests: constants, EventType, EventEnvelope, payload models.

PRD §3.3 is the contract; these spot-check the envelope serialization and
every payload model's dump against the table there.
"""

from datetime import datetime, timedelta

import pydantic
import pytest

from vdreamer import (
    DELTA_BATCH_INTERVAL_S,
    DELTA_BATCH_MAX_TOKENS,
    HEARTBEAT_INTERVAL_S,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_HEADER,
    SSE_BUFFERING_HEADER,
    SSE_BUFFERING_VALUE,
    ArtifactCreatedPayload,
    CitationPayload,
    CitationSource,
    ErrorInfo,
    EventEnvelope,
    EventType,
    InterruptRequestedPayload,
    InterruptResolvedPayload,
    LogPayload,
    MessageCompletedPayload,
    ProgressPayload,
    RunStatusPayload,
    TokenDeltaPayload,
    ToolCallPayload,
    ToolResultPayload,
)

#: The twelve protocol event types, exactly as PRD §3.3 / tasks.md line 92 spell them.
EXPECTED_EVENT_TYPES: dict[str, EventType] = {
    "run.status": EventType.RUN_STATUS,
    "token.delta": EventType.TOKEN_DELTA,
    "thinking.delta": EventType.THINKING_DELTA,
    "message.completed": EventType.MESSAGE_COMPLETED,
    "tool.call": EventType.TOOL_CALL,
    "tool.result": EventType.TOOL_RESULT,
    "artifact.created": EventType.ARTIFACT_CREATED,
    "citation": EventType.CITATION,
    "interrupt.requested": EventType.INTERRUPT_REQUESTED,
    "interrupt.resolved": EventType.INTERRUPT_RESOLVED,
    "progress": EventType.PROGRESS,
    "log": EventType.LOG,
}


def test_protocol_constants() -> None:
    assert PROTOCOL_VERSION == "1.0"
    assert PROTOCOL_VERSION_HEADER == "X-Protocol-Version"
    # A22: heartbeat + no proxy buffering; §3.3 line 211: delta batching.
    assert HEARTBEAT_INTERVAL_S == 15.0
    assert SSE_BUFFERING_HEADER == "X-Accel-Buffering"
    assert SSE_BUFFERING_VALUE == "no"
    assert DELTA_BATCH_INTERVAL_S == 0.05
    assert DELTA_BATCH_MAX_TOKENS == 32


def test_event_type_values() -> None:
    assert len(EventType) == 12
    assert {member.value for member in EventType} == set(EXPECTED_EVENT_TYPES)
    for value, member in EXPECTED_EVENT_TYPES.items():
        assert member.value == value
        # StrEnum: the member *is* its protocol string.
        assert str(member) == value


def make_envelope(**overrides: object) -> EventEnvelope:
    """A minimal valid envelope, with per-test overrides."""
    fields: dict[str, object] = {
        "type": EventType.TOOL_CALL,
        "run_id": "r_1",
        "thread_id": "t_1",
        "agent": "sql-analyst",
        "seq": 42,
        "payload": {"tool_call_id": "tc_1"},
    }
    fields.update(overrides)
    return EventEnvelope(**fields)  # type: ignore[arg-type]


def test_envelope_dump_key_order_and_defaults() -> None:
    envelope = make_envelope()
    dump = envelope.model_dump()
    # The JSON object starts with the protocol version field "v" (PRD §3.3).
    assert list(dump.keys())[0] == "v"
    assert dump["v"] == "1.0"
    assert list(dump.keys()) == [
        "v",
        "type",
        "run_id",
        "thread_id",
        "agent",
        "seq",
        "ts",
        "ns",
        "payload",
    ]


def test_envelope_ns_serializes_to_list() -> None:
    root = make_envelope()
    assert root.ns == ()
    assert root.model_dump(mode="json")["ns"] == []
    nested = make_envelope(ns=("child:abc", "grand:xyz"))
    assert nested.model_dump(mode="json")["ns"] == ["child:abc", "grand:xyz"]


def test_envelope_ts_is_utc_iso8601() -> None:
    envelope = make_envelope()
    parsed = datetime.fromisoformat(envelope.ts)
    assert parsed.utcoffset() == timedelta(0)
    # An explicit ts is carried verbatim (drivers set their own).
    explicit = make_envelope(ts="2026-10-13T12:00:00+00:00")
    assert explicit.ts == "2026-10-13T12:00:00+00:00"


def test_envelope_sse_id() -> None:
    assert make_envelope().sse_id == "r_1:42"


def test_envelope_round_trip() -> None:
    envelope = make_envelope(ns=("child:abc",))
    dumped = envelope.model_dump(mode="json")
    restored = EventEnvelope.model_validate(dumped)
    assert restored == envelope
    # v defaults to the protocol version even when omitted from input.
    stripped = dict(dumped)
    del stripped["v"]
    assert EventEnvelope.model_validate(stripped).v == PROTOCOL_VERSION


def test_run_status_payload() -> None:
    running = RunStatusPayload(status="running", attempt=0)
    assert running.event_payload() == {"status": "running", "attempt": 0}
    failed = RunStatusPayload(
        status="failed",
        attempt=2,
        error={"code": "node_error", "message": "boom"},
    )
    assert failed.event_payload() == {
        "status": "failed",
        "attempt": 2,
        "error": {"code": "node_error", "message": "boom"},
    }
    assert isinstance(failed.error, ErrorInfo)


def test_token_delta_payload() -> None:
    delta = TokenDeltaPayload(message_id="m_1", text="hello")
    # thinking.delta rides the same shape (PRD §3.3).
    thinking = TokenDeltaPayload(message_id="m_1", text="hmm")
    assert delta.event_payload() == {"message_id": "m_1", "text": "hello"}
    assert thinking.event_payload() == {"message_id": "m_1", "text": "hmm"}


def test_message_completed_payload() -> None:
    done = MessageCompletedPayload(message_id="m_1", role="assistant", text="hi")
    assert done.event_payload() == {
        "message_id": "m_1",
        "role": "assistant",
        "text": "hi",
        "citation_ids": [],
    }
    full = MessageCompletedPayload(
        message_id="m_1",
        role="assistant",
        text="hi",
        thinking="because",
        citation_ids=["cit_1"],
        usage={"input_tokens": 3, "output_tokens": 2},
    )
    assert full.event_payload() == {
        "message_id": "m_1",
        "role": "assistant",
        "text": "hi",
        "thinking": "because",
        "citation_ids": ["cit_1"],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


def test_tool_call_payload() -> None:
    call = ToolCallPayload(
        tool_call_id="tc_1",
        name="sql_query",
        source="custom",
        args={"query": "SELECT 1"},  # redacted upstream (A33), not here
    )
    assert call.event_payload() == {
        "tool_call_id": "tc_1",
        "name": "sql_query",
        "source": "custom",
        "args": {"query": "SELECT 1"},
    }


def test_tool_result_payload() -> None:
    ok = ToolResultPayload(
        tool_call_id="tc_1", status="ok", summary="2 rows", duration_ms=12
    )
    assert ok.event_payload() == {
        "tool_call_id": "tc_1",
        "status": "ok",
        "summary": "2 rows",
        "artifact_ids": [],
        "duration_ms": 12,
    }
    errored = ToolResultPayload(
        tool_call_id="tc_1",
        status="error",
        error={"code": "tool_error", "message": "nope"},
        summary="",
        duration_ms=5,
    )
    assert errored.event_payload()["error"] == {"code": "tool_error", "message": "nope"}
    with pytest.raises(pydantic.ValidationError):
        ToolResultPayload(tool_call_id="tc_1", status="bogus", summary="", duration_ms=1)


def test_citation_payload() -> None:
    cited = CitationPayload(
        citation_id="cit_1",
        message_id="m_1",
        source={"artifact_id": "art_1", "title": "Report"},
        locator={"page": 3},
        snippet="quote",
        score=0.9,
    )
    assert cited.event_payload() == {
        "citation_id": "cit_1",
        "message_id": "m_1",
        "source": {"artifact_id": "art_1", "title": "Report"},
        "locator": {"page": 3},
        "snippet": "quote",
        "score": 0.9,
    }
    assert isinstance(cited.source, CitationSource)
    # locator is optional (PRD `locator?`).
    bare = CitationPayload(
        citation_id="cit_2", message_id="m_1", source={"uri": "doc:1", "title": "D"},
        snippet="",
    )
    assert "locator" not in bare.event_payload()
    assert "uri" in bare.event_payload()["source"]


def test_progress_and_log_payloads() -> None:
    assert ProgressPayload(label="step 1/2", pct=0.5).event_payload() == {
        "label": "step 1/2",
        "pct": 0.5,
    }
    assert ProgressPayload(label="working").event_payload() == {"label": "working"}
    assert LogPayload(level="info", message="hi").event_payload() == {
        "level": "info",
        "message": "hi",
    }


def test_interrupt_requested_payload() -> None:
    requested = InterruptRequestedPayload(
        interrupt_id="int_1",
        kind="sql_review",
        actions=["approve", "reject"],
        payload={"sql": "UPDATE t SET x = 1"},
    )
    assert requested.event_payload() == {
        "interrupt_id": "int_1",
        "kind": "sql_review",
        "actions": ["approve", "reject"],
        "payload": {"sql": "UPDATE t SET x = 1"},
    }
    # scope/approver_role/edit_schema/expires_at are §4.2 optionals.
    with_scope = InterruptRequestedPayload(
        interrupt_id="int_1",
        kind="sql_review",
        scope="db:write:prod",
        approver_role="dba",
        actions=["approve"],
        payload={},
        edit_schema={"type": "object"},
        expires_at="2026-10-14T00:00:00+00:00",
    )
    assert with_scope.event_payload()["scope"] == "db:write:prod"
    assert with_scope.event_payload()["approver_role"] == "dba"
    assert with_scope.event_payload()["edit_schema"] == {"type": "object"}
    assert with_scope.event_payload()["expires_at"] == "2026-10-14T00:00:00+00:00"


def test_interrupt_resolved_payload() -> None:
    resolved = InterruptResolvedPayload(
        interrupt_id="int_1", action="approve", resolved_by="user_1"
    )
    assert resolved.event_payload() == {
        "interrupt_id": "int_1",
        "action": "approve",
        "resolved_by": "user_1",
    }


def test_artifact_created_payload() -> None:
    created = ArtifactCreatedPayload(
        artifact_id="art_1",
        question_id="r_1",
        session_id="t_1",
        kind="json",
        name="verdict",
        mime_type="application/json",
        size_bytes=64,
        metadata={"rows": 2},
    )
    assert created.event_payload() == {
        "artifact_id": "art_1",
        "question_id": "r_1",
        "session_id": "t_1",
        "kind": "json",
        "name": "verdict",
        "mime_type": "application/json",
        "size_bytes": 64,
        "metadata": {"rows": 2},
    }
    # PRD line 234 order.
    assert list(ArtifactCreatedPayload.model_fields.keys()) == [
        "artifact_id",
        "question_id",
        "session_id",
        "kind",
        "renderer",
        "name",
        "mime_type",
        "size_bytes",
        "preview",
        "view_url",
        "download_url",
        "metadata",
    ]
    # Until M4/M6 the URL/preview placeholders are None-tolerant and dropped.
    minimal = ArtifactCreatedPayload(
        artifact_id="art_2", question_id="r_2", session_id="t_2", kind="file", name="f"
    )
    dump = minimal.event_payload()
    assert dump["kind"] == "file"
    assert "preview" not in dump
    assert "view_url" not in dump
    assert "download_url" not in dump
