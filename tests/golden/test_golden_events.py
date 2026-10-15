"""The golden-event suite (WP3; PRD §2.4 A31/A43, §3.3, §4.2/§4.5).

For each scenario the same kit graph runs through both drivers; the shared
runner (:mod:`_golden`) asserts driver parity and applies the §3.3 payload
shape table to every event. Each test then asserts its golden expectation:
the ordered event types (:data:`_golden.GOLDEN_SEQUENCES`, recorded from the
drivers' actual output on langgraph 1.2.15) plus the scenario's specific
payload shapes from the §3.3 table.

The exit criterion (docs/tasks.md): the suite passes for the v2 driver; v3
runs as opt-in beta. Every scenario here currently passes with *full* v2↔v3
parity — no xfail anywhere.
"""

from __future__ import annotations

import _golden
import _kit
import pytest

pytestmark = pytest.mark.filterwarnings("ignore:.*v3 streaming protocol.*")


async def test_golden_chat_with_custom_records() -> None:
    """ChattyAgent: the §2.4 "Both" mapping, all four custom record kinds.

    Golden order: ``run.status(running) → progress → log → artifact.created →
    citation → log(untyped fallback) → token.delta (the message's open batch,
    flushed ahead of its completion) → message.completed → run.status(completed)``.
    """
    agent = _kit.ChattyAgent()
    runs = await _golden.golden_runs(agent)
    for driver in ("v2", "v3"):
        events = runs[driver].events
        assert _golden.type_sequence(events) == list(_golden.GOLDEN_SEQUENCES["chat_custom"])

        assert events[0].payload == {"status": "running", "attempt": 0}
        assert events[-1].payload == {"status": "completed", "attempt": 0}

        # progress {label, pct?} (§3.3).
        assert _golden.single(events, "progress").payload == {"label": "parsing", "pct": 10.0}

        # log {level, message} (§3.3) — the node's explicit emit_log.
        node_log = [
            e
            for e in events
            if e.type.value == "log" and e.payload["message"] == "hello from the node"
        ]
        assert len(node_log) == 1
        assert node_log[0].payload == {"level": "info", "message": "hello from the node"}

        # artifact.created (§3.4): identity wiring + the M6 encoded description;
        # no store is bound in the golden harness, so no URLs.
        artifact = _golden.single(events, "artifact.created")
        assert artifact.payload["artifact_id"].startswith("art_")
        assert artifact.payload["question_id"] == artifact.run_id
        assert artifact.payload["session_id"] == artifact.thread_id
        assert artifact.payload["kind"] == "json"
        assert artifact.payload["name"] == "verdict"
        assert artifact.payload["renderer"] == "verdict.v1"
        assert artifact.payload["metadata"] == {}
        assert artifact.payload["mime_type"] == "application/json"
        assert artifact.payload["size_bytes"] == 12
        assert artifact.payload["preview"] == {"body": {"ok": True}}
        for absent in ("view_url", "download_url"):
            assert absent not in artifact.payload, f"nothing stored: no {absent}"

        # citation {citation_id, message_id, source{...}, locator?, snippet
        # ≤300, score?} (§3.3) — bound to the message it cites, no score.
        citation = _golden.single(events, "citation")
        assert citation.payload["citation_id"].startswith("cit_")
        assert citation.payload["source"] == {"uri": "doc://a", "title": "Doc A"}
        assert citation.payload["locator"] == {"page": 3}
        assert citation.payload["snippet"] == "quote"
        assert len(citation.payload["snippet"]) <= _golden.MAX_SNIPPET_CHARS
        assert "score" not in citation.payload

        # message.completed (§3.3): the citation binds to the same message.
        completed = _golden.single(events, "message.completed")
        assert completed.payload["role"] == "ai"
        assert completed.payload["text"] == "Hel"
        assert completed.payload["citation_ids"] == [citation.payload["citation_id"]]
        assert "thinking" not in completed.payload and "usage" not in completed.payload

        # token.delta (§3.3): the single flushed batch for the one message.
        delta = _golden.single(events, "token.delta")
        assert delta.payload["message_id"] == completed.payload["message_id"]
        assert delta.payload["text"] == "Hel"

        # The untyped custom data fell back to a log, repr capped at 500 (§2.4).
        fallback = [
            e
            for e in events
            if e.type.value == "log" and "untyped" in str(e.payload.get("message", ""))
        ]
        assert len(fallback) == 1
        assert fallback[0].payload["level"] == "info"
        assert len(fallback[0].payload["message"]) <= _golden.MAX_LOG_FALLBACK_CHARS


async def test_golden_tool_roundtrip_and_redaction() -> None:
    """ToolAgent: ``tool.call`` (redacted args) → ``tool.result``; plus a
    ``password``-key redaction case (A33 defaults)."""
    agent = _kit.ToolAgent()
    runs = await _golden.golden_runs(agent)
    for driver in ("v2", "v3"):
        events = runs[driver].events
        assert _golden.type_sequence(events) == list(_golden.GOLDEN_SEQUENCES["tool_roundtrip"])

        call = _golden.single(events, "tool.call")
        assert call.payload == {
            "tool_call_id": "call_1",
            "name": "run_query",
            "source": "custom",
            "args": {"query": "SELECT 1", "api_key": "***"},
        }
        result = _golden.single(events, "tool.result")
        assert result.payload == {
            "tool_call_id": "call_1",
            "status": "ok",
            "summary": "rows: 1",
            "artifact_ids": [],
            "duration_ms": 0,  # governed timing arrives with M5 (§2.7)
        }
        assert len(result.payload["summary"]) <= _golden.MAX_TOOL_SUMMARY_CHARS
        _golden.no_leak(events, "hunter2")

    # Redaction case: `password` is one of the A33 default redaction keys.
    pw_agent = _golden.PasswordToolAgent()
    pw_runs = await _golden.golden_runs(pw_agent)
    for driver in ("v2", "v3"):
        events = pw_runs[driver].events
        assert _golden.type_sequence(events) == list(_golden.GOLDEN_SEQUENCES["tool_roundtrip"])
        call = _golden.single(events, "tool.call")
        assert call.payload["args"] == {"user": "ops", "password": "***"}
        _golden.no_leak(events, "s3cret")


async def test_golden_bare_args_governed_call_is_one_call() -> None:
    """A governed tool run with bare args binds to the model's call: one
    ``tool.call``/``tool.result`` pair, under the model's id, redacted."""
    runs = await _golden.golden_runs(_kit.BareArgsToolAgent())
    for driver in ("v2", "v3"):
        events = runs[driver].events
        assert _golden.type_sequence(events) == [
            "run.status",
            "message.completed",
            "tool.call",
            "tool.result",
            "run.status",
        ]
        call = next(e for e in events if e.type.value == "tool.call")
        result = next(e for e in events if e.type.value == "tool.result")
        assert call.payload["tool_call_id"] == "call_model_1"
        assert call.payload["args"] == {"city": "Cairo", "secret": "***"}
        assert result.payload["tool_call_id"] == "call_model_1"
        assert result.payload["status"] == "ok"


async def test_golden_approval_gate_pauses_before_completion() -> None:
    """ApprovalAgent (require_approval, InMemorySaver): the run pauses at
    ``awaiting_approval`` with the §4.2 record — and never completes."""
    agent = _kit.ApprovalAgent()
    runs = await _golden.golden_runs(agent)
    for driver in ("v2", "v3"):
        events = runs[driver].events
        assert _golden.type_sequence(events) == list(_golden.GOLDEN_SEQUENCES["approval_gate"])

        requested = _golden.single(events, "interrupt.requested")
        payload = requested.payload
        assert payload["kind"] == "sql_review"  # from the require_approval record
        assert payload["approver_role"] == "dba"
        assert payload["actions"] == ["approve", "reject", "edit"]
        assert payload["payload"] == {"sql": "UPDATE t SET x = 1"}
        assert payload["scope"] == "db:write"
        assert payload["interrupt_id"]
        assert payload["expires_at"]  # wall-clock TTL (§4.4); normalized() drops it
        assert "edit_schema" not in payload, "absent optionals are omitted"

        assert _golden.statuses(events) == [
            "running",
            "awaiting_approval",
        ], "NO completed event while the approval is pending"


async def test_golden_raw_interrupt_normalizes_to_generic() -> None:
    """RawInterruptAgent: raw ``interrupt(value)`` → kind ``generic``,
    actions ``["respond"]`` (§4.5)."""
    agent = _kit.RawInterruptAgent()
    runs = await _golden.golden_runs(agent)
    for driver in ("v2", "v3"):
        events = runs[driver].events
        assert _golden.type_sequence(events) == list(_golden.GOLDEN_SEQUENCES["raw_interrupt"])
        payload = _golden.single(events, "interrupt.requested").payload
        assert payload["kind"] == "generic"
        assert payload["actions"] == ["respond"]
        assert payload["payload"] == {"value": "just answer this"}
        assert _golden.statuses(events) == ["running", "awaiting_approval"]


async def test_golden_failure_carries_the_error_block() -> None:
    """BoomyAgent: ``run.status(failed)`` with ``error {code, message}``, both
    drivers identical, exception re-raised to the caller."""
    agent = _kit.BoomyAgent()
    runs = await _golden.failing_golden_runs(agent)
    for driver in ("v2", "v3"):
        events = runs[driver]
        assert _golden.type_sequence(events) == list(_golden.GOLDEN_SEQUENCES["failure"])
        assert [envelope.payload for envelope in events] == [
            {"status": "running", "attempt": 0},
            {
                "status": "failed",
                "attempt": 0,
                "error": {"code": "node_error", "message": "boom from streaming node"},
            },
        ]


async def test_golden_expose_thinking_toggle() -> None:
    """§3.3: the same graph twice — thinking present, then suppressed.

    The runner derives ``StreamSpec.expose_thinking`` from the agent (the
    production wiring), so the toggle goes through the same path.
    """
    exposed = _golden.ThinkingAgent()
    runs = await _golden.golden_runs(exposed)
    for driver in ("v2", "v3"):
        events = runs[driver].events
        assert _golden.type_sequence(events) == list(
            _golden.GOLDEN_SEQUENCES["thinking_exposed"]
        )
        completed = _golden.single(events, "message.completed")
        delta = _golden.single(events, "thinking.delta")
        assert delta.payload["message_id"] == completed.payload["message_id"]
        assert delta.payload["text"] == "step one"
        assert completed.payload["thinking"] == "step one"

    suppressed = _golden.ThinkingAgent()
    suppressed.expose_thinking = False
    runs_off = await _golden.golden_runs(suppressed)
    for driver in ("v2", "v3"):
        events = runs_off[driver].events
        assert _golden.type_sequence(events) == list(
            _golden.GOLDEN_SEQUENCES["thinking_suppressed"]
        )
        assert not [e for e in events if e.type.value == "thinking.delta"]
        completed = _golden.single(events, "message.completed")
        assert "thinking" not in completed.payload, "suppression covers the field too"
