"""Milestone 11 exit: the conformance kit passes against the Python reference fold (§6.1).

Three layers:

1. **the kit folds** — every committed recording, replayed through
   ``reference_fold``, reaches exactly its ``expected_state``;
2. **the kit says what it claims** — per-scenario facts checked on the raw
   events, independent of the fold (so a fold bug cannot bless itself), and
   every recorded event validates against ``events.schema.json``;
3. **the kit is current** — re-recording against today's gateway folds to the
   committed states (cross-cutting rule 4: a protocol change re-records the
   kit in the same PR — ``uv run python tests/conformance/record_kit.py``).
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest
from reference_fold import Fold, ProtocolMismatch, fold_recording, parse_sse

from vdreamer.conformance import RECORDINGS, kit_files, load_recording
from vdreamer.protocol_schema import event_json_schema


def _events(recording: dict[str, Any], step: int | None = None) -> list[dict[str, Any]]:
    steps = [s for s in recording["steps"] if s["kind"] == "sse"]
    if step is not None:
        steps = [steps[step]]
    return [json.loads(event.data) for s in steps for event in parse_sse(s["body"])]


def _types(events: list[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


def test_the_kit_is_complete() -> None:
    files = kit_files()
    assert set(files) == {f"{name}.json" for name in RECORDINGS} | {"README.md"}
    for name in RECORDINGS:
        recording = load_recording(name)
        assert recording["name"] == name and recording["protocol_version"] == "1.0"
        assert recording["description"] and recording["steps"]
        assert "expected_state" in recording


@pytest.mark.parametrize("name", RECORDINGS)
def test_every_recording_folds_to_its_expected_state(name: str) -> None:
    recording = load_recording(name)
    assert fold_recording(recording) == recording["expected_state"]


@pytest.mark.parametrize("name", [n for n in RECORDINGS if n != "forward_compat"])
def test_recorded_events_match_the_event_schema(name: str) -> None:
    validator = jsonschema.Draft202012Validator(event_json_schema())
    events = _events(load_recording(name))
    assert events
    for event in events:
        validator.validate(event)
    sse_ids = re_ids(load_recording(name))
    assert all(sse_id.count(":") == 1 for sse_id in sse_ids)  # "{run_id}:{seq}" (A3)


def re_ids(recording: dict[str, Any]) -> list[str]:
    return [
        event.id or "" for step in recording["steps"] if step["kind"] == "sse"
        for event in parse_sse(step["body"])
    ]


# -- what each recording claims -----------------------------------------------------------


def test_chat() -> None:
    events = _events(load_recording("chat"))
    types = _types(events)
    assert types[0] == "run.status" and types[-1] == "run.status"
    assert events[-1]["payload"]["status"] == "completed"
    assert {"progress", "log", "thinking.delta", "message.completed"} <= set(types)
    completed = next(e for e in events if e["type"] == "message.completed")["payload"]
    assert completed["text"] == "Hello! How can I help you today?"
    assert completed["thinking"] == "A greeting: reply briefly."


def test_reconnect() -> None:
    recording = load_recording("reconnect")
    first, second, refresh = [s for s in recording["steps"] if s["kind"] == "sse"]
    assert first["dropped"] and second["last_event_id"] and refresh.get("reset")
    seqs_first = [e["seq"] for e in _events(recording, 0)]
    seqs_second = [e["seq"] for e in _events(recording, 1)]
    assert len(seqs_first) == 4 and seqs_second[0] == 3  # two duplicates to dedup
    assert seqs_first + [s for s in seqs_second if s > 4] == list(range(1, seqs_second[-1] + 1))
    assert refresh["path"].startswith("/api/v1/threads/")
    # Without dedup, the duplicates would double the progress/log entries.
    naive = Fold()
    naive.seen = _AlwaysNew()
    for step in recording["steps"][:2]:
        naive.feed_sse(step["body"])
    assert len(naive.result()["logs"]) == 2
    assert len(recording["expected_state"]["logs"]) == 1


class _AlwaysNew(set[tuple[str, int]]):
    def __contains__(self, item: object) -> bool:
        return False


def test_tool_overflow() -> None:
    events = _events(load_recording("tool_overflow"))
    call = next(e for e in events if e["type"] == "tool.call")["payload"]
    result = next(e for e in events if e["type"] == "tool.result")["payload"]
    assert call["args"] == {"query": "***", "token": "***"}  # redacted (§2.7)
    assert result["status"] == "ok" and result["artifact_ids"]
    created = {e["payload"]["artifact_id"] for e in events if e["type"] == "artifact.created"}
    assert set(result["artifact_ids"]) <= created


@pytest.mark.parametrize(
    ("name", "kind", "action"),
    [
        ("interrupt_require_approval", "sql_review", "approve"),
        ("interrupt_tool_review", "tool_review", "decisions"),
        ("interrupt_mcp_elicitation", "mcp_elicitation", "edit"),
        ("interrupt_generic", "generic", "respond"),
    ],
)
def test_all_four_interrupt_sources(name: str, kind: str, action: str) -> None:
    recording = load_recording(name)
    first, resolve, rest = recording["steps"]
    assert first["kind"] == "sse" and first["dropped"]
    assert resolve["kind"] == "resolve" and resolve["status"] == 200
    assert "expected_version" in resolve["body"]
    assert rest["last_event_id"] == re_ids({"steps": [first]})[-1]
    paused = _events(recording, 0)
    assert paused[-1]["payload"]["status"] == "awaiting_approval"
    requested = next(e for e in paused if e["type"] == "interrupt.requested")["payload"]
    assert requested["kind"] == kind
    resumed = _events(recording, 1)
    resolved = next(e for e in resumed if e["type"] == "interrupt.resolved")["payload"]
    assert resolved["action"] == action
    assert resumed[-1]["payload"]["status"] == "completed"
    assert resolve["path"] == f"/api/v1/interrupts/{requested['interrupt_id']}/resolve"


def test_artifacts() -> None:
    events = _events(load_recording("artifacts"))
    kinds = [e["payload"]["kind"] for e in events if e["type"] == "artifact.created"]
    assert sorted(kinds) == ["file", "html", "image", "json", "table"]
    renders = {a["kind"]: a["render"] for a in load_recording("artifacts")["expected_state"][
        "artifacts"
    ]}
    assert renders == {k: k for k in kinds}  # verdict.v1 is a renderer the reference knows


def test_citations() -> None:
    recording = load_recording("citations")
    (message,) = recording["expected_state"]["messages"]
    assert [c["source"].get("artifact_id") or c["source"].get("uri") for c in message[
        "citations"
    ]] == [recording["expected_state"]["artifacts"][0]["artifact_id"], "kb://regions/emea"]
    assert message["citations"][0]["locator"] == {"page": 4}


def test_forward_compat() -> None:
    recording = load_recording("forward_compat")
    fold = Fold()
    fold.feed_sse(recording["steps"][0]["body"])
    assert fold.ignored == ["telemetry.sample"]
    state = fold.result()
    assert [a["render"] for a in state["artifacts"]] == ["download", "download"]
    assert state["progress"] == {"label": "rendering", "pct": 50.0}
    assert state["status"] == "completed"


def test_another_major_is_refused() -> None:
    fold = Fold()
    envelope = {"v": "2.0", "type": "run.status", "run_id": "r", "thread_id": "t", "agent": "a",
                "seq": 1, "ts": "x", "ns": [], "payload": {"status": "running"}}
    with pytest.raises(ProtocolMismatch):
        fold.feed(envelope)


def test_resumed_nodes_upsert_by_id() -> None:
    state = load_recording("interrupt_tool_review")["expected_state"]
    assert len(state["toolCalls"]) == 1 and state["toolCalls"][0]["status"] == "ok"
    events = _events(load_recording("interrupt_tool_review"))
    assert _types(events).count("tool.call") == 2  # emitted again on resume, same id


# -- the kit is current ---------------------------------------------------------------------


def test_a_fresh_recording_folds_to_the_committed_states() -> None:
    from record_kit import record_all

    fresh = record_all()
    for name in RECORDINGS:
        assert fresh[name]["expected_state"] == load_recording(name)["expected_state"], name


def test_export_schema_ships_the_kit(tmp_path: Any) -> None:
    from vdreamer.export_schema import build_export

    files = build_export({})
    for name in RECORDINGS:
        assert f"conformance/{name}.json" in files
    assert b"## The fold (normative)" in files["conformance/README.md"]
    assert b"\r\n" not in files["conformance/README.md"]
