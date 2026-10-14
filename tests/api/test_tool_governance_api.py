"""Milestone 5 through the REST surface: a governed tool run and an MCP elicitation.

Runs on both store backends (the SQLite one is what ``vdreamer dev`` uses:
artifact rows in the ``.db``, bytes on the filesystem).
"""

from __future__ import annotations

from typing import Any

from _api import (
    BIG_RESULT,
    assert_error,
    events_of,
    headers,
    new_thread,
    read_sse,
    start_run,
    wait_for_status,
)
from fastapi.testclient import TestClient


def test_governed_tool_run_with_an_overflow_artifact(live: TestClient) -> None:
    thread_id = new_thread(live, agent="tooling")
    run_id = start_run(
        live, thread_id, {"input": {"messages": [{"role": "user", "content": "report"}]}}
    )
    wait_for_status(live, run_id, "completed")

    _, frames = read_sse(live, f"/api/v1/runs/{run_id}/events")
    events = events_of(frames)
    types = [event["type"] for event in events]
    assert types.count("tool.call") == 1 and types.count("tool.result") == 1
    assert types.index("tool.call") < types.index("artifact.created") < types.index("tool.result")
    call = next(e["payload"] for e in events if e["type"] == "tool.call")
    assert call["args"] == {"query": "***", "token": "***"}
    assert call["source"] == "custom"
    result = next(e["payload"] for e in events if e["type"] == "tool.result")
    (artifact_id,) = result["artifact_ids"]
    created = next(e["payload"] for e in events if e["type"] == "artifact.created")
    assert created["download_url"] == f"/api/v1/artifacts/{artifact_id}/download"

    meta = live.get(f"/api/v1/artifacts/{artifact_id}", headers=headers())
    assert meta.status_code == 200, meta.text
    assert meta.json()["kind"] == "json"
    raw = live.get(created["view_url"], headers=headers())
    assert raw.status_code == 200 and raw.content == BIG_RESULT.encode()
    # Thread-owner access only (A8).
    stranger = live.get(f"/api/v1/artifacts/{artifact_id}", headers=headers(user="bob"))
    assert stranger.status_code == 404


def _elicitation(live: TestClient) -> tuple[str, str, dict[str, Any]]:
    thread_id = new_thread(live, agent="elicit")
    run_id = start_run(live, thread_id, {"input": {"text": "book"}})
    wait_for_status(live, run_id, "awaiting_approval")
    (record,) = live.get("/api/v1/interrupts", headers=headers()).json()["interrupts"]
    return thread_id, run_id, record


def test_an_mcp_elicitation_is_answered_through_resolve(live: TestClient) -> None:
    thread_id, run_id, record = _elicitation(live)
    assert record["kind"] == "mcp_elicitation"
    assert record["actions"] == ["edit", "reject"]
    assert record["payload"]["tool_name"] == "fx_book"
    assert record["payload"]["requests"][0]["key"] == "k1"

    missing = live.post(
        f"/api/v1/interrupts/{record['interrupt_id']}/resolve",
        json={"action": "edit", "payload": {"other": {}}, "expected_version": 1},
        headers=headers(),
    )
    assert_error(missing, 422, "invalid_edit")

    resolved = live.post(
        f"/api/v1/interrupts/{record['interrupt_id']}/resolve",
        json={"action": "edit", "payload": {"k1": {"date": "2026-11-01"}}, "expected_version": 1},
        headers=headers(),
    )
    assert resolved.status_code == 200, resolved.text
    wait_for_status(live, run_id, "completed")
    values = live.get(f"/api/v1/threads/{thread_id}", headers=headers()).json()["values"]
    assert values["answer"] == {
        "responses": {"k1": {"action": "accept", "content": {"date": "2026-11-01"}}}
    }


def test_rejecting_an_elicitation_declines_every_request(live: TestClient) -> None:
    thread_id, run_id, record = _elicitation(live)
    resolved = live.post(
        f"/api/v1/interrupts/{record['interrupt_id']}/resolve",
        json={"action": "reject", "expected_version": 1},
        headers=headers(),
    )
    assert resolved.status_code == 200, resolved.text
    wait_for_status(live, run_id, "completed")
    values = live.get(f"/api/v1/threads/{thread_id}", headers=headers()).json()["values"]
    assert values["answer"] == {"responses": {"k1": {"action": "decline"}}}
