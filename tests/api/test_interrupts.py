"""The interrupt inbox and resolve endpoint (§4.2, §4.3, A12, A44)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from _api import (
    Stack,
    assert_error,
    events_of,
    headers,
    new_thread,
    read_sse,
    start_run,
    wait_for_status,
)
from fastapi.testclient import TestClient

from vdreamer.interrupts import InterruptRecord, InterruptStatus
from vdreamer.queue import RunState

# -- end to end: require_approval → inbox → resolve → resume → completed --------------------


def _pending(client: TestClient, **kw: Any) -> list[dict[str, Any]]:
    return list(client.get("/api/v1/interrupts", headers=headers(**kw)).json()["interrupts"])


@pytest.mark.smoke  # floor/latest pins (A41): FastAPI + LangGraph together
def test_approval_round_trip_through_the_api(live: TestClient, live_stack: Stack) -> None:
    thread_id = new_thread(live, agent="gate")
    run_id = start_run(live, thread_id, {"input": {"text": "draft-1"}})
    wait_for_status(live, run_id, "awaiting_approval")

    (record,) = _pending(live)
    assert record["run_id"] == run_id and record["kind"] == "review"
    assert record["actions"] == ["approve", "reject", "edit", "feedback"]
    assert record["payload"] == {"draft": "draft-1"}
    assert record["status"] == "pending" and record["version"] == 1
    assert record["expires_at"] is not None
    assert _pending(live, user="bob") == []  # not the initiator, no approver role

    resolved = live.post(
        f"/api/v1/interrupts/{record['interrupt_id']}/resolve",
        json={"action": "edit", "payload": {"draft": "draft-2"}, "expected_version": 1},
        headers=headers(),
    )
    assert resolved.status_code == 200, resolved.text
    body = resolved.json()
    assert body["status"] == "resolved" and body["version"] == 2
    assert body["resolved_by"] == "alice"
    assert body["resolution"] == {"action": "edit", "payload": {"draft": "draft-2"}}

    final = wait_for_status(live, run_id, "completed")
    assert final["attempt"] == 2  # the resume consumed a fresh claim
    thread = live.get(f"/api/v1/threads/{thread_id}", headers=headers()).json()
    assert thread["values"]["decision"] == {"action": "edit", "payload": {"draft": "draft-2"}}

    _, frames = read_sse(live, f"/api/v1/runs/{run_id}/events")
    events = events_of(frames)
    assert [e["type"] for e in events] == [
        "run.status",
        "interrupt.requested",
        "run.status",
        "interrupt.resolved",
        "run.status",
        "run.status",
    ]
    statuses = [e["payload"]["status"] for e in events if e["type"] == "run.status"]
    assert statuses == ["running", "awaiting_approval", "running", "completed"]
    resolved_event = next(e for e in events if e["type"] == "interrupt.resolved")
    assert resolved_event["payload"] == {
        "interrupt_id": record["interrupt_id"],
        "action": "edit",
        "resolved_by": "alice",
    }
    assert [e["seq"] for e in events] == list(range(1, 7))  # one sequence across attempts

    # A second resolver loses (A12: the first resolver wins).
    again = live.post(
        f"/api/v1/interrupts/{record['interrupt_id']}/resolve",
        json={"action": "approve", "expected_version": 2},
        headers=headers(),
    )
    assert_error(again, 409, "version_conflict")


def test_cancel_awaiting_run_cancels_its_interrupts(live: TestClient, live_stack: Stack) -> None:
    run_id = start_run(live, new_thread(live, agent="gate"), {"input": {"text": "x"}})
    wait_for_status(live, run_id, "awaiting_approval")
    (record,) = _pending(live)
    assert live.post(f"/api/v1/runs/{run_id}/cancel", headers=headers()).status_code == 202
    got = live.get(f"/api/v1/interrupts/{record['interrupt_id']}", headers=headers()).json()
    assert got["status"] == "cancelled"
    assert_error(
        live.post(
            f"/api/v1/interrupts/{record['interrupt_id']}/resolve",
            json={"action": "approve", "expected_version": got["version"]},
            headers=headers(),
        ),
        409,
        "interrupt_not_pending",
    )
    _, frames = read_sse(live, f"/api/v1/runs/{run_id}/events")
    assert events_of(frames)[-1]["payload"]["status"] == "cancelled"


# -- seeded records: every resolve outcome -------------------------------------------------


def _seed(stack: Stack, **overrides: Any) -> InterruptRecord:
    async def go() -> InterruptRecord:
        run = await stack.queue.enqueue(
            "gate", {}, thread_id=overrides.pop("thread_id", "t1"), run_id=None, user_id="alice"
        )
        await stack.queue.claim("w")
        await stack.queue.suspend(run.id)
        fields: dict[str, Any] = {
            "interrupt_id": f"int_{run.id}",
            "run_id": run.id,
            "thread_id": run.thread_id,
            "agent": "gate",
            "kind": "sql_review",
            "actions": ("approve", "reject", "edit", "feedback", "respond"),
            "payload": {"sql": "SELECT 1"},
            "initiator_id": "alice",
            "created_at": time.time(),
            "expires_at": time.time() + 3600,
        }
        fields.update(overrides)
        return await stack.interrupts.create(InterruptRecord(**fields))

    return asyncio.run(go())


def _resolve(client: TestClient, record: InterruptRecord, body: dict[str, Any], **kw: Any) -> Any:
    payload = {"expected_version": record.version, **body}
    return client.post(
        f"/api/v1/interrupts/{record.interrupt_id}/resolve", json=payload, headers=headers(**kw)
    )


def test_role_gated_interrupts_go_to_the_role(client: TestClient, stack: Stack) -> None:
    record = _seed(stack, approver_role="dba")
    # The approver sees it in the inbox; so does the initiator (read-only).
    assert [r["interrupt_id"] for r in _pending(client, user="dana", roles=("dba",))] == [
        record.interrupt_id
    ]
    assert [r["interrupt_id"] for r in _pending(client)] == [record.interrupt_id]
    assert _pending(client, user="bob") == []
    assert_error(
        client.get(f"/api/v1/interrupts/{record.interrupt_id}", headers=headers(user="bob")),
        404,
        "interrupt_not_found",
    )
    error = assert_error(_resolve(client, record, {"action": "approve"}), 403, "forbidden_role")
    assert error["details"] == {"approver_role": "dba"}
    response = _resolve(client, record, {"action": "approve"}, user="dana", roles=("dba",))
    assert response.status_code == 200
    run = asyncio.run(stack.queue.get(record.run_id))
    assert run is not None and run.status is RunState.QUEUED
    assert run.resume == {record.interrupt_id: {"action": "approve"}}


def test_stale_version_is_409(client: TestClient, stack: Stack) -> None:
    record = _seed(stack)
    response = client.post(
        f"/api/v1/interrupts/{record.interrupt_id}/resolve",
        json={"action": "approve", "expected_version": 7},
        headers=headers(),
    )
    error = assert_error(response, 409, "version_conflict")
    assert error["details"] == {"version": 1}


def test_expired_interrupt_is_410(client: TestClient, stack: Stack) -> None:
    record = _seed(stack, expires_at=time.time() - 1)
    assert_error(_resolve(client, record, {"action": "approve"}), 410, "interrupt_expired")
    swept = _seed(stack, thread_id="t2", status=InterruptStatus.EXPIRED)
    assert_error(_resolve(client, swept, {"action": "approve"}), 410, "interrupt_expired")


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"action": "explode"}, "invalid_action"),
        ({"action": "edit"}, "invalid_edit"),
        ({"action": "feedback"}, "invalid_request"),
        ({"action": "respond", "text": "  "}, "invalid_request"),
        ({"decisions": [{"action": "approve"}, {"action": "nope"}]}, "invalid_action"),
    ],
)
def test_invalid_resolutions_are_422(
    client: TestClient, stack: Stack, body: dict[str, Any], code: str
) -> None:
    record = _seed(stack)
    assert_error(_resolve(client, record, body), 422, code)
    still = asyncio.run(stack.interrupts.get(record.interrupt_id))
    assert still is not None and still.status is InterruptStatus.PENDING


@pytest.mark.parametrize(
    "body",
    [
        {"expected_version": 1},  # neither form
        {"action": "approve", "decisions": [{"action": "approve"}], "expected_version": 1},
        {"decisions": [], "expected_version": 1},
        {"decisions": [{"action": "approve"}], "text": "x", "expected_version": 1},
        {"action": "approve"},  # no expected_version
    ],
)
def test_malformed_resolve_bodies_are_422(
    client: TestClient, stack: Stack, body: dict[str, Any]
) -> None:
    record = _seed(stack)
    response = client.post(
        f"/api/v1/interrupts/{record.interrupt_id}/resolve", json=body, headers=headers()
    )
    assert_error(response, 422, "invalid_request")


def test_decisions_form_builds_the_ordered_resume_value(client: TestClient, stack: Stack) -> None:
    record = _seed(stack)
    response = _resolve(
        client,
        record,
        {
            "decisions": [
                {"action": "approve"},
                {"action": "edit", "payload": {"sql": "SELECT 2"}},
                {"action": "respond", "text": "use the replica"},
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["resolution"] == {
        "decisions": [
            {"action": "approve"},
            {"action": "edit", "payload": {"sql": "SELECT 2"}},
            {"action": "respond", "text": "use the replica"},
        ]
    }


def test_interrupt_listing_filters_by_status(client: TestClient, stack: Stack) -> None:
    pending = _seed(stack)
    done = _seed(stack, thread_id="t2")
    assert _resolve(client, done, {"action": "approve"}).status_code == 200
    ids = [r["interrupt_id"] for r in _pending(client)]
    assert ids == [pending.interrupt_id]
    resolved = client.get("/api/v1/interrupts?status=resolved", headers=headers()).json()
    assert [r["interrupt_id"] for r in resolved["interrupts"]] == [done.interrupt_id]
    every = client.get("/api/v1/interrupts?status=all", headers=headers()).json()
    assert len(every["interrupts"]) == 2
    assert_error(
        client.get("/api/v1/interrupts?status=bogus", headers=headers()), 422, "invalid_request"
    )
    assert_error(
        client.get("/api/v1/interrupts/int_ghost", headers=headers()), 404, "interrupt_not_found"
    )


def test_resume_waits_for_every_pending_interrupt(client: TestClient, stack: Stack) -> None:
    first = _seed(stack)
    second = asyncio.run(
        stack.interrupts.create(
            InterruptRecord(
                interrupt_id="int_second",
                run_id=first.run_id,
                thread_id=first.thread_id,
                agent="gate",
                kind="generic",
                actions=("respond",),
                initiator_id="alice",
                created_at=time.time(),
            )
        )
    )
    assert _resolve(client, first, {"action": "approve"}).status_code == 200
    run = asyncio.run(stack.queue.get(first.run_id))
    assert run is not None and run.status is RunState.AWAITING_APPROVAL  # one still pending
    assert _resolve(client, second, {"action": "respond", "text": "ok"}).status_code == 200
    run = asyncio.run(stack.queue.get(first.run_id))
    assert run is not None and run.status is RunState.QUEUED
    assert run.resume == {
        first.interrupt_id: {"action": "approve"},
        "int_second": "ok",  # a raw interrupt() resumes with the answer verbatim (§4.5)
    }


def test_inbox_visibility_and_limit_are_store_side(client: TestClient, stack: Stack) -> None:
    from vdreamer.interrupts import Viewer

    gated = _seed(stack, approver_role="dba", initiator_id="carol")
    # Distinct created_at: back-to-back time.time() calls tie on Windows (~16 ms
    # resolution), and a tie orders by the random interrupt id.
    base = time.time()
    own = [_seed(stack, thread_id=f"t-own-{i}", created_at=base + i) for i in range(3)]
    dana = Viewer(user_id="dana", roles=("dba",))
    visible = asyncio.run(stack.interrupts.list(visible_to=dana))
    assert [r.interrupt_id for r in visible] == [gated.interrupt_id]
    page = client.get("/api/v1/interrupts?limit=2", headers=headers()).json()["interrupts"]
    assert [r["interrupt_id"] for r in page] == [r.interrupt_id for r in own[:2]]
    assert_error(
        client.get("/api/v1/interrupts?limit=501", headers=headers()), 422, "invalid_request"
    )
