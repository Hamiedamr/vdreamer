"""Milestone 8 exit: every interrupt source resolves through one endpoint (§4, A44).

All four sources — ``require_approval`` (here inside an ``ApprovalGate``),
``HumanInTheLoopMiddleware`` (``tool_review``), MCP elicitation and a raw
``interrupt()`` (``generic``) — become the same record and event and resolve
through ``POST /interrupts/{id}/resolve``. Plus the policy around it: ``edit``
validation, four-eyes (A32), the version race (A12), headless ``reject``, the
expiry sweeper, and ``interrupt_policies``.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
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
from _hitl_agents import SENT
from fastapi.testclient import TestClient

from vdreamer.interrupts import InterruptStatus, InterruptSweeper
from vdreamer.queue import RunState


def _inbox(client: TestClient, **kw: Any) -> list[dict[str, Any]]:
    response = client.get("/api/v1/interrupts", headers=headers(**kw))
    assert response.status_code == 200, response.text
    return list(response.json()["interrupts"])


def _only_pending(client: TestClient, run_id: str, **kw: Any) -> dict[str, Any]:
    (record,) = [r for r in _inbox(client, **kw) if r["run_id"] == run_id]
    return record


def _resolve(client: TestClient, record: dict[str, Any], body: dict[str, Any], **kw: Any) -> Any:
    return client.post(
        f"/api/v1/interrupts/{record['interrupt_id']}/resolve",
        json={"expected_version": record["version"], **body},
        headers=headers(**kw),
    )


def _values(client: TestClient, thread_id: str, **kw: Any) -> dict[str, Any]:
    return dict(client.get(f"/api/v1/threads/{thread_id}", headers=headers(**kw)).json()["values"])


def _awaiting(client: TestClient, agent: str, run_input: Any, **kw: Any) -> tuple[str, str]:
    thread_id = new_thread(client, agent=agent, **kw)
    run_id = start_run(client, thread_id, {"input": run_input}, **kw)
    wait_for_status(client, run_id, "awaiting_approval", **kw)
    return thread_id, run_id


# -- the exit criterion: one endpoint for all four sources -------------------------------


@pytest.mark.smoke  # floor/latest pins (A41): every interrupt source, end to end
def test_all_four_sources_resolve_through_one_endpoint(live: TestClient) -> None:
    SENT.clear()
    reviewer = {"user": "rita", "roles": ("mail_reviewer",)}
    cases: list[tuple[str, Any, dict[str, Any], dict[str, Any]]] = [
        ("flow", {"question": "orders"}, {"action": "approve"}, {}),
        ("review", {"messages": [{"role": "user", "content": "mail ops"}]},
         {"decisions": [{"action": "approve"}]}, reviewer),
        ("elicit", {"text": "fx"}, {"action": "edit", "payload": {"k1": {"date": "2026-11-01"}}},
         {}),
        ("ask", {"question": "backup"}, {"action": "respond", "text": "eu-west"}, {}),
    ]
    kinds: dict[str, str] = {}
    for agent, run_input, body, resolver in cases:
        thread_id, run_id = _awaiting(live, agent, run_input)
        record = _only_pending(live, run_id)
        kinds[agent] = record["kind"]
        response = _resolve(live, record, body, **resolver)
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "resolved"
        wait_for_status(live, run_id, "completed")
        _, frames = read_sse(live, f"/api/v1/runs/{run_id}/events")
        types = [event["type"] for event in events_of(frames)]
        assert types.count("interrupt.requested") == 1
        assert types.count("interrupt.resolved") == 1
    assert kinds == {
        "flow": "sql_review",
        "review": "tool_review",
        "elicit": "mcp_elicitation",
        "ask": "generic",
    }
    assert SENT == [{"to": "ops@example.internal", "body": "draft"}]


# -- require_approval through ApprovalGate: edit validation, feedback, reject -----------------


def test_approval_gate_feedback_then_validated_edit(live: TestClient) -> None:
    thread_id, run_id = _awaiting(live, "flow", {"question": "orders"})
    record = _only_pending(live, run_id)
    assert record["kind"] == "sql_review"
    assert record["payload"] == {"sql": "SELECT * FROM orders", "plans": 1}
    schema = record["edit_schema"]
    assert schema["properties"]["limit"]["maximum"] == 1000  # SqlEdit's JSON Schema (§4.2)

    # §4.3: an edit is validated against edit_schema before anything resumes.
    bad = _resolve(live, record, {"action": "edit", "payload": {"sql": "", "limit": 5000}})
    error = assert_error(bad, 422, "invalid_edit")
    assert {tuple(e["loc"]) for e in error["details"]["errors"]} == {("sql",), ("limit",)}
    extra = _resolve(live, record, {"action": "edit", "payload": {"sql": "x", "drop": True}})
    assert_error(extra, 422, "invalid_edit")
    assert _only_pending(live, run_id)["status"] == "pending"

    # feedback → replan → back through prepare to a fresh gate.
    assert _resolve(live, record, {"action": "feedback", "text": "only 2026"}).status_code == 200
    deadline = time.monotonic() + 5
    while not (pending := [r for r in _inbox(live) if r["run_id"] == run_id]):
        assert time.monotonic() < deadline, "the replanned gate never re-asked"
        time.sleep(0.02)
    (second,) = pending
    assert second["interrupt_id"] != record["interrupt_id"]
    assert second["payload"] == {"sql": "SELECT * FROM orders /* only 2026 */", "plans": 2}

    edit = {"action": "edit", "payload": {"sql": "SELECT id FROM orders", "limit": 5}}
    assert _resolve(live, second, edit).status_code == 200
    wait_for_status(live, run_id, "completed")
    values = _values(live, thread_id)
    # The edit was written to state before execute ran (§4.3).
    assert values["sql"] == "SELECT id FROM orders" and values["limit"] == 5
    assert values["result"] == "ran SELECT id FROM orders limit 5"


def test_chat_reject_routes_to_on_reject(live: TestClient) -> None:
    thread_id, run_id = _awaiting(live, "flow", {"question": "users"})
    record = _only_pending(live, run_id)
    assert _resolve(live, record, {"action": "reject", "text": "no"}).status_code == 200
    final = wait_for_status(live, run_id, "completed")  # chat: the conversation goes on
    assert final["status"] == "completed"
    assert _values(live, thread_id)["result"] == "not run"


# -- tool_review: interrupt_policies + the middleware's decisions shape ------------------------


def test_tool_review_takes_policies_and_resumes_the_middleware(live: TestClient) -> None:
    SENT.clear()
    run_input = {"messages": [{"role": "user", "content": "mail ops"}]}
    thread_id, run_id = _awaiting(live, "review", run_input)
    record = _only_pending(live, run_id)
    assert record["kind"] == "tool_review"
    assert record["actions"] == ["approve", "edit", "reject"]
    # interrupt_policies (§4.5): the tool-name glob's scope, the kind's role and TTL.
    assert record["approver_role"] == "mail_reviewer"
    assert record["scope"] == "mail:send"
    expires = datetime.fromisoformat(record["expires_at"]).timestamp()
    assert 1.5 * 3600 < expires - time.time() <= 2 * 3600 + 5
    (request,) = record["payload"]["action_requests"]
    assert request["name"] == "send_email"
    assert request["args"] == {"to": "ops@example.internal", "body": "draft"}

    # The initiator sees it but the role resolves it.
    assert_error(_resolve(live, record, {"decisions": [{"action": "approve"}]}), 403,
                 "forbidden_role")
    reviewer = {"user": "rita", "roles": ("mail_reviewer",)}
    wrong = _resolve(live, record, {"decisions": [{"action": "respond", "text": "x"}]}, **reviewer)
    assert_error(wrong, 422, "invalid_action")
    count = _resolve(live, record, {"decisions": [{"action": "approve"}] * 2}, **reviewer)
    assert_error(count, 422, "invalid_request")

    edited = {"to": "ops@example.internal", "body": "final text"}
    response = _resolve(
        live, record, {"decisions": [{"action": "edit", "payload": edited}]}, **reviewer
    )
    assert response.status_code == 200, response.text
    assert response.json()["resolution"] == {
        "decisions": [
            {"type": "edit", "edited_action": {"name": "send_email", "args": edited}}
        ]
    }
    wait_for_status(live, run_id, "completed")
    assert SENT == [edited]  # the reviewer's call ran, not the model's


# -- generic: the answer, verbatim -------------------------------------------------------------


def test_raw_interrupt_resumes_with_the_answer_verbatim(live: TestClient) -> None:
    thread_id, run_id = _awaiting(live, "ask", {"question": "backup"})
    record = _only_pending(live, run_id)
    assert record["kind"] == "generic" and record["actions"] == ["respond"]
    assert record["payload"] == {"value": "which region for backup?"}
    assert_error(_resolve(live, record, {"action": "respond"}), 422, "invalid_request")
    answer = {"region": "eu-west", "tier": 2}
    assert _resolve(live, record, {"action": "respond", "payload": answer}).status_code == 200
    wait_for_status(live, run_id, "completed")
    assert _values(live, thread_id)["answer"] == answer


# -- four-eyes (A32) and the version race (A12) ---------------------------------------------------


def test_strict_scope_needs_an_approver_other_than_the_initiator(live: TestClient) -> None:
    dba = ("dba",)
    _, run_id = _awaiting(live, "strict", {"text": "DELETE FROM t"}, roles=dba)
    record = _only_pending(live, run_id, roles=dba)
    assert record["scope"] == "db:write:prod"
    # alice holds the role, but she started the run: HITL_STRICT_SCOPES forbids it.
    error = assert_error(_resolve(live, record, {"action": "approve"}, roles=dba), 403,
                         "forbidden_role")
    assert error["details"] == {"reason": "four_eyes", "scope": "db:write:prod"}
    response = _resolve(live, record, {"action": "approve"}, user="bob", roles=dba)
    assert response.status_code == 200, response.text
    assert response.json()["resolved_by"] == "bob"
    wait_for_status(live, run_id, "completed", roles=dba)


def test_the_first_of_two_concurrent_resolvers_wins(client: TestClient, stack: Stack) -> None:
    thread_id = new_thread(client, agent="strict")
    run_id = start_run(client, thread_id, {"input": {"text": "x"}})

    async def suspend_with_interrupt() -> None:
        from vdreamer.interrupts import InterruptRecord

        await stack.queue.claim("w")
        await stack.queue.suspend(run_id)
        await stack.interrupts.create(
            InterruptRecord(
                interrupt_id="int_race",
                run_id=run_id,
                thread_id=thread_id,
                agent="strict",
                kind="sql_write",
                actions=("approve", "reject"),
                approver_role="dba",
                initiator_id="alice",
                created_at=time.time(),
                expires_at=time.time() + 60,
            )
        )

    asyncio.run(suspend_with_interrupt())
    record = {"interrupt_id": "int_race", "version": 1}
    first = _resolve(client, record, {"action": "approve"}, user="bob", roles=("dba",))
    second = _resolve(client, record, {"action": "reject"}, user="carol", roles=("dba",))
    assert first.status_code == 200
    error = assert_error(second, 409, "version_conflict")
    assert error["details"]["version"] == 2

    async def race() -> list[Any]:
        from vdreamer.interrupts import InterruptConflict, InterruptRecord

        await stack.interrupts.create(
            InterruptRecord(
                interrupt_id="int_race_2",
                run_id=run_id,
                thread_id=thread_id,
                agent="strict",
                kind="sql_write",
                actions=("approve",),
                created_at=time.time(),
            )
        )
        attempts = [
            stack.interrupts.resolve(
                "int_race_2", expected_version=1, resolved_by=who, resolution={"action": "approve"}
            )
            for who in ("bob", "carol", "dave")
        ]
        results = await asyncio.gather(*attempts, return_exceptions=True)
        assert sum(isinstance(r, InterruptConflict) for r in results) == 2
        return results

    winners = [r for r in asyncio.run(race()) if not isinstance(r, Exception)]
    assert len(winners) == 1 and winners[0].version == 2


# -- headless reject → rejected (§4.3) -------------------------------------------------------------


def _start_headless(live: TestClient, stack: Stack, site_id: str) -> tuple[str, str]:
    async def enqueue() -> tuple[str, str]:
        thread = await stack.threads.create("sitecheck", "svc-mail")
        run = await stack.queue.enqueue(
            "sitecheck", {"site_id": site_id}, thread_id=thread.id, run_id=None,
            user_id="svc-mail",
        )
        return thread.id, run.id

    thread_id, run_id = asyncio.run(enqueue())
    wait_for_status(live, run_id, "awaiting_approval", user="svc-mail")
    return thread_id, run_id


def test_headless_reject_ends_the_run_rejected(live: TestClient, live_stack: Stack) -> None:
    approver = {"user": "sam", "roles": ("site_approver",)}
    _, run_id = _start_headless(live, live_stack, "S-1")
    record = _only_pending(live, run_id, **approver)
    assert record["approver_role"] == "site_approver"
    response = _resolve(live, record, {"action": "reject", "text": "antenna tilt"}, **approver)
    assert response.status_code == 200, response.text
    final = wait_for_status(live, run_id, "rejected", user="svc-mail")
    assert final["attempt"] == 1  # nothing resumed: the gateway ended it
    _, frames = read_sse(live, f"/api/v1/runs/{run_id}/events", user="svc-mail")
    events = events_of(frames)
    assert [e["type"] for e in events][-2:] == ["interrupt.resolved", "run.status"]
    assert events[-1]["payload"]["status"] == "rejected"

    # An approve resumes the headless run as usual.
    _, approved = _start_headless(live, live_stack, "S-2")
    record = _only_pending(live, approved, **approver)
    assert _resolve(live, record, {"action": "approve"}, **approver).status_code == 200
    wait_for_status(live, approved, "completed", user="svc-mail")


# -- expiry (§4.4) ---------------------------------------------------------------------------------


def test_sweeper_expires_the_interrupt_and_the_run(live: TestClient, live_stack: Stack) -> None:
    thread_id, run_id = _awaiting(live, "flow", {"question": "late"})
    record = _only_pending(live, run_id)
    later = time.time() + 2 * 3600  # past the gate's 1 h TTL
    sweeper = InterruptSweeper(
        live_stack.interrupts, live_stack.queue, live_stack.log, clock=lambda: later
    )
    expired = asyncio.run(sweeper.sweep_once())
    assert [r.interrupt_id for r in expired] == [record["interrupt_id"]]
    assert asyncio.run(sweeper.sweep_once()) == []  # idempotent

    final = wait_for_status(live, run_id, "expired")
    assert final["status"] == "expired"
    stored = asyncio.run(live_stack.interrupts.get(record["interrupt_id"]))
    assert stored is not None and stored.status is InterruptStatus.EXPIRED
    assert_error(_resolve(live, record, {"action": "approve"}), 410, "interrupt_expired")
    _, frames = read_sse(live, f"/api/v1/runs/{run_id}/events")
    last = events_of(frames)[-1]
    assert last["payload"] == {
        "status": "expired",
        "attempt": 1,
        "error": {"code": "interrupt_expired", "message": last["payload"]["error"]["message"]},
    }
    # The thread unlocked: a new run is accepted at once (no 409).
    next_run = start_run(live, thread_id, {"input": {"question": "again"}})
    assert asyncio.run(live_stack.queue.get(next_run)) is not None


def test_sweeper_skips_when_not_the_leader(client: TestClient, stack: Stack) -> None:
    class Follower:
        async def is_leader(self) -> bool:
            return False

        async def close(self) -> None:
            return None

    sweeper = InterruptSweeper(
        stack.interrupts, stack.queue, stack.log, leader=Follower(), clock=lambda: 1e12
    )
    assert asyncio.run(sweeper.sweep_once()) == []


def test_sweeper_loop_stops_and_releases_leadership(stack: Stack) -> None:
    released: list[bool] = []

    class Leader:
        async def is_leader(self) -> bool:
            return True

        async def close(self) -> None:
            released.append(True)

    async def go() -> None:
        stop = asyncio.Event()
        sweeper = InterruptSweeper(
            stack.interrupts, stack.queue, stack.log, leader=Leader(), interval_s=0.01
        )
        task = asyncio.create_task(sweeper.run(stop))
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, 1.0)

    asyncio.run(go())
    assert released == [True]
    assert RunState.EXPIRED.value == "expired"
