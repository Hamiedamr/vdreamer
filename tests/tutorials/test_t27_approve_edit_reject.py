"""Lesson 27: approve, reject, edit and feedback — on a proposal and on tool calls."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from _tutorial_kit import TOKEN, served, vdreamer_run, wait_status
from pydantic import ValidationError
from vdreamer_tutorials.t27_approve_edit_reject import credit, tool_review
from vdreamer_tutorials.testing import AgentSession, call, scripted

LATE = "Order A1001 arrived a week late"


@pytest.fixture(autouse=True)
def clean_ledgers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credit, "ISSUED", {})
    monkeypatch.setattr(tool_review, "CREDITS", {})
    monkeypatch.setattr(tool_review, "CANCELLED", set())


# -- part 1: ApprovalGate --------------------------------------------------------------------


async def _proposed(*drafts: str) -> tuple[AgentSession, Any]:
    session = AgentSession(credit.agent, models={"chat": scripted(*drafts)})
    return session, await session.run(LATE)


async def test_the_proposal_waits_and_drafts_stay_private() -> None:
    _, first = await _proposed("Sorry! Here is a EUR 14.90 credit.")
    (review,) = first.interrupts
    assert review["actions"] == ["approve", "reject", "edit", "feedback"]
    assert review["payload"] == {"order_id": "A1001", "amount_eur": 14.9,
                                 "message": "Sorry! Here is a EUR 14.90 credit."}
    assert first.payloads("message.completed") == []  # the customer saw no draft


async def test_approve_executes_the_proposal() -> None:
    session, _ = await _proposed("Sorry! Here is a EUR 14.90 credit.")
    done = await session.resume({"action": "approve", "resolved_by": "bob"})
    assert credit.ISSUED == {"A1001": 14.9}
    assert [p["text"] for p in done.payloads("message.completed")] == [
        "Sorry! Here is a EUR 14.90 credit.", "(EUR 14.90 credited; approved by bob)"
    ]


async def test_reject_executes_nothing() -> None:
    session, _ = await _proposed("Sorry!")
    done = await session.resume({"action": "reject", "resolved_by": "bob"})
    assert credit.ISSUED == {} and "A colleague will follow up" in done.answer


async def test_edit_is_written_to_state_before_execution() -> None:
    session, _ = await _proposed("Sorry!")
    edited = {"amount_eur": 25.0, "message": "We are very sorry; here is EUR 25."}
    await session.resume({"action": "edit", "payload": edited, "resolved_by": "bob"})
    assert credit.ISSUED == {"A1001": 25.0}


async def test_feedback_redrafts_and_asks_again() -> None:
    session, _ = await _proposed("Sorry.", "We are truly sorry for the delay - EUR 14.90.")
    again = await session.resume({"action": "feedback", "text": "warmer, please",
                                  "resolved_by": "bob"})
    (review,) = again.interrupts
    assert review["payload"]["message"] == "We are truly sorry for the delay - EUR 14.90."


def test_the_edit_schema_bounds_what_an_approver_can_do() -> None:
    with pytest.raises(ValidationError):
        credit.CreditEdit(amount_eur=500, message="A very generous credit indeed.")


def test_the_gateway_refuses_an_invalid_edit() -> None:
    headers = {"Authorization": f"Bearer {TOKEN}", "X-User-Id": "alice"}
    backends = {credit.agent.name: lambda: {"models": {"chat": scripted("Sorry! EUR 14.90.")}}}
    with served([credit.agent], backends) as url, \
            httpx.Client(base_url=f"{url}/api/v1", headers=headers) as api:
        thread = api.post(f"/agents/{credit.agent.name}/threads").json()["thread_id"]
        run = api.post(f"/threads/{thread}/runs", json={"input": {"messages": [
            {"role": "user", "content": LATE}]}}).json()["run_id"]
        wait_status(url, run, "awaiting_approval")
        (pending,) = api.get("/interrupts").json()["interrupts"]
        bad = api.post(f"/interrupts/{pending['interrupt_id']}/resolve", json={
            "action": "edit", "payload": {"amount_eur": 500, "message": "Too generous a gift."},
            "expected_version": pending["version"]})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_edit"
    assert credit.ISSUED == {}


# -- part 2: reviewing tool calls ------------------------------------------------------------


async def test_a_reviewer_edits_the_tool_arguments() -> None:
    session = AgentSession(tool_review.agent, models={"chat": scripted(
        call("issue_credit", order_id="A1002", amount_eur=30), "Done.")})
    first = await session.run("A1002 was late, give EUR 30")
    (review,) = first.interrupts
    assert review["kind"] == "tool_review"
    assert review["payload"]["action_requests"][0]["args"] == {"order_id": "A1002",
                                                               "amount_eur": 30}
    edited = {"type": "edit", "edited_action": {"name": "issue_credit",
                                                "args": {"order_id": "A1002", "amount_eur": 15}}}
    done = await session.resume({"decisions": [edited]})
    assert tool_review.CREDITS == {"A1002": 15}
    # tool.call shows what the model proposed; the result says what actually ran.
    assert done.tool_calls[0][1]["amount_eur"] == 30
    assert "Executed instead: issue_credit" in done.tool_results[0]["summary"]


async def test_a_rejected_tool_call_tells_the_model_why() -> None:
    model = scripted(call("issue_credit", order_id="A1002", amount_eur=30), "Understood.")
    session = AgentSession(tool_review.agent, models={"chat": model})
    await session.run("credit please")
    await session.resume({"decisions": [{"type": "reject", "message": "Not eligible."}]})
    assert tool_review.CREDITS == {}
    assert "Not eligible." in model.prompts[1][-1].text


async def test_a_tool_policy_gate_needs_its_approver() -> None:
    model = scripted(call("cancel_order", order_id="A1002"), "Cancelled.")
    session = AgentSession(tool_review.agent, models={"chat": model})
    first = await session.run("cancel A1002")
    (gate,) = first.interrupts
    assert gate["kind"] == "cancel_review" and gate["approver_role"] == "supervisor"
    assert gate["payload"]["args"] == {"order_id": "A1002"}
    await session.resume({"action": "approve", "resolved_by": "sup"})
    assert tool_review.CANCELLED == {"A1002"}


def test_edit_through_vdreamer_run(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(
            call("issue_credit", order_id="A1002", amount_eur=30), "Credit sent.")}}

    with served([tool_review.agent], {tool_review.agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, "edit", '{"order_id": "A1002", "amount_eur": 15}',
                              agent=tool_review.agent.name, once=True, color=False,
                              message="A1002 was late, give EUR 30 credit")
    assert result.code == 0, result.out
    assert "[interrupt tool_review]" in result.out
    assert tool_review.CREDITS == {"A1002": 15}
