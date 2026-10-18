"""Lesson 25: a run that waits for a person — in-process, inline, and over REST."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from _tutorial_kit import TOKEN, served, vdreamer_run, wait_status
from vdreamer_tutorials.t25_human_in_the_loop import agent as lesson
from vdreamer_tutorials.t25_human_in_the_loop.agent import agent
from vdreamer_tutorials.testing import AgentSession


@pytest.fixture(autouse=True)
def no_refunds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lesson, "REFUNDED", {})


async def test_the_run_pauses_with_what_the_approver_sees() -> None:
    session = AgentSession(agent)
    result = await session.run("Please refund order A1001")

    assert result.status == "awaiting_approval"
    (request,) = result.interrupts
    assert request["kind"] == "refund_review"
    assert request["actions"] == ["approve", "reject"]
    assert request["payload"] == {"order_id": "A1001", "amount_eur": 149.0}
    assert lesson.REFUNDED == {}  # nothing happened yet
    waiting = await session.state()
    assert waiting.next == ("approve",)  # checkpointed, waiting at the approval node


async def test_approve_resumes_and_executes() -> None:
    session = AgentSession(agent)
    await session.run("Please refund order A1001")
    result = await session.resume({"action": "approve", "resolved_by": "bob"})

    assert result.status == "completed"
    assert result.answer == "Refunded EUR 149.00 for A1001 (approved by bob)."
    assert lesson.REFUNDED == {"A1001": 149.0}


async def test_reject_resumes_without_executing() -> None:
    session = AgentSession(agent)
    await session.run("Please refund order A1002")
    result = await session.resume({"action": "reject", "resolved_by": "bob"})

    assert result.answer.startswith("The refund was not approved")
    assert lesson.REFUNDED == {}


async def test_work_before_the_pause_runs_once() -> None:
    session = AgentSession(agent)
    first = await session.run("Please refund order A1003")
    resumed = await session.resume({"action": "approve", "resolved_by": "bob"})

    assert [p["label"] for p in first.payloads("progress")] == ["preparing the refund"]
    assert resumed.payloads("progress") == []  # the resume restarted `approve`, not `prepare`


async def test_without_an_order_id_there_is_nothing_to_approve() -> None:
    result = await AgentSession(agent).run("Please refund my order")
    assert result.status == "completed" and result.interrupts == []
    assert result.answer == "Which order should be refunded?"


def test_approved_inline_in_vdreamer_run(monkeypatch: Any) -> None:
    with served([agent]) as url:
        result = vdreamer_run(url, monkeypatch, "approve", agent=agent.name, once=True,
                              color=False, message="Please refund order A1001")
    assert result.code == 0, result.out
    assert "[interrupt refund_review] actions: approve, reject" in result.out
    assert "[interrupt resolved: approve by alice]" in result.out
    assert "Refunded EUR 149.00 for A1001 (approved by alice)." in result.out


def test_approved_over_rest(monkeypatch: Any) -> None:
    headers = {"Authorization": f"Bearer {TOKEN}", "X-User-Id": "alice"}
    with served([agent]) as url, httpx.Client(base_url=f"{url}/api/v1",
                                              headers=headers) as api:
        thread = api.post(f"/agents/{agent.name}/threads").json()["thread_id"]
        run = api.post(f"/threads/{thread}/runs", json={"input": {"messages": [
            {"role": "user", "content": "Please refund order A1002"}]}}).json()["run_id"]
        wait_status(url, run, "awaiting_approval")

        (pending,) = api.get("/interrupts").json()["interrupts"]
        resolved = api.post(f"/interrupts/{pending['interrupt_id']}/resolve",
                            json={"action": "approve", "expected_version": pending["version"]})
        wait_status(url, run, "completed")
        stale = api.post(f"/interrupts/{pending['interrupt_id']}/resolve",
                         json={"action": "reject", "expected_version": pending["version"]})
    assert pending["payload"] == {"order_id": "A1002", "amount_eur": 89.0}
    assert resolved.status_code == 200
    assert stale.status_code == 409  # decided once; a second decision is refused
    assert lesson.REFUNDED == {"A1002": 89.0}
