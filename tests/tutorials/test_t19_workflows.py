"""Lesson 19: a headless workflow — fixed steps, a classified branch, triggered by systems."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from _tutorial_kit import SERVICE_TOKEN, TOKEN, served, vdreamer_run
from vdreamer_tutorials.t19_workflows.agent import (
    Classification,
    agent,
    check_payment,
    check_return,
    route,
)
from vdreamer_tutorials.testing import AgentSession, RunResult, call, scripted, structured

from vdreamer import VDreamerContext, bind_context

TICKET = {"ticket_id": "T-1", "body": "Was order A1002 charged twice?"}


def test_deterministic_steps() -> None:
    # The steps call emit_progress, which needs a run: bind one, as a worker would.
    context = VDreamerContext(run_id="run_1", thread_id="t", agent_name=agent.name)
    with bind_context(context):
        payment = check_payment({"ticket_id": "T", "body": "", "order_id": "A1002"})
        window = check_return({"ticket_id": "T", "body": "", "order_id": "A1003"})
    assert payment["findings"].endswith("DUPLICATE CHARGE")
    assert window["findings"] == "delivered 47 days ago, outside the 30-day window"
    assert [r["label"] for r in context.pending] == ["checking payments",
                                                     "checking the return window"]
    assert route({"ticket_id": "T", "body": "", "category": "other"}) == "draft_reply"


async def _run(classification: Any, *replies: str, body: str = TICKET["body"]) -> RunResult:
    model = scripted(classification, *replies)
    return await AgentSession(agent, models={"chat": model}).run(
        {"ticket_id": "T-1", "body": body}, headless=True
    )


@pytest.mark.parametrize(
    ("category", "steps"),
    [
        ("billing", ["classifying", "checking payments", "drafting a reply", "done"]),
        ("returns", ["classifying", "checking the return window", "drafting a reply", "done"]),
        ("other", ["classifying", "drafting a reply", "done"]),
        ("complaint", ["classifying", "escalated to the support team", "done"]),
    ],
)
async def test_each_category_takes_its_path(category: str, steps: list[str]) -> None:
    replies = [] if category == "complaint" else ["Our reply."]
    result = await _run(structured(Classification(category=category)), *replies)  # type: ignore[arg-type]

    assert [p["label"] for p in result.payloads("progress")] == steps
    (resolution,) = result.payloads("artifact.created")
    assert resolution["preview"]["body"]["category"] == category
    assert len(result.payloads("message.completed")) == 1  # one reply, streamed once


async def test_the_reply_is_grounded_in_the_checked_facts() -> None:
    model = scripted(structured(Classification(category="billing")), "We will refund one.")
    await AgentSession(agent, models={"chat": model}).run(TICKET, headless=True)
    assert "DUPLICATE CHARGE" in model.prompts[1][0].text


async def test_code_overrules_the_model_on_checkable_facts() -> None:
    wrong_id = structured(Classification(category="billing", order_id="A9999"))
    result = await _run(wrong_id, "ok")
    assert result.payloads("artifact.created")[0]["preview"]["body"]["order_id"] == "A1002"


async def test_an_unusable_classification_falls_back_to_other() -> None:
    result = await _run(call("Classification", category="shipping"), "ok")
    assert result.payloads("artifact.created")[0]["preview"]["body"]["category"] == "other"


def test_systems_trigger_it_idempotently(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(structured(Classification(category="billing")),
                                            "We are refunding the duplicate charge.")}}

    with served([agent], {agent.name: backends}) as url:
        first = vdreamer_run(url, monkeypatch, agent=agent.name, trigger=True, service=True,
                             idempotency_key="ticket-T-1", input_json='{"ticket_id": "T-1", '
                             '"body": "Was order A1002 charged twice?"}', color=False)
        retry = httpx.post(f"{url}/api/v1/agents/{agent.name}/trigger",
                           headers={"Authorization": f"Bearer {SERVICE_TOKEN}",
                                    "Idempotency-Key": "ticket-T-1"},
                           json={"input": TICKET})
        chat = httpx.post(f"{url}/api/v1/agents/{agent.name}/threads",
                          headers={"Authorization": f"Bearer {TOKEN}", "X-User-Id": "alice"})
    assert first.code == 0, first.out
    assert "We are refunding the duplicate charge." in first.out
    assert retry.json()["run_id"] in first.out  # the same key returns the same run
    assert chat.status_code == 400  # a HEADLESS agent has no chat threads
    assert chat.json()["error"]["code"] == "mode_not_supported"
