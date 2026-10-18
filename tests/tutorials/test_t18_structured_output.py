"""Lesson 18: typed input, a validated structured result, a retry, a fallback, an artifact."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from _tutorial_kit import TOKEN, served, vdreamer_run
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from pydantic import ValidationError
from vdreamer_tutorials.t18_structured_output.agent import agent
from vdreamer_tutorials.t18_structured_output.models import (
    UNCLASSIFIED,
    TicketInput,
    TicketTriage,
)
from vdreamer_tutorials.testing import AgentSession, call, scripted, structured

from vdreamer import BaseAgent, BuildContext

TICKET = {"subject": "Charged twice", "body": "Order A1002 was charged twice!"}
GOOD = TicketTriage(category="billing", priority="urgent", order_ids=["A1002"],
                    summary="Double charge on A1002.", needs_human=True)


def test_the_contracts_reject_bad_data() -> None:
    with pytest.raises(ValidationError):
        TicketInput(subject="", body="x")
    with pytest.raises(ValidationError, match="order ids look like"):
        TicketTriage(category="billing", priority="low", order_ids=["1002"], summary="s",
                     needs_human=False)
    assert TicketTriage.model_json_schema()["properties"]["priority"]["enum"] == [
        "low", "normal", "high", "urgent"
    ]


async def test_a_valid_answer_becomes_an_artifact_and_a_message() -> None:
    result = await AgentSession(agent, models={"chat": scripted(structured(GOOD))}).run(TICKET)

    assert result.values["triage"] == GOOD.model_dump()
    (artifact,) = result.payloads("artifact.created")
    assert artifact["kind"] == "json" and artifact["preview"]["body"] == GOOD.model_dump()
    assert result.answer == "billing / urgent — needs a human: Double charge on A1002."


async def test_an_invalid_answer_is_retried_with_the_reason() -> None:
    wrong = call("TicketTriage", category="shipping", priority="high", summary="x",
                 needs_human=False)
    model = scripted(wrong, structured(GOOD))
    result = await AgentSession(agent, models={"chat": model}).run(TICKET)

    assert result.values["triage"] == GOOD.model_dump()
    feedback = model.prompts[1][-1]
    assert isinstance(feedback, HumanMessage)
    assert "category: Input should be 'order_status', 'returns'" in feedback.text


async def test_after_the_retry_a_safe_fallback() -> None:
    wrong = call("TicketTriage", category="x", priority="y", summary="s", needs_human=False)
    result = await AgentSession(agent, models={"chat": scripted(wrong, "I think it's billing")}
                                ).run(TICKET)

    assert result.values["triage"] == UNCLASSIFIED.model_dump()
    assert result.values["validation_errors"][1] == "no TicketTriage call"
    assert result.payloads("log")[0]["level"] == "warning"


async def test_create_agents_response_format_is_the_prebuilt_alternative() -> None:
    class Prebuilt(BaseAgent):
        name = "t18-prebuilt"
        models = ("chat",)

        def build(self, ctx: BuildContext) -> Any:
            return create_agent(ctx.models["chat"], tools=[],
                                response_format=ToolStrategy(TicketTriage),
                                **ctx.agent_kwargs())

    result = await AgentSession(Prebuilt(), models={"chat": scripted(structured(GOOD))}).run(
        "Order A1002 was charged twice!"
    )
    assert result.values["structured_response"] == GOOD


def test_through_the_gateway(monkeypatch: Any) -> None:
    with served([agent], {agent.name: lambda: {"models": {"chat": scripted(structured(GOOD))}}}
                ) as url:
        bad = httpx.post(f"{url}/api/v1/threads/{_thread(url)}/runs",
                         headers=_headers(), json={"input": {"subject": "", "body": "x"}})
        good = vdreamer_run(url, monkeypatch, agent=agent.name, once=True, color=False,
                            input_json='{"subject": "Charged twice", '
                                       '"body": "Order A1002 was charged twice!"}')
    assert bad.status_code == 422  # rejected before any run: the input_schema at work
    assert good.code == 0, good.out
    assert "[artifact] json 'triage'" in good.out
    assert "billing / urgent — needs a human" in good.out


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}", "X-User-Id": "alice"}


def _thread(url: str) -> str:
    response = httpx.post(f"{url}/api/v1/agents/{agent.name}/threads", headers=_headers())
    return str(response.json()["thread_id"])
