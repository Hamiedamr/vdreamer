"""Lesson 6: testing an agent at every level — rule, tool, agent, trace, gateway."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from _tutorial_kit import served, vdreamer_run
from langchain_core.messages import SystemMessage, ToolMessage
from vdreamer_tutorials.t06_testing_and_traces.agent import (
    SYSTEM_PROMPT,
    ReturnsAgent,
    make_check_return,
    return_decision,
)
from vdreamer_tutorials.testing import AgentSession, ScriptExhausted, call, scripted

TODAY = date(2026, 10, 1)  # A1003 was delivered on 2026-09-01: exactly 30 days ago


# -- level 1: the business rule, no LangChain at all -----------------------------------------


@pytest.mark.parametrize(
    ("today", "verdict"),
    [
        (date(2026, 10, 1), "can be returned: delivered 30 days ago"),     # the last day
        (date(2026, 10, 2), "cannot be returned: delivered 31 days ago"),  # one day late
        (date(2026, 9, 1), "can be returned: delivered 0 days ago"),       # delivery day
    ],
)
def test_the_30_day_window_boundaries(today: date, verdict: str) -> None:
    assert verdict in return_decision("a1003", today)


def test_undelivered_and_unknown_orders() -> None:
    assert "has not been delivered yet" in return_decision("A1002", TODAY)
    # A1001 is delivered on 2026-10-04: after TODAY. Found by the gateway test below,
    # which once printed "delivered -3 days ago".
    assert "has not been delivered yet" in return_decision("A1001", TODAY)
    assert return_decision("Z1", TODAY) == "No order with id 'Z1' was found."


# -- level 2: the tool — its contract and its wiring to the clock ------------------------------


def test_the_tool_reads_the_injected_clock() -> None:
    tool = make_check_return(lambda: TODAY)
    assert tool.name == "check_return"
    assert tool.invoke({"order_id": "A1003"}).startswith("Order A1003 can be returned")


# -- level 3: the agent, with a scripted model -------------------------------------------------


def _session(*replies: Any) -> tuple[AgentSession, Any]:
    model = scripted(*replies)
    return AgentSession(ReturnsAgent(today=lambda: TODAY), models={"chat": model}), model


async def test_the_agent_checks_then_answers() -> None:
    session, model = _session(
        call("check_return", order_id="A1003"),
        "Yes — order A1003 can still be returned today, the last day of its 30-day window.",
    )
    result = await session.run("Can I still return order A1003?")

    # Behavior, not wording: the right tool, the right argument, a completed run.
    assert result.status == "completed"
    assert result.tool_calls == [("check_return", {"order_id": "A1003"})]
    # What the model was given: the system prompt, then the tool's verdict.
    assert isinstance(model.prompts[0][0], SystemMessage)
    assert model.prompts[0][0].text == SYSTEM_PROMPT
    verdict = model.prompts[1][-1]
    assert isinstance(verdict, ToolMessage)
    assert verdict.text == "Order A1003 can be returned: delivered 30 days ago, within 30."


async def test_the_trace_reads_like_a_story() -> None:
    session, _ = _session(call("check_return", order_id="A1002"), "Not delivered yet.")
    result = await session.run("Return A1002?")

    assert result.trace() == "\n".join([
        "run.status        running",
        "message.completed ai: ''",
        "tool.call         check_return({'order_id': 'A1002'})",
        "tool.result       ok: Order A1002 has not been delivered yet, so it cannot be "
        "returned yet.",
        "message.completed ai: 'Not delivered yet.'",
        "run.status        completed",
    ])


async def test_an_unscripted_extra_model_call_fails_loudly() -> None:
    session, model = _session(call("check_return", order_id="A1003"))  # no final answer scripted
    result = await session.run("Return A1003?")

    assert result.status == "failed"
    assert isinstance(result.error, ScriptExhausted)
    assert model.calls_made == 2


# -- level 4: the whole stack: gateway, worker, SSE, `vdreamer run` ----------------------------


def test_through_the_gateway(monkeypatch: Any) -> None:
    agent = ReturnsAgent(today=lambda: TODAY)

    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(call("check_return", order_id="A1003"), "Yes.")}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name,
                              message="Return A1003?", once=True, color=False)
    assert result.code == 0, result.out
    assert "<- ok Order A1003 can be returned: delivered 30 days ago, within 30." in result.out
    assert "[run completed]" in result.out
