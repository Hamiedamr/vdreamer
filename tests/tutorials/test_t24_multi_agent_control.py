"""Lesson 24: limits and containment at every level of a multi-agent system."""

from __future__ import annotations

from typing import Any

from _tutorial_kit import served, vdreamer_run
from vdreamer_tutorials.t24_multi_agent_control.agent import GuardedSupervisorAgent, Limits
from vdreamer_tutorials.testing import (
    AgentSession,
    RoutedChatModel,
    RunResult,
    ScriptedChatModel,
    call,
    calls,
    reply,
    scripted,
)

SUPERVISOR, ORDERS, PRODUCTS = "You coordinate", "existing orders only", "product questions only"


async def _run(limits: Limits, **routes: ScriptedChatModel) -> RunResult:
    keys = {"supervisor": SUPERVISOR, "orders": ORDERS, "products": PRODUCTS}
    model = RoutedChatModel(routes={keys[name]: script for name, script in routes.items()})
    return await AgentSession(GuardedSupervisorAgent(limits), models={"chat": model}).run(
        "Help me"
    )


async def test_a_runaway_specialist_is_stopped_not_the_run() -> None:
    looping = scripted(*[call("search_catalog", query="tent")] * 10)
    result = await _run(
        Limits(max_specialist_model_calls=2),
        supervisor=scripted(call("ask_products", question="Tents?"), "Partial answer."),
        products=looping,
    )
    assert looping.calls_made == 2
    delegation = [r for r in result.tool_results if r["summary"].startswith("Model call")]
    assert delegation[0]["summary"] == "Model call limits exceeded: run limit (2/2)"
    assert result.status == "completed" and result.answer == "Partial answer."


async def test_a_slow_specialist_times_out() -> None:
    result = await _run(
        Limits(delegation_timeout_s=0.3),
        supervisor=scripted(call("ask_orders", question="Status of A1001?"), "Try later."),
        orders=ScriptedChatModel(replies=[reply("too late")], delay_s=1.0),
    )
    assert result.tool_results[0]["error"] == {
        "code": "timeout", "message": "the tool did not finish within 0.3s"
    }
    assert result.answer == "Try later."


async def test_a_specialist_whose_model_is_down_is_contained() -> None:
    supervisor = scripted(call("ask_orders", question="Status of A1001?"),
                          "Order lookups are unavailable right now.")
    result = await _run(Limits(), supervisor=supervisor,
                        orders=scripted(ConnectionError("orders model down")))

    assert result.tool_results[0]["error"]["code"] == "transport"
    seen = supervisor.prompts[1][-1]
    assert seen.text == "Error (transport): orders model down"  # the supervisor can explain
    assert result.status == "completed"


async def test_asking_one_specialist_too_often_is_blocked() -> None:
    result = await _run(
        Limits(max_asks_per_specialist=1),
        supervisor=scripted(calls(("ask_products", {"question": "a"}),
                                  ("ask_products", {"question": "b"})), "Answered one."),
        products=scripted("A."),
    )
    statuses = sorted(r["status"] for r in result.tool_results)
    assert statuses == ["error", "ok"]
    blocked = [r for r in result.tool_results if r["status"] == "error"][0]
    assert "Tool call limit exceeded" in blocked["error"]["message"]


async def test_the_total_delegation_budget() -> None:
    result = await _run(
        Limits(max_delegations=2, max_asks_per_specialist=5),
        supervisor=scripted(calls(("ask_orders", {"question": "a"}),
                                  ("ask_orders", {"question": "b"}),
                                  ("ask_products", {"question": "c"})), "Done."),
        orders=scripted("A.", "B."), products=scripted("C."),
    )
    assert sorted(r["status"] for r in result.tool_results) == ["error", "ok", "ok"]


async def test_the_recursion_limit_is_the_last_backstop() -> None:
    model = RoutedChatModel(routes={
        SUPERVISOR: scripted(*[call("ask_orders", question="again?")] * 10),
        ORDERS: scripted(*["no idea"] * 10),
    })
    session = AgentSession(GuardedSupervisorAgent(Limits(max_asks_per_specialist=10,
                                                         max_delegations=10)),
                           models={"chat": model})
    result = await session.run("loop", recursion_limit=6)
    assert result.status == "failed"
    assert "Recursion limit of 6 reached" in result.payloads("run.status")[-1]["error"]["message"]


def test_through_the_gateway(monkeypatch: Any) -> None:
    agent = GuardedSupervisorAgent(Limits(delegation_timeout_s=0.3))

    def backends() -> dict[str, Any]:
        return {"models": {"chat": RoutedChatModel(routes={
            SUPERVISOR: scripted(call("ask_orders", question="A1001?"),
                                 "Order lookups are slow right now; please try again soon."),
            ORDERS: ScriptedChatModel(replies=[reply("late")], delay_s=1.0),
        })}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name, message="Where is A1001?",
                              once=True, color=False)
    assert result.code == 0, result.out
    assert "<- error Error (timeout): the tool did not finish within 0.3s" in result.out
    assert "please try again soon" in result.out
