"""Lesson 3: the ReAct loop by hand — reason, act, observe, and a step budget."""

from __future__ import annotations

from typing import Any

from _tutorial_kit import served, vdreamer_run
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from vdreamer_tutorials.t03_react_loop.agent import MAX_STEPS, agent, steps_this_turn
from vdreamer_tutorials.testing import AgentSession, call, calls, scripted


async def test_two_orders_take_two_cycles_then_an_answer() -> None:
    model = scripted(
        call("get_order_status", order_id="A1001"),
        call("get_order_status", order_id="A1003"),
        "A1001 is on its way; A1003 was delivered on 28 September.",
    )
    result = await AgentSession(agent, models={"chat": model}).run("Compare A1001 and A1003")

    assert result.status == "completed"
    assert [args["order_id"] for _, args in result.tool_calls] == ["A1001", "A1003"]
    assert model.calls_made == 3
    # Each model call saw every observation so far.
    observed = [[m.text for m in prompt if isinstance(m, ToolMessage)] for prompt in model.prompts]
    assert observed == [
        [],
        ["Order A1001: shipped on 2026-10-01, arriving 2026-10-04"],
        ["Order A1001: shipped on 2026-10-01, arriving 2026-10-04",
         "Order A1003: delivered on 2026-09-28"],
    ]
    print(result.trace())  # `pytest -s` shows the reason → act → observe sequence


async def test_parallel_tool_calls_in_one_step() -> None:
    model = scripted(
        calls(("get_order_status", {"order_id": "A1001"}),
              ("get_order_status", {"order_id": "A1002"})),
        "Both found.",
    )
    result = await AgentSession(agent, models={"chat": model}).run("A1001 and A1002?")

    assert len(result.tool_results) == 2 and model.calls_made == 2
    assert result.answer == "Both found."


async def test_the_step_budget_stops_a_runaway_loop() -> None:
    model = scripted(*[call("get_order_status", order_id="A1001") for _ in range(10)])
    session = AgentSession(agent, models={"chat": model})
    result = await session.run("Check A1001 forever")

    assert result.status == "completed"
    assert model.calls_made == MAX_STEPS
    assert result.answer.startswith(f"I could not finish this within {MAX_STEPS} steps")
    # The last request was answered with "skipped" tool messages: the history stays valid.
    messages = (await session.state()).values["messages"]
    last_request = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls][-1]
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    assert {c["id"] for c in last_request.tool_calls} <= answered


async def test_the_budget_is_per_turn_not_per_thread() -> None:
    model = scripted(
        call("get_order_status", order_id="A1001"), call("get_order_status", order_id="A1002"),
        call("get_order_status", order_id="A1003"), "Three done.",
        call("get_order_status", order_id="A1001"), "Still shipped.",
    )
    session = AgentSession(agent, models={"chat": model})
    first = await session.run("Check all three")
    second = await session.run("And A1001 again?")

    assert first.answer == "Three done." and second.answer == "Still shipped."


async def test_an_unknown_tool_becomes_an_error_observation() -> None:
    model = scripted(call("cancel_order", order_id="A1001"), "Sorry, I cannot cancel orders.")
    result = await AgentSession(agent, models={"chat": model}).run("Cancel A1001")

    assert result.status == "completed"
    error = [m for m in model.prompts[1] if isinstance(m, ToolMessage)][0]
    assert error.status == "error" and "no tool named 'cancel_order'" in error.text
    assert result.answer == "Sorry, I cannot cancel orders."


def test_counting_steps_this_turn() -> None:
    history = [HumanMessage("a"), AIMessage("x"), HumanMessage("b"),
               AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
               ToolMessage("r", tool_call_id="1"), AIMessage("y")]
    assert steps_this_turn(history) == 2
    assert steps_this_turn([HumanMessage("only")]) == 0


def test_through_the_gateway(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(
            call("get_order_status", order_id="A1001"),
            call("get_order_status", order_id="A1003"),
            "A1001 ships, A1003 arrived.",
        )}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name,
                              message="Compare A1001 and A1003", once=True, color=False)
    assert result.code == 0, result.out
    assert result.out.count("-> get_order_status(") == 2
    assert "A1001 ships, A1003 arrived." in result.out


async def test_langgraphs_recursion_limit_fails_the_run_instead() -> None:
    # Each node execution is one superstep: reason, act, reason, act, reason → limit 5 hit.
    model = scripted(*[call("get_order_status", order_id="A1001") for _ in range(10)])
    result = await AgentSession(agent, models={"chat": model}).run("Loop", recursion_limit=5)

    assert result.status == "failed"
    error = result.payloads("run.status")[-1]["error"]
    assert error["code"] == "node_error" and "Recursion limit of 5 reached" in error["message"]
