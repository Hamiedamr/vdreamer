"""Lesson 22: a supervisor delegating to specialists through tools."""

from __future__ import annotations

from typing import Any

from _tutorial_kit import served, vdreamer_run
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from vdreamer_tutorials.t21_specialists.specialists import SPECIALISTS
from vdreamer_tutorials.t22_supervisor.agent import agent, make_delegate, supervisor_prompt
from vdreamer_tutorials.testing import AgentSession, RoutedChatModel, call, calls, scripted

SUPERVISOR, ORDERS, PRODUCTS, POLICIES = (
    "You coordinate a support team", "existing orders only", "product questions only",
    "quote shop policies only",
)


def test_each_specialist_becomes_a_described_tool() -> None:
    tools = [make_delegate(spec, scripted()) for spec in SPECIALISTS]
    specs = [convert_to_openai_tool(t)["function"] for t in tools]
    assert [s["name"] for s in specs] == ["ask_orders", "ask_products", "ask_policies"]
    assert specs[0]["description"].startswith("Answers questions about existing orders")
    assert list(specs[0]["parameters"]["properties"]) == ["question"]
    assert "- ask_policies: Quotes the shop's returns" in supervisor_prompt(SPECIALISTS)


async def test_the_supervisor_asks_two_specialists_in_parallel() -> None:
    routes = {
        SUPERVISOR: scripted(
            calls(("ask_products", {"question": "Is the family tent (TENT-4P) in stock?"}),
                  ("ask_policies", {"question": "What is the returns policy?"})),
            "Yes, 3 family tents are in stock, and unused items can be returned within 30 "
            "days.",
        ),
        PRODUCTS: scripted(call("search_catalog", query="family tent"), "3 in stock, EUR 289."),
        POLICIES: scripted(call("get_policy", topic="returns"), "Returns within 30 days."),
    }
    model = RoutedChatModel(routes=routes)
    result = await AgentSession(agent, models={"chat": model}).run(
        "Is the family tent in stock, and can I return it if it's too big?"
    )

    assert result.status == "completed"
    top = [name for name, _ in result.tool_calls if name.startswith("ask_")]
    assert sorted(top) == ["ask_policies", "ask_products"]
    # Each specialist saw only the supervisor's question, not the customer's message.
    products_prompt = routes[PRODUCTS].prompts[0]
    assert [m.text for m in products_prompt if isinstance(m, HumanMessage)] == [
        "Is the family tent (TENT-4P) in stock?"
    ]
    # The supervisor's final answer was written from the specialists' answers.
    observed = sorted(m.text for m in routes[SUPERVISOR].prompts[1] if isinstance(m, ToolMessage))
    assert observed == ["3 in stock, EUR 289.", "Returns within 30 days."]


async def test_nested_tool_calls_are_governed_and_namespaced() -> None:
    routes = {
        SUPERVISOR: scripted(call("ask_orders", question="Status of A1001?"), "It shipped."),
        ORDERS: scripted(call("get_order", order_id="A1001"), "A1001 has shipped."),
    }
    result = await AgentSession(agent, models={"chat": RoutedChatModel(routes=routes)}).run(
        "Where is A1001?"
    )
    events = {p["name"]: e for e in result.events if e.type.value == "tool.call"
              for p in [e.payload]}
    assert events["ask_orders"].ns == ()  # the supervisor's own call
    assert events["get_order"].ns and events["get_order"].ns[0].startswith("tools:")


async def test_a_specialist_that_fails_is_contained_for_the_supervisor() -> None:
    routes = {
        SUPERVISOR: scripted(call("ask_orders", question="Status of A9999?"),
                             "I could not find that order."),
        ORDERS: scripted(call("get_order", order_id="A9999"), "There is no order A9999."),
    }
    result = await AgentSession(agent, models={"chat": RoutedChatModel(routes=routes)}).run(
        "Where is A9999?"
    )
    statuses = {r["status"] for r in result.tool_results}
    assert statuses == {"ok", "error"}  # the inner lookup failed; the delegation succeeded
    assert result.answer == "I could not find that order."


def test_through_the_gateway(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        return {"models": {"chat": RoutedChatModel(routes={
            SUPERVISOR: scripted(call("ask_policies", question="Warranty on tents?"),
                                 "Tents have a 2-year warranty."),
            POLICIES: scripted(call("get_policy", topic="warranty"), "2 years."),
        })}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name, once=True, color=False,
                              message="What warranty do tents have?")
    assert result.code == 0, result.out
    assert "-> ask_policies(" in result.out and "-> get_policy(" in result.out
    assert "Tents have a 2-year warranty." in result.out
