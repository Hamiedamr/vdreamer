"""Lesson 23: handoffs between specialists, shared state, and a clean client stream."""

from __future__ import annotations

from typing import Any

from _tutorial_kit import served, vdreamer_run
from langgraph.graph import START, StateGraph
from vdreamer_tutorials.t21_specialists.specialists import SPECIALISTS
from vdreamer_tutorials.t23_handoffs.agent import (
    DeskState,
    agent,
    build_desk_agent,
    route_first,
)
from vdreamer_tutorials.testing import AgentSession, RoutedChatModel, call, scripted

from vdreamer import BaseAgent, BuildContext

ORDERS, PRODUCTS, POLICIES = "existing orders only", "product questions only", "policies only"


def _routes(**scripts: Any) -> RoutedChatModel:
    keys = {"orders": ORDERS, "products": PRODUCTS, "policies": POLICIES}
    return RoutedChatModel(routes={keys[name]: script for name, script in scripts.items()})


def test_the_first_message_goes_to_first_contact() -> None:
    assert route_first({"messages": []}) == "orders"
    assert route_first({"messages": [], "active_agent": "policies"}) == "policies"


async def test_a_handoff_moves_the_customer_and_carries_a_note() -> None:
    products = scripted(call("search_catalog", query="family tent"), "TENT-4P: 3 in stock.")
    model = _routes(
        orders=scripted(call("transfer_to_products",
                             note="Has order A1001 (TENT-2P), wants a bigger tent.")),
        products=products,
    )
    session = AgentSession(agent, models={"chat": model})
    result = await session.run("I ordered A1001 but I need a bigger tent")

    assert [n for n, _ in result.tool_calls] == ["transfer_to_products", "search_catalog"]
    assert result.answer == "TENT-4P: 3 in stock."
    values = (await session.state()).values
    assert values["active_agent"] == "products"
    assert values["handoff_notes"] == ["Has order A1001 (TENT-2P), wants a bigger tent."]
    # The products specialist read the note in its system prompt.
    assert products.prompts[0][0].text.endswith(
        "Notes from colleagues: Has order A1001 (TENT-2P), wants a bigger tent."
    )


async def test_the_next_turn_continues_with_the_active_specialist() -> None:
    model = _routes(
        orders=scripted(call("transfer_to_policies", note="Asks about returns.")),
        policies=scripted(call("get_policy", topic="returns"), "30 days.", "Yes, unused only."),
    )
    session = AgentSession(agent, models={"chat": model})
    await session.run("Can I return things?")
    second = await session.run("Even if I used it once?")

    # Only the new answer reaches the client: no earlier message is sent again.
    assert [p["text"] for p in second.payloads("message.completed")] == ["Yes, unused only."]
    assert second.tool_calls == []


def test_each_specialist_can_hand_off_to_every_other() -> None:
    for spec in SPECIALISTS:
        graph = build_desk_agent(spec, scripted())
        tools = set(graph.nodes["tools"].bound.tools_by_name)  # create_agent's tool node
        others = {f"transfer_to_{o.name}" for o in SPECIALISTS if o is not spec}
        assert tools == {t.name for t in spec.tools} | others, spec.name


async def test_why_nodes_return_only_new_messages() -> None:
    """Adding a specialist graph directly as a node returns the whole conversation each
    time, and vdreamer streams every AI message in an update: old answers come back."""

    class Direct(BaseAgent):
        name = "t23-direct"
        models = ("chat",)

        def build(self, ctx: BuildContext) -> Any:
            graph: StateGraph[Any] = StateGraph(DeskState)
            graph.add_node("orders", build_desk_agent(SPECIALISTS[0], ctx.models["chat"]))
            graph.add_edge(START, "orders")
            return graph

    session = AgentSession(Direct(), models={"chat": scripted("First answer.",
                                                              "Second answer.")})
    await session.run("one")
    second = await session.run("two")
    assert sorted(p["text"] for p in second.payloads("message.completed")) == [
        "First answer.", "Second answer."
    ]


def test_through_the_gateway(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        return {"models": {"chat": _routes(
            orders=scripted(call("transfer_to_products", note="Wants a bigger tent.")),
            products=scripted(call("search_catalog", query="family tent"),
                              "The family tent has 3 in stock."),
        )}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name, once=True, color=False,
                              message="I need a bigger tent than my A1001 one")
    assert result.code == 0, result.out
    assert '-> transfer_to_products({"note": "Wants a bigger tent."})' in result.out
    assert "The family tent has 3 in stock." in result.out
