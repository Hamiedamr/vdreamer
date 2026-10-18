"""Lesson 21: specialists — clear boundaries, their own tools, each testable alone."""

from __future__ import annotations

from typing import Any

import pytest
from _tutorial_kit import served, vdreamer_run
from vdreamer_tutorials.t21_specialists.agent import (
    SpecialistAgent,
    orders_agent,
    policies_agent,
    products_agent,
)
from vdreamer_tutorials.t21_specialists.specialists import SPECIALISTS, Specialist
from vdreamer_tutorials.testing import AgentSession, call, scripted


def test_every_tool_has_exactly_one_owner() -> None:
    owners: dict[str, str] = {}
    for spec in SPECIALISTS:
        for tool in spec.tools:
            assert tool.name not in owners, f"{tool.name} owned by {owners[tool.name]} too"
            owners[tool.name] = spec.name
    assert owners == {"get_order": "orders", "search_catalog": "products",
                      "get_policy": "policies"}


@pytest.mark.parametrize("spec", SPECIALISTS, ids=lambda s: s.name)
def test_each_specialist_states_its_boundary(spec: Specialist) -> None:
    assert "outside your area" in spec.prompt
    assert spec.responsibility.endswith(".") and len(spec.responsibility) < 100


@pytest.mark.parametrize(
    ("agent", "question", "tool_call", "fact"),
    [
        (orders_agent, "What is in A1001?", call("get_order", order_id="A1001"), "TENT-2P"),
        (products_agent, "Family tent in stock?", call("search_catalog", query="family tent"),
         '"stock": 3'),
        (policies_agent, "Returns?", call("get_policy", topic="returns"), "30 days"),
    ],
    ids=["orders", "products", "policies"],
)
async def test_each_specialist_alone(agent: SpecialistAgent, question: str, tool_call: Any,
                                     fact: str) -> None:
    model = scripted(tool_call, "Answer.")
    result = await AgentSession(agent, models={"chat": model}).run(question)

    assert model.bound_tools[0] == [t.name for t in agent.spec.tools]  # only its own tools
    assert fact in result.tool_results[0]["summary"]
    assert model.prompts[0][0].text == agent.spec.prompt


async def test_a_specialist_cannot_reach_another_specialists_tool() -> None:
    model = scripted(call("get_policy", topic="returns"), "That is outside my area.")
    result = await AgentSession(orders_agent, models={"chat": model}).run("Returns policy?")
    error = result.tool_results[0]["error"]
    assert error["code"] == "tool_error" and "get_policy" in error["message"]


def test_each_is_served_on_its_own(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(call("get_order", order_id="A1002"), "Processing.")}}

    with served([orders_agent], {orders_agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent="t21-orders", message="A1002?",
                              once=True, color=False)
    assert result.code == 0, result.out
    assert '"status": "processing"' in result.out
