"""Lesson 7: four tools — each one alone, the set as a whole, and the agent combining them."""

from __future__ import annotations

import json
from typing import Any

import pytest
from _tutorial_kit import served, vdreamer_run
from langchain_core.messages import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from vdreamer_tutorials.t07_multiple_tools.agent import (
    TOOLS,
    agent,
    check_stock,
    get_policy,
    search_products,
)
from vdreamer_tutorials.testing import AgentSession, call, calls, scripted

# -- each tool on its own ----------------------------------------------------------------------


def test_search_matches_every_keyword() -> None:
    assert [p["sku"] for p in search_products.invoke({"query": "tent"})] == ["TENT-2P", "TENT-4P"]
    assert [p["sku"] for p in search_products.invoke({"query": "family tent"})] == ["TENT-4P"]
    assert search_products.invoke({"query": "camping", "max_results": 2}) == [
        {"sku": "TENT-2P", "name": "Trail tent, 2-person", "price_eur": 149.0},
        {"sku": "TENT-4P", "name": "Family tent, 4-person", "price_eur": 289.0},
    ]
    assert search_products.invoke({"query": "kayak"}) == []


@pytest.mark.parametrize(
    ("sku", "answer"),
    [
        ("TENT-4P", "TENT-4P: 3 in stock."),
        ("tent-2p", "TENT-2P: out of stock, expected back on 2026-10-20."),
        ("2-person tent",
         "Unknown SKU '2-PERSON TENT'. Use search_products to find the right SKU."),
    ],
)
def test_stock_answers(sku: str, answer: str) -> None:
    assert check_stock.invoke({"sku": sku}) == answer


# -- the tool set as a whole -------------------------------------------------------------------


def test_the_tool_set_is_unambiguous() -> None:
    specs = [convert_to_openai_tool(tool)["function"] for tool in TOOLS]
    names = [spec["name"] for spec in specs]
    assert len(set(names)) == len(names)
    assert len({spec["description"] for spec in specs}) == len(specs)
    for spec in specs:  # every argument explains itself to the model
        for name, prop in spec["parameters"]["properties"].items():
            assert prop.get("description"), f"{spec['name']}.{name} has no description"


# -- the agent combining them ------------------------------------------------------------------


async def test_a_product_name_takes_search_then_stock() -> None:
    model = scripted(
        call("search_products", query="2-person tent"),
        call("check_stock", sku="TENT-2P"),
        "The 2-person trail tent is out of stock; it is expected back on 20 October.",
    )
    result = await AgentSession(agent, models={"chat": model}).run(
        "Is the 2-person tent in stock?"
    )

    assert [name for name, _ in result.tool_calls] == ["search_products", "check_stock"]
    # The SKU used in step 2 came from what the model observed in step 1.
    found = json.loads([m for m in model.prompts[1] if isinstance(m, ToolMessage)][-1].text)
    assert found[0]["sku"] == result.tool_calls[1][1]["sku"] == "TENT-2P"
    assert model.bound_tools[0] == [
        "search_products", "check_stock", "get_order_status", "get_policy"
    ]


async def test_independent_lookups_in_one_turn() -> None:
    model = scripted(
        calls(("check_stock", {"sku": "TENT-4P"}), ("check_stock", {"sku": "BAG-0C"}),
              ("get_policy", {"topic": "shipping"})),
        "Both are in stock, and the order ships free.",
    )
    result = await AgentSession(agent, models={"chat": model}).run(
        "Do you have TENT-4P and BAG-0C, and is shipping free?"
    )

    assert model.calls_made == 2  # three tools, one round trip
    assert [r["status"] for r in result.tool_results] == ["ok", "ok", "ok"]
    observations = sorted(m.text for m in model.prompts[1] if isinstance(m, ToolMessage))
    assert observations == sorted([
        "TENT-4P: 3 in stock.", "BAG-0C: 12 in stock.",
        get_policy.invoke({"topic": "shipping"}),
    ])


async def test_an_unknown_policy_topic_is_rejected_by_the_schema() -> None:
    model = scripted(call("get_policy", topic="privacy"), "I can only quote three policies.")
    result = await AgentSession(agent, models={"chat": model}).run("Privacy policy?")

    assert result.tool_results[0]["error"]["code"] == "invalid_args"
    assert result.status == "completed"


def test_through_the_gateway(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(
            call("search_products", query="lamp"), call("check_stock", sku="LAMP-USB"),
            "Yes, 40 rechargeable lamps are in stock at EUR 24.50.",
        )}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name,
                              message="Any camping lamps in stock?", once=True, color=False)
    assert result.code == 0, result.out
    assert "-> search_products(" in result.out and "-> check_stock(" in result.out
    assert "<- ok LAMP-USB: 40 in stock." in result.out
