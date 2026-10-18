"""Lesson 4: a custom tool — its contract (name, description, schema) and its behavior."""

from __future__ import annotations

import json
from typing import Any

import pytest
from _tutorial_kit import served, vdreamer_run
from langchain_core.messages import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from vdreamer_tutorials.t04_custom_tool.agent import agent, estimate_shipping
from vdreamer_tutorials.testing import AgentSession, call, scripted


def test_what_the_model_sees() -> None:
    spec = convert_to_openai_tool(estimate_shipping)["function"]

    assert spec["name"] == "estimate_shipping"
    assert spec["description"] == "Estimate the price and delivery time of shipping one parcel."
    params = spec["parameters"]
    assert params["required"] == ["weight_kg", "destination"]
    assert params["properties"]["weight_kg"]["type"] == "number"
    assert params["properties"]["destination"]["enum"] == ["domestic", "eu", "international"]
    assert params["properties"]["express"] == {
        "default": False,
        "description": "Use the express service: faster, but more expensive.",
        "type": "boolean",
    }


def test_the_tool_carries_a_vdreamer_policy() -> None:
    policy = estimate_shipping.metadata["vdreamer"]  # defaults: lesson 5 changes them
    assert policy["timeout_s"] == 60.0 and policy["max_result_chars"] == 8000


@pytest.mark.parametrize(
    ("args", "cost", "days"),
    [
        ({"weight_kg": 1, "destination": "domestic"}, 5.70, 2),
        ({"weight_kg": 2.5, "destination": "eu"}, 15.25, 4),
        ({"weight_kg": 2.5, "destination": "eu", "express": True}, 27.45, 2),
        ({"weight_kg": 10, "destination": "international", "express": True}, 142.20, 4),
    ],
)
def test_prices_and_delivery_days(args: dict[str, Any], cost: float, days: int) -> None:
    quote = estimate_shipping.invoke(args)
    assert quote["cost_eur"] == cost and quote["delivery_days"] == days


def test_called_with_a_tool_call_it_returns_json_in_a_tool_message() -> None:
    message = estimate_shipping.invoke({
        "type": "tool_call", "id": "call_1", "name": "estimate_shipping",
        "args": {"weight_kg": 2.5, "destination": "eu"},
    })
    assert isinstance(message, ToolMessage) and message.tool_call_id == "call_1"
    assert json.loads(message.text)["cost_eur"] == 15.25


async def test_the_agent_quotes_with_the_tool() -> None:
    model = scripted(
        call("estimate_shipping", weight_kg=2.5, destination="eu", express=True),
        "Express to Spain costs EUR 27.45 and takes about 2 days.",
    )
    result = await AgentSession(agent, models={"chat": model}).run(
        "How much to send 2.5 kg to Spain, fast?"
    )

    assert result.status == "completed"
    assert result.tool_calls == [
        ("estimate_shipping", {"weight_kg": 2.5, "destination": "eu", "express": True})
    ]
    assert json.loads(result.tool_results[0]["summary"])["cost_eur"] == 27.45
    assert model.bound_tools[0] == ["estimate_shipping"]
    assert "27.45" in result.answer


def test_through_the_gateway(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(
            call("estimate_shipping", weight_kg=1, destination="domestic"), "EUR 5.70, 2 days."
        )}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name,
                              message="1 kg, same country?", once=True, color=False)
    assert result.code == 0, result.out
    assert '"cost_eur": 5.7' in result.out and "EUR 5.70, 2 days." in result.out
