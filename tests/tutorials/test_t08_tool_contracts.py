"""Lesson 8: reusable tools — a domain behind a protocol, a tool factory, idempotent writes."""

from __future__ import annotations

from typing import Any

import pytest
from _tutorial_kit import served, vdreamer_run
from langchain_core.tools import BaseTool
from vdreamer_tutorials.t08_tool_contracts.agent import InventoryAgent
from vdreamer_tutorials.t08_tool_contracts.inventory import (
    InMemoryInventory,
    InsufficientStock,
    StockLevel,
)
from vdreamer_tutorials.t08_tool_contracts.tools import make_inventory_tools
from vdreamer_tutorials.testing import AgentSession, call, scripted

from vdreamer import BuildContext, VDreamerContext, bind_context

# -- the domain, without any agent code --------------------------------------------------------


def test_reserving_is_idempotent_per_key() -> None:
    inventory = InMemoryInventory({"TENT-4P": 3})
    first = inventory.reserve("TENT-4P", 2, idempotency_key="k1")
    again = inventory.reserve("TENT-4P", 2, idempotency_key="k1")  # a retry
    assert again == first
    assert inventory.level("TENT-4P") == StockLevel(
        sku="TENT-4P", available_units=1, reserved_units=2
    )
    with pytest.raises(InsufficientStock):
        inventory.reserve("TENT-4P", 2, idempotency_key="k2")


# -- the tools: a contract over any Inventory ----------------------------------------------------


def _tools(units: dict[str, int]) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in make_inventory_tools(InMemoryInventory(units))}


def test_the_contract_the_model_sees() -> None:
    tools = _tools({"TENT-4P": 3})
    assert sorted(tools) == ["check_stock", "reserve_stock"]
    reserve = tools["reserve_stock"].tool_call_schema.model_json_schema()
    assert reserve["required"] == ["sku", "units"]
    assert reserve["properties"]["units"]["maximum"] == 10
    assert tools["reserve_stock"].metadata["vdreamer"]["side_effects"] is True
    assert tools["check_stock"].metadata["vdreamer"]["side_effects"] is False


def test_each_agent_instance_gets_its_own_inventory() -> None:
    a, b = _tools({"TENT-4P": 3}), _tools({"TENT-4P": 0})
    assert a["check_stock"].invoke({"sku": "TENT-4P"})["available_units"] == 3
    assert b["check_stock"].invoke({"sku": "TENT-4P"})["available_units"] == 0


def test_a_retried_call_reserves_once() -> None:
    """A worker that crashes after the tool ran re-executes the same checkpointed tool
    call. Governance gives that call the same idempotency key, so stock moves once."""
    inventory = InMemoryInventory({"TENT-4P": 3})
    (_, reserve) = BuildContext().tools.govern(make_inventory_tools(inventory))
    tool_call = {"type": "tool_call", "id": "call_7", "name": "reserve_stock",
                 "args": {"sku": "TENT-4P", "units": 2}}
    context = VDreamerContext(run_id="run_1", thread_id="t", agent_name="t08-inventory")

    with bind_context(context):
        first = reserve.invoke(tool_call)
        retry = reserve.invoke(tool_call)

    assert first.content == retry.content  # the same reservation, returned twice
    level = inventory.level("TENT-4P")
    assert level is not None and level.reserved_units == 2


# -- the agent --------------------------------------------------------------------------------


async def test_the_agent_checks_then_reserves() -> None:
    inventory = InMemoryInventory({"TENT-4P": 3})
    model = scripted(
        call("check_stock", sku="TENT-4P"),
        call("reserve_stock", sku="TENT-4P", units=2),
        "Reserved 2 units of TENT-4P, reservation R-0001.",
    )
    result = await AgentSession(InventoryAgent(inventory), models={"chat": model}).run(
        "Reserve 2 family tents (TENT-4P)"
    )

    assert [r["status"] for r in result.tool_results] == ["ok", "ok"]
    level = inventory.level("TENT-4P")
    assert level is not None and level.available_units == 1


async def test_domain_errors_become_actionable_tool_errors() -> None:
    model = scripted(call("reserve_stock", sku="TENT-4P", units=5), "Only 3 left.")
    result = await AgentSession(InventoryAgent(InMemoryInventory({"TENT-4P": 3})),
                                models={"chat": model}).run("Reserve 5")
    assert result.tool_results[0]["error"] == {
        "code": "tool_error",
        "message": "Cannot reserve: only 3 units of TENT-4P available. Offer fewer units.",
    }


def test_the_agent_policy_overrides_the_decorator() -> None:
    # The decorators say timeout_s=10; the agent's {"*_stock": Policy(timeout_s=5)} wins.
    governed = BuildContext(tool_policies=InventoryAgent.tool_policies).tools.govern(
        make_inventory_tools(InMemoryInventory({}))
    )
    assert [tool.policy.timeout_s for tool in governed] == [5, 5]  # type: ignore[attr-defined]


def test_through_the_gateway(monkeypatch: Any) -> None:
    agent = InventoryAgent(InMemoryInventory({"BAG-0C": 12}))

    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(call("reserve_stock", sku="BAG-0C", units=1),
                                            "Done: R-0001.")}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name,
                              message="Reserve one BAG-0C", once=True, color=False)
    assert result.code == 0, result.out
    assert '<- ok {"reservation_id": "R-0001", "sku": "BAG-0C", "units": 1}' in result.out
