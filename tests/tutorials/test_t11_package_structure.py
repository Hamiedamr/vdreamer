"""Lesson 11: the helpdesk package — each layer tested alone, plus its import rules."""

from __future__ import annotations

import ast
from datetime import date
from typing import Any

import pytest
from _tutorial_kit import LESSONS, served, vdreamer_run
from vdreamer_tutorials.t11_package_structure.helpdesk import (
    HelpdeskAgent,
    HelpdeskSettings,
    InMemoryOrders,
    Order,
)
from vdreamer_tutorials.t11_package_structure.helpdesk.domain import can_cancel, demo_orders
from vdreamer_tutorials.t11_package_structure.helpdesk.prompts import system_prompt
from vdreamer_tutorials.t11_package_structure.helpdesk.tools import make_tools
from vdreamer_tutorials.testing import AgentSession, call, scripted

PACKAGE = LESSONS / "t11_package_structure" / "helpdesk"

#: module → the sibling modules and outside libraries it may import.
ALLOWED = {
    "domain": {"pydantic"},
    "settings": set(),
    "prompts": {"settings"},
    "tools": {"domain", "settings", "langchain_core", "vdreamer"},
    "agent": {"domain", "settings", "prompts", "tools", "langchain", "vdreamer"},
}
STDLIB = {"__future__", "dataclasses", "datetime", "typing"}


def _imports(module: str) -> set[str]:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            found.add(node.module.split(".")[0] if node.level == 0 and node.module
                      else str(node.module))
    return found - STDLIB


@pytest.mark.parametrize("module", sorted(ALLOWED))
def test_each_layer_imports_only_what_is_below_it(module: str) -> None:
    assert _imports(module) <= ALLOWED[module], module


# -- each layer alone ----------------------------------------------------------------------------


def test_domain_rules() -> None:
    orders = demo_orders()
    assert [o.order_id for o in orders.for_customer(" ADA@example.com ")] == ["A1002", "A1001"]
    shipped, processing = orders.get("a1001"), orders.get("A1002")
    assert shipped is not None and processing is not None
    assert not can_cancel(shipped) and can_cancel(processing)


def test_the_prompt_comes_from_settings() -> None:
    assert "Example Gear Co" in system_prompt(HelpdeskSettings(shop_name="Example Gear Co"))


def test_tools_over_a_custom_repository_and_settings() -> None:
    orders = InMemoryOrders([
        Order(order_id=f"B{i}", customer_email="x@example.com", status="processing",
              total_eur=1.0, placed_on=date(2026, 1, i)) for i in range(1, 9)
    ])
    tools = {t.name: t for t in make_tools(orders, HelpdeskSettings(max_orders_listed=2))}
    assert sorted(tools) == ["can_cancel", "get_order", "list_orders"]
    assert [o["order_id"] for o in tools["list_orders"].invoke({"email": "x@example.com"})] == [
        "B8", "B7"
    ]


# -- assembled ------------------------------------------------------------------------------------


async def test_the_assembled_agent() -> None:
    model = scripted(call("can_cancel", order_id="A1002"), "Yes, A1002 can still be cancelled.")
    result = await AgentSession(HelpdeskAgent(), models={"chat": model}).run("Cancel A1002?")

    assert result.tool_results[0]["summary"] == (
        "Order A1002 can be cancelled (status: processing)."
    )
    assert "Trailhead Outfitters" in model.prompts[0][0].text


def test_through_the_gateway(monkeypatch: Any) -> None:
    agent = HelpdeskAgent(settings=HelpdeskSettings(shop_name="Example Gear Co"))

    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(call("get_order", order_id="A9"), "Not found.")}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name, message="A9?",
                              once=True, color=False)
    assert result.code == 0, result.out
    assert "Error (tool_error): No order A9 exists." in result.out
