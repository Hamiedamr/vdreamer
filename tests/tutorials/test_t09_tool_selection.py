"""Lesson 9: what the model sees (by role), what may execute, and how often."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from _tutorial_kit import served, vdreamer_run
from vdreamer_tutorials.t09_tool_selection import agent as lesson
from vdreamer_tutorials.t09_tool_selection.agent import agent
from vdreamer_tutorials.testing import AgentSession, call, calls, scripted


@pytest.fixture(autouse=True)
def fresh_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lesson, "ORDERS", copy.deepcopy(lesson.ORDERS))


@pytest.mark.parametrize(
    ("roles", "offered"),
    [
        ((), ["get_order_status"]),
        (("staff",), ["get_order_status", "get_order_details", "issue_refund"]),
    ],
)
async def test_the_model_only_sees_tools_for_the_callers_roles(
    roles: tuple[str, ...], offered: list[str]
) -> None:
    model = scripted("How can I help?")
    await AgentSession(agent, models={"chat": model}, roles=roles).run("hello")
    assert model.bound_tools == [offered]


async def test_a_hidden_tool_is_still_refused_at_execution() -> None:
    # A model can still name a tool it was not offered (it saw it earlier, or guessed).
    model = scripted(call("issue_refund", order_id="A1001", reason="angry"), "I cannot.")
    result = await AgentSession(agent, models={"chat": model}).run("Refund A1001!")

    assert result.tool_results[0]["error"] == {
        "code": "denied", "message": "requires one of the roles ['staff']"
    }
    assert lesson.ORDERS["A1001"]["refunded"] is False


async def test_staff_can_refund_once_per_run() -> None:
    model = scripted(
        calls(("issue_refund", {"order_id": "A1001", "reason": "damaged"}),
              ("issue_refund", {"order_id": "A1002", "reason": "late"})),
        "Refunded one order; the second needs another request.",
    )
    result = await AgentSession(agent, models={"chat": model}, roles=("staff",)).run(
        "Refund A1001 and A1002"
    )

    statuses = sorted(r["status"] for r in result.tool_results)
    assert statuses == ["error", "ok"]
    blocked = [r for r in result.tool_results if r["status"] == "error"][0]
    assert "Tool call limit exceeded" in blocked["error"]["message"]
    assert sum(order["refunded"] for order in lesson.ORDERS.values()) == 1


async def test_the_limit_is_per_run_not_per_thread() -> None:
    model = scripted(
        call("issue_refund", order_id="A1001", reason="a"), "Done.",
        call("issue_refund", order_id="A1002", reason="b"), "Done again.",
    )
    session = AgentSession(agent, models={"chat": model}, roles=("staff",))
    first = await session.run("Refund A1001")
    second = await session.run("Refund A1002")
    assert first.tool_results[0]["status"] == second.tool_results[0]["status"] == "ok"


def test_roles_come_from_the_caller_through_the_gateway(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(
            call("get_order_details", order_id="A1002"), "A1002 totals EUR 89."
        )}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name, roles=("staff",),
                              message="Details of A1002", once=True, color=False)
    assert result.code == 0, result.out
    assert '"total_eur": 89.0' in result.out
