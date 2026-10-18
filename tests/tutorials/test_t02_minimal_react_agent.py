"""Lesson 2: a minimal ReAct agent — create_agent, one tool, one model alias."""

from __future__ import annotations

from typing import Any

from _tutorial_kit import served, vdreamer_run
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from vdreamer_tutorials.t02_minimal_react_agent.agent import (
    SYSTEM_PROMPT,
    OrderStatusAgent,
    agent,
    get_order_status,
)
from vdreamer_tutorials.testing import AgentSession, call, scripted

from vdreamer import BuildContext
from vdreamer.validate import validate_agent


def test_the_tool_on_its_own() -> None:
    assert get_order_status.invoke({"order_id": " a1001 "}) == (
        "Order A1001: shipped on 2026-10-01, arriving 2026-10-04"
    )
    assert get_order_status.invoke({"order_id": "Z9"}) == "No order with id 'Z9' was found."


async def test_the_agent_calls_the_tool_then_answers() -> None:
    model = scripted(
        call("get_order_status", order_id="A1001"),
        "Your order A1001 shipped on October 1st and arrives on October 4th.",
    )
    result = await AgentSession(agent, models={"chat": model}).run("Where is order A1001?")

    assert result.status == "completed"
    assert result.tool_calls == [("get_order_status", {"order_id": "A1001"})]
    assert result.tool_results[0]["status"] == "ok"
    assert result.answer.startswith("Your order A1001 shipped")
    # Reason → act → observe: the second model call saw the tool's result.
    first_prompt, second_prompt = model.prompts
    assert isinstance(first_prompt[0], SystemMessage) and first_prompt[0].text == SYSTEM_PROMPT
    observation = second_prompt[-1]
    assert isinstance(observation, ToolMessage)
    assert observation.text == "Order A1001: shipped on 2026-10-01, arriving 2026-10-04"
    assert model.bound_tools[0] == ["get_order_status"]


async def test_without_an_order_id_the_model_can_answer_directly() -> None:
    model = scripted("Could you tell me your order id? It looks like A1001.")
    result = await AgentSession(agent, models={"chat": model}).run("Where is my parcel?")

    assert result.status == "completed"
    assert result.tool_calls == []
    assert "order id" in result.answer
    assert model.calls_made == 1


def test_through_the_gateway_with_vdreamer_run(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        model = scripted(call("get_order_status", order_id="A1002"), "It is being packed.")
        return {"models": {"chat": model}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name,
                              message="Where is A1002?", once=True, color=False)
    assert result.code == 0, result.out
    assert '-> get_order_status({"order_id": "A1002"}) [custom]' in result.out
    assert "<- ok Order A1002: processing: the warehouse is packing it" in result.out
    assert "It is being packed." in result.out


def test_skipping_agent_kwargs_fails_validation() -> None:
    class DocsStyle(OrderStatusAgent):
        """The LangChain docs' wiring: its own checkpointer, no vdreamer middleware."""

        name = "t02-docs-style"

        def build(self, ctx: BuildContext) -> Any:
            return create_agent(ctx.models["chat"], tools=[get_order_status],
                                system_prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver())

    checks = validate_agent(DocsStyle(), models={"chat": scripted()})
    failed = {c.name for c in checks if not c.ok}
    assert failed == {"checkpointer", "tools.governed"}
