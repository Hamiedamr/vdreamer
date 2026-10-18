"""Lesson 5: validated inputs, contained errors, timeouts, redaction, bounded outputs."""

from __future__ import annotations

import json
from typing import Any, TypedDict

import pytest
from _tutorial_kit import served, vdreamer_run
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from vdreamer_tutorials.t05_tool_errors import agent as lesson
from vdreamer_tutorials.t05_tool_errors.agent import OrderDeskAgent, agent, get_order
from vdreamer_tutorials.testing import AgentSession, RunResult, call, scripted

from vdreamer import BaseAgent, BuildContext, Policy


async def _one_call(tool_call: Any, then: str = "ok", target: BaseAgent = agent
                    ) -> tuple[RunResult, ToolMessage]:
    """Run one scripted tool call; return the run and the observation the model got."""
    model = scripted(tool_call, then)
    result = await AgentSession(target, models={"chat": model}).run("help")
    observation = model.prompts[1][-1]
    assert isinstance(observation, ToolMessage)
    return result, observation


async def test_a_valid_lookup() -> None:
    result, seen = await _one_call(call("get_order", order_id="A1001", email="ada@example.com"))
    assert result.tool_results[0]["status"] == "ok"
    assert json.loads(seen.text) == {
        "order_id": "A1001", "status": "shipped", "tracking_id": "TRK-1001"
    }


async def test_invalid_arguments_are_rejected_before_the_tool_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lesson, "ORDERS_DB_ONLINE", False)  # would raise if the tool ran
    result, seen = await _one_call(call("get_order", order_id="1001", email="ada"))

    assert result.status == "completed"  # contained: the model gets to try again
    error = result.tool_results[0]["error"]
    assert error["code"] == "invalid_args"
    assert "order_id: String should match pattern" in error["message"]
    assert seen.status == "error" and seen.text.startswith("Error (invalid_args):")


async def test_a_tool_exception_is_an_explanation_for_the_model() -> None:
    result, seen = await _one_call(call("get_order", order_id="A9999", email="ada@example.com"))
    assert result.tool_results[0]["error"] == {
        "code": "tool_error", "message": "No order A9999 was found for that email address."
    }
    assert seen.text == "Error (tool_error): No order A9999 was found for that email address."


async def test_an_unexpected_failure_is_contained_and_its_secret_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lesson, "ORDERS_DB_ONLINE", False)
    result, seen = await _one_call(call("get_order", order_id="A1001", email="ada@example.com"))

    assert result.status == "completed"
    assert seen.text == "Error (tool_error): orders-db login failed: password=***"
    assert "hunter2" not in json.dumps([e.payload for e in result.events])


async def test_a_hanging_call_times_out() -> None:
    result, seen = await _one_call(call("track_parcel", tracking_id="TRK-SLOW"))
    assert result.tool_results[0]["error"]["code"] == "timeout"
    assert seen.text == "Error (timeout): the tool did not finish within 1s"


async def test_redacted_arguments_never_reach_the_events() -> None:
    result, _ = await _one_call(call("get_order", order_id="A1001", email="ada@example.com"))
    assert result.tool_calls == [("get_order", {"order_id": "A1001", "email": "***"})]
    assert "ada@example.com" not in json.dumps([e.payload for e in result.events])


async def test_an_oversized_result_is_truncated_and_kept_as_an_artifact() -> None:
    result, seen = await _one_call(call("order_history", email="ada@example.com"))

    (artifact,) = result.payloads("artifact.created")
    assert artifact["kind"] == "file" and artifact["name"] == "order_history result"
    assert seen.text.startswith("2026-09-30: order event #30")
    assert "result truncated to 400 of 1370 characters" in seen.text
    assert artifact["artifact_id"] in seen.text  # the model is told where the rest is


async def test_fail_run_turns_a_contained_error_into_a_failed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Strict(OrderDeskAgent):
        name = "t05-strict"
        tool_policies = {"get_order": Policy(fail_run=True)}  # lesson 8: agent-level policies

    monkeypatch.setattr(lesson, "ORDERS_DB_ONLINE", False)
    model = scripted(call("get_order", order_id="A1001", email="ada@example.com"))
    result = await AgentSession(Strict(), models={"chat": model}).run("help")

    assert result.status == "failed"
    error = result.payloads("run.status")[-1]["error"]
    assert "password=***" in error["message"] and "hunter2" not in error["message"]


class _State(TypedDict, total=False):
    result: str


def _direct_caller(governed: bool) -> BaseAgent:
    """A one-node graph that calls get_order itself — through governance or not."""

    class DirectCaller(BaseAgent):
        name = "t05-governed" if governed else "t05-ungoverned"

        def build(self, ctx: BuildContext) -> StateGraph[Any]:
            (tool,) = ctx.tools.govern([get_order]) if governed else (get_order,)

            async def lookup(state: _State) -> dict[str, Any]:
                output = await tool.ainvoke({"order_id": "A1001", "email": "ada@example.com"})
                return {"result": str(output)}

            graph: StateGraph[Any] = StateGraph(_State)
            graph.add_node("lookup", lookup)
            graph.add_edge(START, "lookup")
            graph.add_edge("lookup", END)
            return graph

    return DirectCaller()


async def test_without_governance_the_same_failure_kills_the_run_and_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lesson, "ORDERS_DB_ONLINE", False)
    raw = await AgentSession(_direct_caller(governed=False)).run({})
    governed = await AgentSession(_direct_caller(governed=True)).run({})

    assert raw.status == "failed"
    assert "password=hunter2" in raw.payloads("run.status")[-1]["error"]["message"]
    assert governed.status == "completed"
    assert governed.values["result"] == "Error (tool_error): orders-db login failed: password=***"


def test_through_the_gateway_the_artifact_is_stored(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(
            call("order_history", email="ada@example.com"), "Here are the latest events."
        )}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name,
                              message="My history, ada@example.com", once=True, color=False)
    assert result.code == 0, result.out
    assert '-> order_history({"email": "***"})' in result.out
    assert "[artifact] file 'order_history result'" in result.out
    assert "ada@example.com" not in result.out  # vdreamer run shows events, not the input
