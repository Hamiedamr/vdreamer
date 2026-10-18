"""Lesson 20: plan with a model, fan out with Send, combine — and reset per turn."""

from __future__ import annotations

from typing import Any

import pytest
from _tutorial_kit import served, vdreamer_run
from pydantic import ValidationError
from vdreamer_tutorials.t20_planning.agent import MAX_TASKS, Plan, Task, agent, do_task, gather
from vdreamer_tutorials.testing import AgentSession, call, scripted, structured

THREE = Plan(tasks=[Task(kind="order_status", order_id="A1001"),
                    Task(kind="return_check", order_id="A1003"),
                    Task(kind="policy", topic="warranty")])
QUESTION = "Where is A1001, can I still return A1003, and what is your warranty on boots?"


def test_workers_are_plain_functions() -> None:
    assert do_task(Task(kind="order_status", order_id="A1002")) == "Order A1002: processing"
    assert do_task(Task(kind="return_check", order_id="A1001")) == (
        "Order A1001: delivered 12 days ago, can be returned"
    )


def test_the_plan_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Plan(tasks=[Task(kind="policy", topic="returns")] * (MAX_TASKS + 1))


def test_the_resettable_reducer() -> None:
    assert gather(["a"], ["b"]) == ["a", "b"]
    assert gather(["a"], None) == []
    assert gather(["a"], []) == ["a"]  # why [] cannot reset an appending reducer


async def test_every_task_runs_and_the_answer_sees_all_results() -> None:
    model = scripted(structured(THREE), "Here is everything you asked.")
    result = await AgentSession(agent, models={"chat": model}).run(QUESTION)

    assert sorted(p["message"] for p in result.payloads("log")) == [  # parallel: any order
        "working on order_status A1001", "working on policy warranty",
        "working on return_check A1003",
    ]
    facts = model.prompts[1][0].text
    assert "A1001: shipped" in facts and "can no longer be returned" in facts
    assert "2-year warranty" in facts
    assert result.payloads("artifact.created")[0]["preview"]["body"]["tasks"][2]["topic"] == (
        "warranty"
    )
    assert result.answer == "Here is everything you asked."


async def test_no_tasks_means_a_direct_answer() -> None:
    model = scripted(structured(Plan(tasks=[])), "You're welcome!")
    result = await AgentSession(agent, models={"chat": model}).run("thanks!")
    assert result.payloads("log") == [] and result.answer == "You're welcome!"


async def test_an_unusable_plan_becomes_no_tasks() -> None:
    model = scripted(call("Plan", tasks=[{"kind": "refund"}]), "Let me help.")
    result = await AgentSession(agent, models={"chat": model}).run("refund me")
    assert result.payloads("progress")[0]["label"] == "planned 0 tasks"


async def test_results_do_not_leak_into_the_next_turn() -> None:
    model = scripted(structured(THREE), "All answered.", structured(Plan(tasks=[])), "Bye!")
    session = AgentSession(agent, models={"chat": model})
    await session.run(QUESTION)
    await session.run("thanks, bye")

    assert (await session.state()).values["results"] == []
    assert "- (none)" in model.prompts[3][0].text


def test_through_the_gateway(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(structured(THREE), "Three answers in one.")}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name, message=QUESTION, once=True,
                              color=False)
    assert result.code == 0, result.out
    assert "[artifact] json 'plan'" in result.out and "[progress] planned 3 tasks" in result.out
    assert result.out.count("[log info] working on") == 3
    assert "Three answers in one." in result.out
