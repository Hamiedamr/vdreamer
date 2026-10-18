"""Lesson 13: typed state, reducers, checkpoints and per-thread history."""

from __future__ import annotations

from typing import Any

import httpx
from _tutorial_kit import TOKEN, served, vdreamer_run
from langchain_core.messages import HumanMessage
from vdreamer_tutorials.t13_state_and_history.agent import agent, extract, facts, merge_unique
from vdreamer_tutorials.testing import AgentSession, scripted


def test_the_merge_reducer() -> None:
    assert merge_unique(["refunds"], ["delivery", "refunds"]) == ["refunds", "delivery"]
    assert merge_unique([], []) == []


def test_extract_is_plain_bookkeeping() -> None:
    update = extract({"messages": [HumanMessage("I'm Ada@Example.com, refund please")]})
    assert update == {"turn": 1, "topics": ["refunds"], "customer_email": "ada@example.com"}
    assert "customer_email" not in extract({"messages": [HumanMessage("no email here")],
                                            "turn": 1})
    # Found by the multi-turn test below: a sentence's full stop is not part of the address.
    ended = extract({"messages": [HumanMessage("Use ada.lovelace@example.com.")]})
    assert ended["customer_email"] == "ada.lovelace@example.com"


async def test_state_accumulates_across_runs_on_a_thread() -> None:
    model = scripted("Noted.", "It ships tomorrow.", "Your refund is on its way.")
    session = AgentSession(agent, models={"chat": model})
    await session.run("Hi, I'm ada@example.com, a refund question")
    await session.run("When will my new order be delivered?")
    await session.run("Actually use ada.lovelace@example.com. And the refund?")

    values = (await session.state()).values
    assert values["turn"] == 3
    assert values["topics"] == ["refunds", "delivery"]  # merged by the reducer
    assert values["customer_email"] == "ada.lovelace@example.com"  # overwritten
    assert len(values["messages"]) == 6  # appended by add_messages
    # The model was told the facts gathered so far, every turn.
    assert "Topics so far: refunds, delivery." in model.prompts[2][0].text


async def test_threads_are_isolated() -> None:
    session = AgentSession(agent, models={"chat": scripted("a", "b")})
    await session.run("ada@example.com, refund", thread_id="t-ada")
    await session.run("delivery question", thread_id="t-lin")

    lin = (await session.state("t-lin")).values
    assert lin["turn"] == 1 and lin["topics"] == ["delivery"]
    assert lin.get("customer_email") is None


async def test_every_step_is_a_checkpoint() -> None:
    session = AgentSession(agent, models={"chat": scripted("a", "b")})
    await session.run("refund")
    await session.run("delivery")

    config = {"configurable": {"thread_id": "thread-1"}}
    history = [snap async for snap in session.runner.compiled.aget_state_history(config)]
    # Per run: the input, extract, respond, and the final state — newest first.
    assert [snap.next for snap in history[:4]] == [(), ("respond",), ("extract",), ("__start__",)]
    assert len(history) == 8
    before_second_run = history[4].values
    assert before_second_run["turn"] == 1 and before_second_run["topics"] == ["refunds"]


def test_facts_default_when_state_is_empty() -> None:
    assert facts({"messages": []}) == (
        "Turn 1 of this conversation. Customer email: unknown. Topics so far: none."
    )


def test_the_gateway_exposes_the_threads_state(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted("Noted, ada.")}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name, once=True, color=False,
                              message="I'm ada@example.com with an invoice question")
        thread = httpx.get(f"{url}/api/v1/threads/{result.thread_id()}",
                           headers={"Authorization": f"Bearer {TOKEN}", "X-User-Id": "alice"})
    assert result.code == 0, result.out
    values = thread.json()["values"]
    assert values["customer_email"] == "ada@example.com" and values["topics"] == ["billing"]
    assert "ada@example.com" not in result.out  # state is private: no event carried it
