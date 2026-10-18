"""Lesson 14: the transcript and working memory, both scoped to one thread."""

from __future__ import annotations

from typing import Any

from _tutorial_kit import served, vdreamer_run
from langchain_core.messages import HumanMessage, SystemMessage
from vdreamer_tutorials.t14_short_term_memory.agent import BASE_PROMPT, agent
from vdreamer_tutorials.testing import AgentSession, call, scripted


async def test_the_transcript_is_sent_again_every_turn() -> None:
    model = scripted("Nice to meet you, Ada.", "Your name is Ada.")
    session = AgentSession(agent, models={"chat": model})
    await session.run("Hi, I'm Ada")
    await session.run("What's my name?")

    second = model.prompts[1]
    assert [m.text for m in second if isinstance(m, HumanMessage)] == [
        "Hi, I'm Ada", "What's my name?"
    ]


async def test_a_remembered_preference_shapes_every_later_prompt() -> None:
    model = scripted(
        call("remember_preference", topic="language", value="German"), "Alles klar.",
        "Das Zelt kostet 149 Euro.",
    )
    session = AgentSession(agent, models={"chat": model})
    first = await session.run("From now on, please answer in German")
    await session.run("How much is the 2-person tent?")

    assert first.tool_calls == [("remember_preference", {"topic": "language", "value": "German"})]
    assert (await session.state()).values["preferences"] == {"language": "German"}
    system = model.prompts[2][0]
    assert isinstance(system, SystemMessage)
    assert system.text == f"{BASE_PROMPT} Always respect the customer's preferences — " \
                          "language: German."


async def test_preferences_accumulate_and_update() -> None:
    model = scripted(
        call("remember_preference", topic="language", value="German"), "OK.",
        call("remember_preference", topic="length", value="short"), "OK.",
        call("remember_preference", topic="Language", value="French"), "D'accord.",
    )
    session = AgentSession(agent, models={"chat": model})
    for message in ("German please", "Keep it short", "Actually, French"):
        await session.run(message)
    assert (await session.state()).values["preferences"] == {
        "language": "French", "length": "short"
    }


async def test_memory_ends_with_the_thread() -> None:
    model = scripted(call("remember_preference", topic="language", value="German"), "OK.", "Hi!")
    session = AgentSession(agent, models={"chat": model})
    await session.run("German please", thread_id="monday")
    await session.run("Hello again", thread_id="tuesday")

    assert model.prompts[2][0].text.startswith(BASE_PROMPT + " When the customer states")
    assert [m.text for m in model.prompts[2] if isinstance(m, HumanMessage)] == ["Hello again"]


def test_through_the_gateway_on_one_thread(monkeypatch: Any) -> None:
    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(
            call("remember_preference", topic="units", value="metric"), "Noted.",
            "It weighs 2.1 kg.",
        )}}

    with served([agent], {agent.name: backends}) as url:
        first = vdreamer_run(url, monkeypatch, agent=agent.name, message="Metric units please",
                             once=True, color=False)
        second = vdreamer_run(url, monkeypatch, agent=agent.name, thread=first.thread_id(),
                              message="How heavy is the tent?", once=True, color=False)
    assert first.code == 0 and second.code == 0, first.out + second.out
    assert '-> remember_preference({"topic": "units", "value": "metric"})' in first.out
    assert "It weighs 2.1 kg." in second.out
