"""Lesson 16: summarising old turns, keeping recent ones, and what clients see."""

from __future__ import annotations

import json
from typing import Any

from _tutorial_kit import served, vdreamer_run
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from vdreamer_tutorials.t16_context_management.agent import (
    BASE_PROMPT,
    LongConversationAgent,
    turn_cut,
)
from vdreamer_tutorials.testing import AgentSession, ScriptedChatModel, call, scripted

from vdreamer import BaseAgent, BuildContext

SUMMARY = "Asked about A1001 (shipped) and A1002 (processing)."


def test_the_cut_falls_on_a_user_message() -> None:
    history = [HumanMessage("a"), AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
               ToolMessage("r", tool_call_id="1"), AIMessage("x"),
               HumanMessage("b"), AIMessage("y"), HumanMessage("c")]
    assert turn_cut(history, 2) == 4
    assert turn_cut(history, 3) == 0  # only three turns: nothing to cut


async def _four_turns(model: ScriptedChatModel) -> tuple[AgentSession, list[Any]]:
    session = AgentSession(LongConversationAgent(max_turns=3, keep_turns=1),
                           models={"chat": model})
    results = [await session.run(q) for q in ("q1", "q2", "q3", "q4")]
    return session, results


async def test_old_turns_become_a_summary_in_the_system_prompt() -> None:
    # Turn 4 crosses max_turns: one summary call (scripted 4th), then the answer.
    model = scripted("one", "two", "three", SUMMARY, "four")
    session, _ = await _four_turns(model)

    values = (await session.state()).values
    assert values["summary"] == SUMMARY
    assert [m.text for m in values["messages"]] == ["q4", "four"]
    answering = model.prompts[4]
    assert answering[0].text == f"{BASE_PROMPT}\nEarlier in this conversation: {SUMMARY}"
    assert [m.text for m in answering[1:]] == ["q4"]
    # The summariser saw its instructions, then the old turns.
    assert [m.text for m in model.prompts[3][1:]] == ["q1", "one", "q2", "two", "q3", "three"]


async def test_clients_see_only_the_new_answer() -> None:
    _, results = await _four_turns(scripted("one", "two", "three", SUMMARY, "four"))
    last = results[-1]
    assert [p["text"] for p in last.payloads("message.completed")] == ["four"]
    assert SUMMARY not in json.dumps([e.payload for e in last.events])
    assert last.payloads("log") == [{"level": "info", "message": "summarised 6 earlier messages"}]


async def test_summaries_fold_into_the_next_summary() -> None:
    # max_turns=2: turn 3 summarises turns 1-2 (S1); turn 5 summarises turns 3-4 plus S1.
    model = scripted("1", "2", "S1", "3", "4", "S2", "5")
    session = AgentSession(LongConversationAgent(max_turns=2, keep_turns=1),
                           models={"chat": model})
    for q in ("q1", "q2", "q3", "q4", "q5"):
        await session.run(q)

    second_summary_prompt = model.prompts[5]
    assert second_summary_prompt[1].text == "Summary of what came before: S1"
    assert [m.text for m in second_summary_prompt[2:]] == ["q3", "3", "q4", "4"]
    assert (await session.state()).values["summary"] == "S2"


async def test_tool_calls_stay_with_their_results() -> None:
    model = scripted(
        call("get_order_status", order_id="A1001"), "A1001 shipped.",
        call("get_order_status", order_id="A1002"), "A1002 is processing.",
        "Welcome.", SUMMARY, "Anything else?",
    )
    session = AgentSession(LongConversationAgent(max_turns=3, keep_turns=2),
                           models={"chat": model})
    for question in ("A1001?", "A1002?", "hi", "thanks"):
        await session.run(question)

    messages = (await session.state()).values["messages"]
    calls = {c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    results = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    assert calls == results


async def test_langchains_summarization_middleware_shows_why_this_lesson_does_it_itself() -> None:
    """The prebuilt middleware works, but rewrites the message list: kept AI messages are
    re-sent to clients, and without a `nostream` model the summary streams as deltas."""

    class Prebuilt(BaseAgent):
        name = "t16-prebuilt"
        models = ("chat",)

        def build(self, ctx: BuildContext) -> Any:
            kwargs = ctx.agent_kwargs()
            summarize = SummarizationMiddleware(ctx.models["chat"], trigger=("messages", 6),
                                                keep=("messages", 2))
            return create_agent(ctx.models["chat"], tools=[],
                                middleware=[*kwargs.pop("middleware"), summarize], **kwargs)

    session = AgentSession(Prebuilt(),
                           models={"chat": scripted("one", "two", "three", SUMMARY, "four")})
    results = [await session.run(q) for q in ("q1", "q2", "q3", "q4")]
    last = results[-1]
    assert [p["text"] for p in last.payloads("message.completed")] == ["three", "four"]
    assert SUMMARY in [p["text"] for p in last.payloads("token.delta")]


def test_through_the_gateway(monkeypatch: Any) -> None:
    agent = LongConversationAgent(max_turns=2, keep_turns=1)

    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted("one", "two", SUMMARY, "three")}}

    with served([agent], {agent.name: backends}) as url:
        first = vdreamer_run(url, monkeypatch, agent=agent.name, message="q1", once=True,
                             color=False)
        thread = first.thread_id()
        vdreamer_run(url, monkeypatch, agent=agent.name, thread=thread, message="q2",
                     once=True, color=False)
        third = vdreamer_run(url, monkeypatch, agent=agent.name, thread=thread, message="q3",
                             once=True, color=False)
    assert "three" in third.out and "two" not in third.out and SUMMARY not in third.out
    assert "[log info] summarised 4 earlier messages" in third.out
