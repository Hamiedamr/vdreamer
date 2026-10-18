"""Lesson 1: the smallest agent vdreamer can serve — one node, no model.

There is no LLM here on purpose: the lesson is about the *serving* contract
(``BaseAgent``, ``build(ctx)``, threads, runs and events), and this agent runs
with zero configuration::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t01-echo \
        --input '{"messages": [{"role": "user", "content": "hello"}]}'
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode


class State(TypedDict):
    # add_messages appends new messages to the list instead of replacing it.
    messages: Annotated[list[AnyMessage], add_messages]


def respond(state: State) -> dict[str, Any]:
    """Echo the latest user message and count the turns in this thread."""
    vdreamer.emit_progress("reading your message", 50)
    question = state["messages"][-1].text
    turn = sum(isinstance(message, HumanMessage) for message in state["messages"])
    vdreamer.emit_log(f"turn {turn}: received {len(question)} characters")
    return {"messages": [AIMessage(f"You said: {question} (message {turn} in this thread)")]}


class EchoAgent(BaseAgent):
    name = "t01-echo"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> StateGraph:
        graph = StateGraph(State)
        graph.add_node("respond", respond)
        graph.add_edge(START, "respond")
        graph.add_edge("respond", END)
        return graph  # uncompiled: vdreamer compiles it with its checkpointer


agent = EchoAgent()
