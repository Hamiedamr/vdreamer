"""Lesson 13: state beyond messages — typed fields, reducers, and per-thread history.

The agent keeps three things per thread besides the conversation: the
customer's email (the latest one wins), the topics raised so far (merged), and
a turn counter. A deterministic ``extract`` node fills them; the model then
answers with those facts in its system prompt::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t13-support-thread \
        --models config/models.ollama.yaml
    uv run vdreamer run t13-support-thread --message "Hi, I'm ada@example.com, a refund question"
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict

from vdreamer import BaseAgent, BuildContext, Mode

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")  # no trailing "."
TOPIC_WORDS = {
    "refund": "refunds", "return": "returns", "deliver": "delivery", "ship": "delivery",
    "invoice": "billing", "charge": "billing", "warranty": "warranty",
}


def merge_unique(existing: list[str], new: list[str]) -> list[str]:
    """A reducer: keep every topic once, in the order first raised."""
    return existing + [topic for topic in new if topic not in existing]


class SupportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # reducer: append (or replace by id)
    topics: NotRequired[Annotated[list[str], merge_unique]]  # reducer: union
    customer_email: NotRequired[str | None]  # no reducer: the latest write wins
    turn: NotRequired[int]


def extract(state: SupportState) -> dict[str, Any]:
    """Deterministic bookkeeping: no model needed to count turns or spot an email."""
    latest = state["messages"][-1].text
    update: dict[str, Any] = {
        "turn": state.get("turn", 0) + 1,
        "topics": [topic for word, topic in TOPIC_WORDS.items() if word in latest.lower()],
    }
    if found := EMAIL.search(latest):
        update["customer_email"] = found.group(0).lower()
    return update


def facts(state: SupportState) -> str:
    """What the model is told about this thread, from state rather than re-reading history."""
    return (
        f"Turn {state.get('turn', 1)} of this conversation. "
        f"Customer email: {state.get('customer_email') or 'unknown'}. "
        f"Topics so far: {', '.join(state.get('topics', [])) or 'none'}."
    )


class SupportThreadAgent(BaseAgent):
    name = "t13-support-thread"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> StateGraph:
        model = ctx.models["chat"]

        async def respond(state: SupportState) -> dict[str, Any]:
            system = SystemMessage(
                "You are a support agent for an online shop. Be brief. " + facts(state)
            )
            return {"messages": [await model.ainvoke([system, *state["messages"]])]}

        graph = StateGraph(SupportState)
        graph.add_node("extract", extract)
        graph.add_node("respond", respond)
        graph.add_edge(START, "extract")
        graph.add_edge("extract", "respond")
        graph.add_edge("respond", END)
        return graph


agent = SupportThreadAgent()
