"""Lesson 16: keeping a long conversation inside the model's context window.

When a thread has more than ``max_turns`` user turns, the oldest turns are
summarised by a background model call, the summary is kept in its own state key
and rendered into the system prompt, and the old messages are deleted from the
thread by id. Cutting only at a user message never separates a tool call from
its result, and nothing already shown to the client is re-sent::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t16-long-conversation \
        --models config/models.ollama.yaml
    uv run vdreamer run t16-long-conversation
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, before_model, dynamic_prompt
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.runtime import Runtime

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode

BASE_PROMPT = "You are the order assistant of an outdoor-gear shop."
SUMMARY_INSTRUCTIONS = (
    "Summarise the conversation below for the assistant who will continue it. Keep order "
    "ids, decisions, open questions and customer preferences; drop small talk. Plain text, "
    "at most 120 words."
)
ORDERS = {"A1001": "shipped", "A1002": "processing", "A1003": "delivered"}


@vdreamer.tool
def get_order_status(order_id: str) -> str:
    """Look up the status of an order by its id, for example "A1001"."""
    return f"Order {order_id}: {ORDERS.get(order_id.strip().upper(), 'not found')}"


class SummarizedState(AgentState[Any]):
    summary: NotRequired[str]


def background_model(model: BaseChatModel) -> BaseChatModel:
    """A copy of ``model`` whose calls are not streamed to clients.

    Every chat-model call inside a run streams to the client as ``token.delta``
    events, a summariser's included. LangGraph skips calls tagged ``nostream``.
    """
    return model.model_copy(update={"tags": [*(model.tags or []), "nostream"]})


def turn_cut(messages: Sequence[AnyMessage], keep_turns: int) -> int:
    """Index where the last ``keep_turns`` user turns start (0: nothing to cut).

    Cutting at a user message keeps every tool call with its result.
    """
    starts = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    return starts[-keep_turns] if len(starts) > keep_turns else 0


class LongConversationAgent(BaseAgent):
    name = "t16-long-conversation"
    mode = Mode.CHAT
    models = ("chat",)

    def __init__(self, max_turns: int = 12, keep_turns: int = 4) -> None:
        if not 0 < keep_turns < max_turns:
            raise ValueError("need 0 < keep_turns < max_turns")
        self.max_turns = max_turns
        self.keep_turns = keep_turns

    def build(self, ctx: BuildContext) -> Any:
        summarizer = background_model(ctx.models["chat"])

        @before_model
        async def summarize_old_turns(
            state: SummarizedState, runtime: Runtime[Any]
        ) -> dict[str, Any] | None:
            messages = state["messages"]
            turns = sum(isinstance(m, HumanMessage) for m in messages)
            if turns <= self.max_turns:
                return None
            old = messages[: turn_cut(messages, self.keep_turns)]
            earlier = state.get("summary")
            transcript = [SystemMessage(SUMMARY_INSTRUCTIONS)]
            if earlier:
                transcript.append(HumanMessage(f"Summary of what came before: {earlier}"))
            summary = await summarizer.ainvoke([*transcript, *old])
            vdreamer.emit_log(f"summarised {len(old)} earlier messages")
            return {
                "summary": summary.text,
                "messages": [RemoveMessage(id=m.id) for m in old if m.id is not None],
            }

        @dynamic_prompt
        def prompt_with_summary(request: ModelRequest) -> str:
            summary = request.state.get("summary")
            return f"{BASE_PROMPT}\nEarlier in this conversation: {summary}" if summary \
                else BASE_PROMPT

        kwargs = ctx.agent_kwargs()
        return create_agent(
            ctx.models["chat"],
            tools=[get_order_status],
            state_schema=SummarizedState,
            middleware=[*kwargs.pop("middleware"), summarize_old_turns, prompt_with_summary],
            **kwargs,
        )


agent = LongConversationAgent()
