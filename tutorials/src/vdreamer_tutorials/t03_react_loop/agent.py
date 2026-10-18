"""Lesson 3: the ReAct loop built by hand as a LangGraph ``StateGraph``.

The same behavior ``create_agent`` gives you in lesson 2, with every step
visible: ``reason`` (call the model), ``act`` (run the requested tools and
record what they returned), and a routing function that decides whether to
loop, answer, or stop because the step budget ran out::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t03-react-loop \
        --models config/models.ollama.yaml
    uv run vdreamer run t03-react-loop --message "Compare orders A1001 and A1003"
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode

ORDERS = {
    "A1001": "shipped on 2026-10-01, arriving 2026-10-04",
    "A1002": "processing: the warehouse is packing it",
    "A1003": "delivered on 2026-09-28",
}

SYSTEM_PROMPT = (
    "You are the order-support assistant of an online shop. Use the get_order_status "
    "tool for every order the customer mentions, one call per order. Never invent a status."
)

#: How many model calls one user turn may take before the agent gives up.
MAX_STEPS = 4


@vdreamer.tool
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an order by its id, for example "A1001"."""
    key = order_id.strip().upper()
    status = ORDERS.get(key)
    return f"Order {key}: {status}" if status else f"No order with id {order_id!r} was found."


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def steps_this_turn(messages: Sequence[AnyMessage]) -> int:
    """Model replies since the latest user message (the thread keeps earlier turns)."""
    steps = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage):
            steps += 1
    return steps


class ReactLoopAgent(BaseAgent):
    name = "t03-react-loop"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> StateGraph:
        tools = ctx.tools.govern([get_order_status])  # governed: events, timeouts, containment
        tools_by_name = {tool.name: tool for tool in tools}
        model = ctx.models["chat"].bind_tools(tools)  # the model sees the tools' schemas

        async def reason(state: State) -> dict[str, Any]:
            """Reason: the model reads the conversation and answers or asks for tools."""
            response = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
            return {"messages": [response]}

        async def act(state: State) -> dict[str, Any]:
            """Act and observe: run each requested tool; each result becomes a ToolMessage."""
            request = state["messages"][-1]
            assert isinstance(request, AIMessage)
            observations: list[ToolMessage] = []
            for tool_call in request.tool_calls:
                tool = tools_by_name.get(tool_call["name"])
                if tool is None:  # the model asked for a tool that does not exist
                    observations.append(ToolMessage(
                        f"Error: there is no tool named {tool_call['name']!r}.",
                        tool_call_id=tool_call["id"], status="error",
                    ))
                    continue
                observations.append(await tool.ainvoke(tool_call))
            return {"messages": observations}

        def give_up(state: State) -> dict[str, Any]:
            """The step budget ran out: say so instead of looping forever."""
            request = state["messages"][-1]
            assert isinstance(request, AIMessage)
            # Every tool call needs an answer, or the next turn's prompt is invalid.
            skipped = [
                ToolMessage("Skipped: the step budget ran out.", tool_call_id=call["id"],
                            status="error")
                for call in request.tool_calls
            ]
            apology = AIMessage(
                f"I could not finish this within {MAX_STEPS} steps. "
                "Could you ask about fewer orders at a time?"
            )
            return {"messages": [*skipped, apology]}

        def after_reasoning(state: State) -> Literal["act", "give_up", "__end__"]:
            last = state["messages"][-1]
            if not isinstance(last, AIMessage) or not last.tool_calls:
                return END  # no tool calls: the model has answered
            if steps_this_turn(state["messages"]) >= MAX_STEPS:
                return "give_up"
            return "act"

        graph = StateGraph(State)
        graph.add_node("reason", reason)
        graph.add_node("act", act)
        graph.add_node("give_up", give_up)
        graph.add_edge(START, "reason")
        graph.add_conditional_edges("reason", after_reasoning, ["act", "give_up", END])
        graph.add_edge("act", "reason")  # observe, then reason again
        graph.add_edge("give_up", END)
        return graph


agent = ReactLoopAgent()
