"""``hello`` — a first agent: streaming, a governed tool, a table artifact, an approval.

Needs no model server: the model is LangChain's ``GenericFakeChatModel``, which
streams its canned answer word by word, so the whole protocol is visible with
zero configuration::

    vdreamer dev --agent playground.hello:agent
    vdreamer run hello --message "hi"
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import vdreamer
from vdreamer import ArtifactKind, BaseAgent, BuildContext, Mode

GREETING = "Hello there! I am a fake model, streaming my answer token by token."


@vdreamer.tool(redact=["secret"], timeout_s=5)
def lookup_city(city: str, secret: str = "") -> str:
    """Return the population of a city."""
    return {"cairo": "10M", "paris": "2M"}.get(city.lower(), "unknown")


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    approved: bool


class HelloAgent(BaseAgent):
    name = "hello"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> StateGraph:
        model = GenericFakeChatModel(messages=iter([AIMessage(content=GREETING)] * 1000))
        (lookup,) = ctx.tools.govern([lookup_city])

        async def think(state: State) -> dict[str, Any]:
            vdreamer.emit_progress("thinking", 20)
            vdreamer.emit_log("calling the model")
            return {"messages": [await model.ainvoke(state["messages"])]}

        async def use_tool(state: State) -> dict[str, Any]:
            await lookup.ainvoke({"city": "Cairo", "secret": "hunter2"})  # secret shows as ***
            vdreamer.emit_progress("tool done", 50)
            return {}

        async def make_table(state: State) -> dict[str, Any]:
            await vdreamer.aemit_artifact(
                "cities",
                ArtifactKind.TABLE,
                [{"city": "Cairo", "population_m": 10}, {"city": "Paris", "population_m": 2}],
            )
            vdreamer.emit_progress("table ready", 80)
            return {}

        def approve(state: State) -> dict[str, Any]:
            decision = vdreamer.require_approval("demo_review", {"question": "Publish the table?"})
            text = f"Decision: {decision['action']} by {decision.get('resolved_by')}"
            return {"approved": decision["action"] == "approve", "messages": [AIMessage(text)]}

        graph = StateGraph(State)
        graph.add_node("think", think)
        graph.add_node("use_tool", use_tool)
        graph.add_node("make_table", make_table)
        graph.add_node("approve", approve)
        graph.add_edge(START, "think")
        graph.add_edge("think", "use_tool")
        graph.add_edge("use_tool", "make_table")
        graph.add_edge("make_table", "approve")
        graph.add_edge("approve", END)
        return graph


agent = HelloAgent()
