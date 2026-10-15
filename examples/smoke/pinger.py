"""The smoke agent (Milestone 0).

A trivial reference agent. It exists so the CI matrix (A41) has an agent to
run against the floor and latest 1.x pins, and so the plain-pip install job
can run ``vdreamer validate`` on a real agent. The real reference agents
UC1–UC3 (PRD §7.3) land with Milestone 12; each example becomes a uv
workspace member then.
"""

from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from vdreamer import BaseAgent, BuildContext, Mode


class PingerInput(BaseModel):
    """Trigger/input body; declared because mode is BOTH (D16, A16)."""

    message: str = ""


class PingerState(TypedDict):
    """Graph state: one node. ``messages`` carries the reply to clients."""

    message: str
    reply: str
    messages: NotRequired[Annotated[list[AnyMessage], add_messages]]


def _reply(state: PingerState) -> dict[str, Any]:
    # State keys are the agent's own; only messages (and artifacts, logs,
    # progress) reach a client, so the reply is also an assistant message.
    reply = f"pong: {state['message']}"
    return {"reply": reply, "messages": [AIMessage(reply)]}


class PingerAgent(BaseAgent):
    """Echo one node through the vdreamer contract."""

    name = "pinger"
    mode = Mode.BOTH
    models = ()
    resources = ()
    input_schema = PingerInput
    expose_thinking = True
    access_roles = ()

    def build(self, ctx: BuildContext) -> StateGraph:
        graph = StateGraph(PingerState)
        graph.add_node("reply", _reply)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph


agent = PingerAgent()
