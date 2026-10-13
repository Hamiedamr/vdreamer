"""The smoke agent (Milestone 0).

A trivial reference agent. It exists so the CI matrix (A41) has an agent to
run against the floor and latest 1.x pins, and so the plain-pip install job
can run ``vdreamer validate`` on a real agent. The real reference agents
UC1–UC3 (PRD §7.3) land with Milestone 12; each example becomes a uv
workspace member then.
"""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from vdreamer import BaseAgent, BuildContext, Mode


class PingerInput(BaseModel):
    """Trigger/input body; declared because mode is BOTH (D16, A16)."""

    message: str = ""


class PingerState(TypedDict):
    """Graph state: one node."""

    message: str
    reply: str


def _reply(state: PingerState) -> dict[str, Any]:
    return {"reply": f"pong: {state['message']}"}


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
