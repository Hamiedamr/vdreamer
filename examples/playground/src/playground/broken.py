"""``broken`` — the failure paths a governed tool contains, then one it does not.

Three tools fail in the three contained ways (an exception → ``tool_error``, a
deadline → ``timeout``, an oversized result → truncated + stored as a ``file``
artifact); the run carries on. Then a node raises, which fails the run with
``node_error``::

    vdreamer dev --agent playground.broken:agent --input '{"message": "go"}'
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode


@vdreamer.tool
def flaky(x: int) -> str:
    """Always fails — and its message carries a secret, which must not leak."""
    raise RuntimeError("database exploded: password=hunter2")


@vdreamer.tool(timeout_s=0.5)
def slow(x: int) -> str:
    """Misses its deadline."""
    time.sleep(2)
    return "late"


@vdreamer.tool(max_result_chars=100)
def big(x: int) -> str:
    """Returns far more than the model should see."""
    return "x" * 5000


class BrokenInput(BaseModel):
    message: str = ""


class State(TypedDict, total=False):
    message: str
    out: list[str]


class BrokenAgent(BaseAgent):
    name = "broken"
    mode = Mode.HEADLESS
    input_schema = BrokenInput

    def build(self, ctx: BuildContext) -> StateGraph:
        tools = {tool.name: tool for tool in ctx.tools.govern([flaky, slow, big])}

        async def call_tools(state: State) -> dict[str, Any]:
            return {
                "out": [
                    str(await tools[n].ainvoke({"x": 1}))[:200] for n in ("flaky", "slow", "big")
                ]
            }

        def explode(state: State) -> dict[str, Any]:
            raise ValueError("node blew up")

        graph = StateGraph(State)
        graph.add_node("call_tools", call_tools)
        graph.add_node("explode", explode)
        graph.add_edge(START, "call_tools")
        graph.add_edge("call_tools", "explode")
        graph.add_edge("explode", END)
        return graph


agent = BrokenAgent()
