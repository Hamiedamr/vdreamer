"""Lesson 23: specialists that hand the conversation to each other, through shared state.

No supervisor: whichever specialist is active talks to the customer directly.
When a request is outside its area, it calls ``transfer_to_<other>(note)``; the
note goes into shared state for the next specialist, and the graph routes there.
The active specialist is remembered, so the next turn continues with it::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t23-support-desk \
        --models config/models.ollama.yaml
    uv run vdreamer run t23-support-desk     # "I ordered A1001 but need a bigger tent"
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode, VDreamerToolMiddleware

from ..t21_specialists.specialists import SPECIALISTS, Specialist

FIRST_CONTACT = "orders"


class DeskState(AgentState[Any]):
    """Shared by every specialist: the conversation plus how it is being handled."""

    active_agent: NotRequired[str]  # who answers the next message
    handoff_to: NotRequired[str | None]  # set by a transfer during this step
    handoff_notes: NotRequired[list[str]]  # what each specialist told the next


def make_handoff(target: Specialist) -> BaseTool:
    """``transfer_to_<target>(note)``: ends the current specialist's turn (return_direct)."""

    @vdreamer.tool(
        f"transfer_to_{target.name}",
        description=f"Hand the customer over to the {target.name} specialist: "
                    f"{target.responsibility} Write a note with what they need to know.",
        return_direct=True,
    )
    def handoff(note: str, runtime: ToolRuntime) -> Command[Any]:
        done = ToolMessage(f"Transferred to {target.name}.", tool_call_id=runtime.tool_call_id)
        notes = [*runtime.state.get("handoff_notes", []), note]
        return Command(update={"messages": [done], "handoff_to": target.name,
                               "handoff_notes": notes})

    return handoff


@dynamic_prompt
def prompt_with_notes(request: ModelRequest) -> str:
    """Each specialist's own prompt, plus the notes colleagues left for it."""
    notes = request.state.get("handoff_notes", [])
    shared = "\nNotes from colleagues: " + " | ".join(notes) if notes else ""
    return str(request.system_prompt) + shared


def build_desk_agent(spec: Specialist, model: BaseChatModel) -> Any:
    others = [make_handoff(other) for other in SPECIALISTS if other is not spec]
    prompt = spec.prompt.replace(
        "If asked about anything else, say it is outside your area.",
        "If asked about anything else, transfer the customer to the right colleague.",
    )
    return create_agent(model, tools=[*spec.tools, *others], system_prompt=prompt,
                        state_schema=DeskState, name=spec.name,
                        middleware=[VDreamerToolMiddleware(), prompt_with_notes])


def as_node(name: str, agent: Any) -> Callable[[DeskState], Coroutine[Any, Any, dict[str, Any]]]:
    """Run one specialist on the shared state; return only what changed.

    Returning only *new* messages matters: vdreamer streams every AI message in a
    node's update, so returning the whole conversation would re-send old ones.
    """

    async def node(state: DeskState) -> dict[str, Any]:
        out = await agent.ainvoke({**state, "handoff_to": None})
        known = {message.id for message in state["messages"]}
        target = out.get("handoff_to")
        return {
            "messages": [m for m in out["messages"] if m.id not in known],
            "handoff_to": target,
            "active_agent": target or name,
            "handoff_notes": out.get("handoff_notes", []),
        }

    return node


def route_after(state: DeskState) -> str:
    return state.get("handoff_to") or END


def route_first(state: DeskState) -> str:
    return state.get("active_agent", FIRST_CONTACT)


class SupportDeskAgent(BaseAgent):
    name = "t23-support-desk"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> StateGraph:
        names = [spec.name for spec in SPECIALISTS]
        graph = StateGraph(DeskState)
        for spec in SPECIALISTS:
            specialist = build_desk_agent(spec, ctx.models["chat"])
            graph.add_node(spec.name, as_node(spec.name, specialist))
            graph.add_conditional_edges(spec.name, route_after, [*names, END])
        graph.add_conditional_edges(START, route_first, names)
        return graph


agent = SupportDeskAgent()
