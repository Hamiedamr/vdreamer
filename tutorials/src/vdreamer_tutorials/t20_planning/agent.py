"""Lesson 20: plan, fan out, combine — one message with several questions.

``plan``: a model splits the customer's message into typed tasks (structured
output). ``work``: LangGraph's ``Send`` runs one worker per task, in parallel;
each worker is deterministic code. ``answer``: a model writes one reply from all
results. vdreamer has no planner of its own: this is a LangGraph pattern::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t20-planner \
        --models config/models.ollama.yaml
    uv run vdreamer run t20-planner --message \
        "Where is A1001, can I still return A1003, and what is your warranty on boots?"
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from pydantic import BaseModel, Field, ValidationError

import vdreamer
from vdreamer import ArtifactKind, BaseAgent, BuildContext, Mode

MAX_TASKS = 4
ORDERS = {"A1001": "shipped, arriving 2026-10-04", "A1002": "processing"}
DELIVERED_DAYS_AGO = {"A1001": 12, "A1003": 47}
POLICIES = {
    "returns": "Unused items can be returned within 30 days of delivery.",
    "shipping": "Orders over EUR 100 ship free; standard delivery takes 2-4 working days.",
    "warranty": "Tents and boots carry a 2-year warranty against manufacturing defects.",
}


class Task(BaseModel):
    kind: Literal["order_status", "return_check", "policy"]
    order_id: str | None = Field(default=None, description='For order tasks, e.g. "A1001".')
    topic: Literal["returns", "shipping", "warranty"] | None = Field(
        default=None, description="For policy tasks."
    )


class Plan(BaseModel):
    """The independent tasks needed to answer the customer's message."""

    tasks: list[Task] = Field(max_length=MAX_TASKS)


def gather(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Parallel workers append their results; ``None`` resets the list for a new turn.

    (Returning ``[]`` would not clear it: an appending reducer appends nothing.)
    """
    return [] if new is None else [*(existing or []), *new]


class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tasks: NotRequired[list[dict[str, Any]]]
    results: NotRequired[Annotated[list[str], gather]]  # parallel workers append here


def do_task(task: Task) -> str:
    """One worker: deterministic, so the facts in the answer are exact."""
    if task.kind == "order_status":
        status = ORDERS.get(task.order_id or "")
        return f"Order {task.order_id}: {status or 'not found'}"
    if task.kind == "return_check":
        days = DELIVERED_DAYS_AGO.get(task.order_id or "")
        if days is None:
            return f"Order {task.order_id}: not delivered, so no return yet"
        ok = "can" if days <= 30 else "can no longer"
        return f"Order {task.order_id}: delivered {days} days ago, {ok} be returned"
    return f"Policy ({task.topic}): {POLICIES.get(task.topic or '', 'no such policy')}"


async def work(payload: dict[str, Any]) -> dict[str, Any]:
    """Runs once per planned task, in parallel (``Send``)."""
    task = Task.model_validate(payload["task"])
    vdreamer.emit_log(f"working on {task.kind} {task.order_id or task.topic}")
    return {"results": [do_task(task)]}


def fan_out(state: PlannerState) -> list[Send] | str:
    tasks = state.get("tasks", [])
    if not tasks:
        return "answer"  # nothing to look up: answer directly
    return [Send("work", {"task": task}) for task in tasks]


class PlannerAgent(BaseAgent):
    name = "t20-planner"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> StateGraph:
        model = ctx.models["chat"]
        planner = model.with_structured_output(Plan, method="function_calling")

        async def plan(state: PlannerState) -> dict[str, Any]:
            try:
                result = await planner.ainvoke([
                    SystemMessage(
                        "Split the customer's latest message into independent lookup tasks "
                        f"(at most {MAX_TASKS}) with the Plan tool. Use no task for small talk."
                    ),
                    state["messages"][-1],
                ])
            except (ValidationError, OutputParserException):
                result = None
            tasks = result.tasks if isinstance(result, Plan) else []
            await vdreamer.aemit_artifact(  # a JSON artifact holds an object, not a list
                "plan", ArtifactKind.JSON, {"tasks": [t.model_dump() for t in tasks]}
            )
            vdreamer.emit_progress(f"planned {len(tasks)} tasks", 20)
            # results=None: start this turn with no results (see gather).
            return {"tasks": [t.model_dump() for t in tasks], "results": None}

        async def answer(state: PlannerState) -> dict[str, Any]:
            facts = "\n".join(f"- {r}" for r in state.get("results", [])) or "- (none)"
            reply = await model.ainvoke([
                SystemMessage("You are a shop assistant. Answer every part of the customer's "
                              f"message using only these facts:\n{facts}"),
                HumanMessage(state["messages"][-1].text),
            ])
            return {"messages": [reply]}

        graph = StateGraph(PlannerState)
        graph.add_node("plan", plan)
        graph.add_node("work", work)
        graph.add_node("answer", answer)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", fan_out, ["work", "answer"])
        graph.add_edge("work", "answer")  # answer waits for every worker
        graph.add_edge("answer", END)
        return graph


agent = PlannerAgent()
