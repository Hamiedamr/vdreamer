"""Lesson 25: pause a run until a person approves — ``vdreamer.require_approval``.

``prepare`` works out the refund; ``approve`` pauses the run and asks; the
decision routes to ``execute`` or ``decline``. No model needed, so it runs as is::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t25-refund-approval
    uv run vdreamer run t25-refund-approval --message "Please refund order A1001"
    # vdreamer run shows the request and asks: approve / reject
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode

ORDERS = {"A1001": 149.0, "A1002": 89.0, "A1003": 24.5}
REFUNDED: dict[str, float] = {}  # stands in for the payment system


class RefundState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    refund: NotRequired[dict[str, Any] | None]
    decision: NotRequired[dict[str, Any]]


def prepare(state: RefundState) -> dict[str, Any]:
    """Work before the pause. A resumed run restarts ``approve``, never this node."""
    vdreamer.emit_progress("preparing the refund", 30)
    match = re.search(r"\bA\d{4}\b", state["messages"][-1].text.upper())
    if match is None or match.group(0) not in ORDERS:
        return {"refund": None, "messages": [AIMessage("Which order should be refunded?")]}
    order_id = match.group(0)
    return {"refund": {"order_id": order_id, "amount_eur": ORDERS[order_id]}}


def approve(state: RefundState) -> dict[str, Any]:
    """Pause here until someone decides. The decision is this function's return value."""
    decision = vdreamer.require_approval(
        "refund_review",
        state["refund"],  # what the approver sees
        actions=("approve", "reject"),
    )
    return {"decision": decision}


def execute(state: RefundState) -> dict[str, Any]:
    refund = state["refund"]
    assert refund is not None
    REFUNDED[refund["order_id"]] = refund["amount_eur"]
    who = state["decision"].get("resolved_by")
    return {"messages": [AIMessage(
        f"Refunded EUR {refund['amount_eur']:.2f} for {refund['order_id']} (approved by {who})."
    )]}


def decline(state: RefundState) -> dict[str, Any]:
    return {"messages": [AIMessage("The refund was not approved. A colleague will contact you.")]}


def after_prepare(state: RefundState) -> Literal["approve", "__end__"]:
    return "approve" if state.get("refund") else END


def after_decision(state: RefundState) -> Literal["execute", "decline"]:
    return "execute" if state["decision"].get("action") == "approve" else "decline"


class RefundApprovalAgent(BaseAgent):
    name = "t25-refund-approval"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> StateGraph:
        graph = StateGraph(RefundState)
        for node in (prepare, approve, execute, decline):
            graph.add_node(node.__name__, node)
        graph.add_edge(START, "prepare")
        graph.add_conditional_edges("prepare", after_prepare, ["approve", END])
        graph.add_conditional_edges("approve", after_decision, ["execute", "decline"])
        graph.add_edge("execute", END)
        graph.add_edge("decline", END)
        return graph


agent = RefundApprovalAgent()
