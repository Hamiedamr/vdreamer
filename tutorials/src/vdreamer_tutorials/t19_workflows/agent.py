"""Lesson 19: a ticket workflow — fixed steps, a model where judgement is needed, branches.

A headless agent: other systems start it (``POST /agents/t19-ticket-workflow/trigger``
with a service token), never a chat. The path through the graph is decided by
code, from a structured classification::

    classify ──┬─ billing ──► check_payment ──┐
               ├─ returns ──► check_return ───┼─► draft_reply ─► finish
               ├─ other ──────────────────────┘
               └─ complaint ─► escalate ─────────────────────────► finish

::

    cd tutorials
    echo "VDREAMER_SERVICE_TOKENS=helpdesk:helpdesk-token-0123456789" >> .env
    uv run vdreamer dev --registry vdreamer.yaml --agent t19-ticket-workflow \
        --models config/models.ollama.yaml
    HELPDESK_TOKEN=helpdesk-token-0123456789 uv run vdreamer run t19-ticket-workflow \
        --trigger --service --token-env HELPDESK_TOKEN --idempotency-key ticket-123 \
        --input '{"ticket_id": "T-123", "body": "Was order A1002 charged twice?"}'
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ValidationError

import vdreamer
from vdreamer import ArtifactKind, BaseAgent, BuildContext, Mode

Category = Literal["billing", "returns", "complaint", "other"]
PAYMENTS = {"A1001": ["EUR 149.00 on 2026-09-28"],
            "A1002": ["EUR 89.00 on 2026-10-02", "EUR 89.00 on 2026-10-02"]}
DELIVERED_DAYS_AGO = {"A1001": 12, "A1003": 47}


class TicketIn(BaseModel):
    ticket_id: str = Field(pattern=r"^T-\d+$")
    body: str = Field(min_length=1, max_length=5000)


class Classification(BaseModel):
    """How to handle a support ticket."""

    category: Category
    order_id: str | None = Field(default=None, description='The order id, like "A1001".')


class WorkflowState(TypedDict):
    ticket_id: str
    body: str
    messages: NotRequired[Annotated[list[AnyMessage], add_messages]]
    category: NotRequired[Category]
    order_id: NotRequired[str | None]
    findings: NotRequired[str]
    reply: NotRequired[str]
    escalated: NotRequired[bool]


def find_order_id(text: str) -> str | None:
    match = re.search(r"\bA\d{4}\b", text)
    return match.group(0) if match else None


def check_payment(state: WorkflowState) -> dict[str, Any]:
    """Deterministic step: what the payment system says about the order."""
    vdreamer.emit_progress("checking payments", 40)
    charges = PAYMENTS.get(state.get("order_id") or "", [])
    duplicate = len(charges) != len(set(charges))
    note = f"charges: {charges or 'none found'}" + ("; DUPLICATE CHARGE" if duplicate else "")
    return {"findings": note}


def check_return(state: WorkflowState) -> dict[str, Any]:
    """Deterministic step: is the order inside the 30-day return window?"""
    vdreamer.emit_progress("checking the return window", 40)
    days = DELIVERED_DAYS_AGO.get(state.get("order_id") or "")
    if days is None:
        return {"findings": "order not delivered or unknown: no return possible yet"}
    verdict = "within" if days <= 30 else "outside"
    return {"findings": f"delivered {days} days ago, {verdict} the 30-day window"}


def escalate(state: WorkflowState) -> dict[str, Any]:
    """No model writes to an unhappy customer: a person takes over."""
    vdreamer.emit_progress("escalated to the support team", 80)
    reply = "A member of our team will contact you within a day."
    return {"escalated": True, "reply": reply, "messages": [AIMessage(reply)]}


def route(state: WorkflowState) -> str:
    return {"billing": "check_payment", "returns": "check_return",
            "complaint": "escalate"}.get(state["category"], "draft_reply")


async def finish(state: WorkflowState) -> dict[str, Any]:
    resolution = {
        "ticket_id": state["ticket_id"], "category": state["category"],
        "order_id": state.get("order_id"), "findings": state.get("findings"),
        "escalated": state.get("escalated", False), "reply": state["reply"],
    }
    await vdreamer.aemit_artifact("resolution", ArtifactKind.JSON, resolution)
    vdreamer.emit_progress("done", 100)
    return {}


class TicketWorkflowAgent(BaseAgent):
    name = "t19-ticket-workflow"
    mode = Mode.HEADLESS
    models = ("chat",)
    input_schema = TicketIn

    def build(self, ctx: BuildContext) -> StateGraph:
        model = ctx.models["chat"]
        classifier = model.with_structured_output(Classification, method="function_calling")

        async def classify(state: WorkflowState) -> dict[str, Any]:
            vdreamer.emit_progress("classifying", 10)
            try:
                result = await classifier.ainvoke([
                    SystemMessage("Classify this support ticket with the Classification tool."),
                    HumanMessage(state["body"]),
                ])
            except (ValidationError, OutputParserException):
                result = None
            if not isinstance(result, Classification):
                result = Classification(category="other")
            # Code, not the model, has the last word on facts it can check itself.
            order_id = find_order_id(state["body"]) or result.order_id
            return {"category": result.category, "order_id": order_id}

        async def draft_reply(state: WorkflowState) -> dict[str, Any]:
            vdreamer.emit_progress("drafting a reply", 70)
            facts = state.get("findings", "no system checks were needed")
            answer = await model.ainvoke([
                SystemMessage("Write a short, friendly reply to the customer. Use only these "
                              f"facts from our systems: {facts}."),
                HumanMessage(state["body"]),
            ])
            # The model's own message goes into `messages`: clients see it stream once.
            return {"reply": answer.text, "messages": [answer]}

        graph = StateGraph(WorkflowState)
        for node in (classify, check_payment, check_return, escalate, draft_reply, finish):
            graph.add_node(node.__name__, node)
        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify", route, ["check_payment", "check_return", "escalate", "draft_reply"]
        )
        graph.add_edge("check_payment", "draft_reply")
        graph.add_edge("check_return", "draft_reply")
        graph.add_edge("draft_reply", "finish")
        graph.add_edge("escalate", "finish")
        graph.add_edge("finish", END)
        return graph


agent = TicketWorkflowAgent()
