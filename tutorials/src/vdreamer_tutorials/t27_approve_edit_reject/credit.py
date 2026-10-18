"""Lesson 27, part 1: a goodwill credit proposed, then approved, rejected, edited or redrafted.

``vdreamer.ApprovalGate`` builds the whole review loop::

    prepare ─► gate ─┬─ approve / edit ─► execute
                     ├─ reject ─────────► on_reject
                     └─ feedback ───────► replan ─► prepare ─► gate …

``edit`` payloads are validated against ``CreditEdit`` (by the gateway, and again
in the gate) and written into state before ``execute`` runs::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t27-credit-approval \
        --models config/models.ollama.yaml
    uv run vdreamer run t27-credit-approval --message "Order A1001 arrived a week late"
"""

from __future__ import annotations

import re
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from vdreamer import ApprovalGate, BaseAgent, BuildContext, Mode

ORDER_TOTALS = {"A1001": 149.0, "A1002": 89.0}
ISSUED: dict[str, float] = {}  # stands in for the payment system
CREDIT_RATE, CREDIT_CAP_EUR = 0.10, 50.0


class CreditEdit(BaseModel):
    """What an approver may change. Its fields are state keys."""

    amount_eur: float = Field(gt=0, le=100)
    message: str = Field(min_length=10, max_length=500)


class CreditState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    order_id: NotRequired[str]
    amount_eur: NotRequired[float]
    message: NotRequired[str]
    review: NotRequired[dict[str, Any]]  # the gate's decision_key


def default_credit(order_id: str) -> float:
    return round(min(ORDER_TOTALS[order_id] * CREDIT_RATE, CREDIT_CAP_EUR), 2)


class CreditApprovalAgent(BaseAgent):
    name = "t27-credit-approval"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> StateGraph:
        # Drafts go to the approver, not the customer: keep them out of the stream.
        drafter = ctx.models["chat"].model_copy(update={"tags": ["nostream"]})

        async def draft(state: CreditState, feedback: str | None = None) -> str:
            ask = (f"Write a short apology to a customer whose order {state['order_id']} had a "
                   f"problem, offering a goodwill credit of EUR {state['amount_eur']:.2f}.")
            if feedback:
                ask += f" Previous draft: {state.get('message')!r}. Reviewer feedback: {feedback}"
            reply = await drafter.ainvoke([SystemMessage(ask), state["messages"][0]])
            return reply.text

        async def prepare(state: CreditState) -> dict[str, Any]:
            """Fill in what is missing; a replanned or edited proposal is kept."""
            update: dict[str, Any] = {}
            if "order_id" not in state:
                found = re.search(r"\bA\d{4}\b", state["messages"][-1].text.upper())
                update["order_id"] = found.group(0) if found else "A1001"
            if "amount_eur" not in state:
                update["amount_eur"] = default_credit(update.get("order_id")
                                                      or state["order_id"])
            if "message" not in state:
                update["message"] = await draft({**state, **update})
            return update

        async def replan(state: CreditState) -> dict[str, Any]:
            return {"message": await draft(state, feedback=state["review"].get("text"))}

        def execute(state: CreditState) -> dict[str, Any]:
            ISSUED[state["order_id"]] = state["amount_eur"]
            who = state["review"].get("resolved_by")
            return {"messages": [AIMessage(state["message"]),
                                 AIMessage(f"(EUR {state['amount_eur']:.2f} credited; "
                                           f"approved by {who})")]}

        def on_reject(state: CreditState) -> dict[str, Any]:
            return {"messages": [AIMessage(
                "We're sorry for the trouble with your order. A colleague will follow up.")]}

        gate = ApprovalGate(
            CreditState,
            kind="credit_review",
            payload=lambda s: {"order_id": s["order_id"], "amount_eur": s["amount_eur"],
                               "message": s["message"]},
            prepare=prepare,
            execute=execute,
            on_reject=on_reject,
            replan=replan,
            edit_schema=CreditEdit,
            decision_key="review",
        )
        return gate.build()


agent = CreditApprovalAgent()


