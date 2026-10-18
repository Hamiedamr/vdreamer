"""Lesson 26: two kinds of pause — asking the customer, and asking someone with authority.

A warranty claim needs the product's serial number; if the message lacks it, the
run asks the customer (``respond`` with text) and re-asks if the answer is not a
valid serial. A replacement then needs a warranty manager's approval
(``approver_role``): the customer cannot approve their own claim. Started by a
system (headless), there is nobody to ask for details, so the run stops politely::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t26-warranty-claims
    uv run vdreamer run t26-warranty-claims --user ada --message "Tent pole broke, order A1001"
    uv run vdreamer run t26-warranty-claims --user sam --roles warranty_manager --inbox
"""

from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode

ORDER_DATES = {"A1001": date(2026, 3, 2), "A1002": date(2023, 5, 20)}
TODAY = date(2026, 10, 18)
WARRANTY_YEARS = 2
SERIAL = re.compile(r"\bSN-\d{6}\b")
MAX_QUESTIONS = 2
MANAGER_ROLE = "warranty_manager"


class ClaimInput(BaseModel):
    message: str = Field(min_length=3, description="The customer's warranty claim.")


class ClaimState(TypedDict):
    message: str
    messages: NotRequired[Annotated[list[AnyMessage], add_messages]]
    order_id: NotRequired[str | None]
    serial: NotRequired[str | None]
    questions_asked: NotRequired[int]
    in_warranty: NotRequired[bool]
    decision: NotRequired[dict[str, Any]]


def intake(state: ClaimState) -> dict[str, Any]:
    order = re.search(r"\bA\d{4}\b", state["message"].upper())
    serial = SERIAL.search(state["message"].upper())
    return {"order_id": order.group(0) if order else None,
            "serial": serial.group(0) if serial else None, "questions_asked": 0}


def ask_serial(state: ClaimState) -> dict[str, Any]:
    """Pause for *information*: the initiator answers with text (``respond``)."""
    asked = state.get("questions_asked", 0)
    question = "What is the serial number on the product's label? It looks like SN-123456."
    if asked:
        question = "That did not look like a serial number. " + question
    answer = vdreamer.require_approval("claim_details", {"question": question},
                                       actions=("respond",))
    found = SERIAL.search(str(answer.get("text", "")).upper())
    return {"serial": found.group(0) if found else None, "questions_asked": asked + 1}


def assess(state: ClaimState) -> dict[str, Any]:
    bought = ORDER_DATES.get(state.get("order_id") or "")
    in_warranty = bought is not None and (TODAY - bought).days <= 365 * WARRANTY_YEARS
    return {"in_warranty": in_warranty}


def manager_review(state: ClaimState) -> dict[str, Any]:
    """Pause for *authority*: only a warranty manager may decide (the gateway enforces it)."""
    decision = vdreamer.require_approval(
        "warranty_replacement",
        {"order_id": state.get("order_id"), "serial": state.get("serial"),
         "claim": state["message"]},
        actions=("approve", "reject"),
        approver_role=MANAGER_ROLE,
        ttl_hours=72,
    )
    return {"decision": decision}


def reply(text: str) -> dict[str, Any]:
    return {"messages": [AIMessage(text)]}


def conclude(state: ClaimState) -> dict[str, Any]:
    decision = state.get("decision", {})
    if decision.get("action") == "approve":
        return reply(f"Your claim for {state['order_id']} was approved by "
                     f"{decision.get('resolved_by')}: a replacement is on its way.")
    return reply("Your claim was reviewed and not approved; we will email you the reasons.")


def after_intake(state: ClaimState) -> Literal["ask_serial", "assess", "need_details", "no_order"]:
    if not state.get("order_id"):
        return "no_order"
    if state.get("serial"):
        return "assess"
    # Headless runs have nobody to ask: stop instead of pausing.
    return "need_details" if vdreamer.current_context().headless else "ask_serial"


def after_question(state: ClaimState) -> Literal["ask_serial", "assess", "need_details"]:
    if state.get("serial"):
        return "assess"
    return "ask_serial" if state.get("questions_asked", 0) < MAX_QUESTIONS else "need_details"


def after_assess(state: ClaimState) -> Literal["manager_review", "out_of_warranty"]:
    return "manager_review" if state.get("in_warranty") else "out_of_warranty"


class WarrantyClaimsAgent(BaseAgent):
    name = "t26-warranty-claims"
    mode = Mode.BOTH
    input_schema = ClaimInput

    def build(self, ctx: BuildContext) -> StateGraph:
        graph = StateGraph(ClaimState)
        graph.add_node("intake", intake)
        graph.add_node("ask_serial", ask_serial)
        graph.add_node("assess", assess)
        graph.add_node("manager_review", manager_review)
        graph.add_node("conclude", conclude)
        graph.add_node("no_order", lambda s: reply("Which order is the claim about?"))
        graph.add_node("need_details", lambda s: reply(
            "We need the product's serial number (SN-…) to process the claim."))
        graph.add_node("out_of_warranty", lambda s: reply(
            f"This order is older than {WARRANTY_YEARS} years, so it is out of warranty."))
        graph.add_edge(START, "intake")
        graph.add_conditional_edges("intake", after_intake,
                                    ["ask_serial", "assess", "need_details", "no_order"])
        graph.add_conditional_edges("ask_serial", after_question,
                                    ["ask_serial", "assess", "need_details"])
        graph.add_conditional_edges("assess", after_assess, ["manager_review", "out_of_warranty"])
        graph.add_edge("manager_review", "conclude")
        for end in ("conclude", "no_order", "need_details", "out_of_warranty"):
            graph.add_edge(end, END)
        return graph


agent = WarrantyClaimsAgent()
