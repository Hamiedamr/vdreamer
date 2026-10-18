"""Lesson 18: turning free text into a validated, typed result.

A ticket comes in as typed input (``input_schema``); the model must return a
``TicketTriage``; pydantic validates it; one retry shows the model its mistake;
after that, a safe fallback. Clients receive the result as a JSON artifact plus
a readable message::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t18-ticket-triage \
        --models config/models.ollama.yaml
    uv run vdreamer run t18-ticket-triage --once --input \
        '{"subject": "Charged twice", "body": "Order A1002 was charged twice!"}'
"""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

import vdreamer
from vdreamer import ArtifactKind, BaseAgent, BuildContext, Mode

from .models import UNCLASSIFIED, TicketInput, TicketTriage

TRIAGE_PROMPT = (
    "You triage customer-support tickets for an outdoor-gear shop. Classify the ticket by "
    "calling the TicketTriage tool exactly once. Do not answer the customer."
)


def describe(exc: Exception) -> str:
    """What was wrong, in words the model can act on: one entry per invalid field."""
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
            for e in exc.errors(include_url=False)
        )
    return str(exc).splitlines()[0]


class TriageState(TypedDict):
    messages: NotRequired[Annotated[list[AnyMessage], add_messages]]  # the readable result
    subject: str
    body: str
    customer_email: NotRequired[str | None]
    triage: NotRequired[dict[str, Any]]
    validation_errors: NotRequired[list[str]]


class TicketTriageAgent(BaseAgent):
    name = "t18-ticket-triage"
    mode = Mode.BOTH  # chat clients and other systems (lesson 19) alike
    models = ("chat",)
    input_schema = TicketInput

    def build(self, ctx: BuildContext) -> StateGraph:
        # function_calling: the schema is sent as a tool; works on vLLM and Ollama alike.
        classifier = ctx.models["chat"].with_structured_output(
            TicketTriage, method="function_calling"
        )

        async def triage(state: TriageState) -> dict[str, Any]:
            prompt: list[Any] = [
                SystemMessage(TRIAGE_PROMPT),
                HumanMessage(f"Subject: {state['subject']}\n\n{state['body']}"),
            ]
            errors: list[str] = []
            for _attempt in range(2):  # one retry, with the error shown to the model
                try:
                    result = await classifier.ainvoke(prompt)
                except (ValidationError, OutputParserException) as exc:
                    result, problem = None, describe(exc)
                else:
                    problem = "no TicketTriage call"
                if isinstance(result, TicketTriage):
                    return {"triage": result.model_dump(), "validation_errors": errors}
                errors.append(problem)
                prompt.append(HumanMessage(
                    f"Your previous answer was invalid ({problem}). Call TicketTriage again "
                    "with valid values."
                ))
            vdreamer.emit_log(f"triage fell back after {len(errors)} invalid answers", "warning")
            return {"triage": UNCLASSIFIED.model_dump(), "validation_errors": errors}

        async def report(state: TriageState) -> dict[str, Any]:
            result = TicketTriage.model_validate(state["triage"])
            await vdreamer.aemit_artifact("triage", ArtifactKind.JSON, result.model_dump())
            flag = " — needs a human" if result.needs_human else ""
            line = f"{result.category} / {result.priority}{flag}: {result.summary}"
            return {"messages": [AIMessage(line)]}

        graph = StateGraph(TriageState)
        graph.add_node("triage", triage)
        graph.add_node("report", report)
        graph.add_edge(START, "triage")
        graph.add_edge("triage", "report")
        graph.add_edge("report", END)
        return graph


agent = TicketTriageAgent()
