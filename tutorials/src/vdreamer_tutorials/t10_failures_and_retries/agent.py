"""Lesson 10: an agent that survives a flaky API, a failing model, and an empty answer.

- ``ToolRetryMiddleware`` retries the parcel-tracking call on transient errors;
- ``ModelRetryMiddleware`` retries a failed model call, then
  ``ModelFallbackMiddleware`` switches to the ``backup`` model alias;
- ``replace_empty_answer`` (an ``after_model`` hook) never lets an empty reply
  reach the customer.

::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t10-resilient-tracker \
        --models config/models.ollama.yaml
    uv run vdreamer run t10-resilient-tracker --message "Where is parcel TRK-1001?"
"""

from __future__ import annotations

from typing import Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    after_model,
)
from langchain_core.messages import AIMessage, RemoveMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode

#: Errors worth retrying: the call may succeed if simply made again.
TRANSIENT = (ConnectionError, TimeoutError)

FALLBACK_ANSWER = (
    "Sorry, I could not put an answer together just now. Could you rephrase or try again?"
)


class FlakyCarrier:
    """A stand-in for a carrier API that fails a few times before answering."""

    def __init__(self, failures_before_success: int = 0) -> None:
        self.failures_left = failures_before_success
        self.calls = 0

    def locate(self, tracking_id: str) -> str:
        self.calls += 1
        if self.failures_left > 0:
            self.failures_left -= 1
            raise ConnectionError("carrier API: 503 Service Unavailable")
        return f"{tracking_id}: in transit, at the Frankfurt hub"


def make_track_parcel(carrier: FlakyCarrier) -> BaseTool:
    @vdreamer.tool(timeout_s=30)  # the timeout covers every retry
    def track_parcel(tracking_id: str) -> str:
        """Ask the carrier where a parcel is, by tracking id (for example "TRK-1001")."""
        return carrier.locate(tracking_id)

    return track_parcel


@after_model
def replace_empty_answer(state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
    """An empty final answer (no text, no tool calls) is removed and a fallback added.

    The fallback is a *new* message: clients already received the empty one as it
    streamed, and only a new message produces new events.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and not last.tool_calls and not last.text.strip():
        vdreamer.emit_log("the model returned an empty answer; replaced it", level="warning")
        assert last.id is not None
        return {"messages": [RemoveMessage(id=last.id), AIMessage(FALLBACK_ANSWER)]}
    return None


SYSTEM_PROMPT = (
    "You track parcels for an online shop with the track_parcel tool. If the tool keeps "
    "failing, apologise and suggest trying again in a few minutes."
)


class ResilientTrackerAgent(BaseAgent):
    name = "t10-resilient-tracker"
    mode = Mode.CHAT
    models = ("chat", "backup")

    def __init__(self, carrier: FlakyCarrier | None = None, retry_delay_s: float = 1.0) -> None:
        self.carrier = carrier if carrier is not None else FlakyCarrier(failures_before_success=1)
        self.retry_delay_s = retry_delay_s

    def build(self, ctx: BuildContext) -> Any:
        kwargs = ctx.agent_kwargs()
        middleware = [
            *kwargs.pop("middleware"),
            # Outer: if the primary model still fails after its retries, use the backup.
            ModelFallbackMiddleware(ctx.models["backup"]),
            # Inner: retry each model, re-raising when out of retries so the fallback runs.
            ModelRetryMiddleware(max_retries=2, initial_delay=self.retry_delay_s,
                                 on_failure="error"),
            ToolRetryMiddleware(tools=["track_parcel"], max_retries=2, retry_on=TRANSIENT,
                                initial_delay=self.retry_delay_s),
            replace_empty_answer,
        ]
        return create_agent(
            ctx.models["chat"],
            tools=[make_track_parcel(self.carrier)],
            system_prompt=SYSTEM_PROMPT,
            middleware=middleware,
            **kwargs,
        )


agent = ResilientTrackerAgent()
