"""Four custom middlewares, each one concern, each testable on its own.

- ``input_guard`` (``before_agent``): refuse oversized or suspicious input before
  any model call;
- ``AuditTrail`` (``awrap_tool_call``): one audit entry per tool call, with who
  asked and what happened;
- ``MaintenanceSwitch`` (``awrap_tool_call``): switch individual tools off at
  run time without redeploying;
- ``mask_emails`` (``wrap_model_call``): rewrite the model's answer before it is
  saved or shown — together with a ``nostream`` model, so nothing unmasked streams.

Authorization itself is vdreamer's: tool ``scope``/``requires_roles``, checked by
the governance middleware that ``ctx.agent_kwargs()`` adds first.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    before_agent,
    wrap_model_call,
)
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

import vdreamer

MAX_INPUT_CHARS = 2000
SUSPICIOUS = ("ignore previous instructions", "ignore all previous", "system prompt")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


# -- validation: input ------------------------------------------------------------------------


@before_agent(can_jump_to=["end"])
def input_guard(state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
    """Refuse before spending a model call: too long, or an obvious injection attempt."""
    text = state["messages"][-1].text
    problem = None
    if len(text) > MAX_INPUT_CHARS:
        problem = f"Your message is too long ({len(text)} characters; the limit is " \
                  f"{MAX_INPUT_CHARS})."
    elif any(phrase in text.lower() for phrase in SUSPICIOUS):
        problem = "I can't help with that request."
    if problem is None:
        return None
    vdreamer.emit_log(f"input refused: {problem}", level="warning")
    return {"messages": [AIMessage(problem)], "jump_to": "end"}


# -- logging: an audit trail of tool calls -----------------------------------------------------


@dataclass
class AuditEntry:
    run_id: str
    user_id: str | None
    tool: str
    status: str


@dataclass
class AuditTrail(AgentMiddleware[Any, Any, Any]):
    """Records every tool call: the ``sink`` decides where entries go (a list, a log, a DB)."""

    sink: Callable[[AuditEntry], None]

    def __post_init__(self) -> None:
        super().__init__()

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        result = await handler(request)
        status = result.status if isinstance(result, ToolMessage) else "ok"
        context = vdreamer.current_context()
        self.sink(AuditEntry(run_id=context.run_id, user_id=context.user_id,
                             tool=request.tool_call["name"], status=status))
        return result


# -- execution control: switch tools off at run time -------------------------------------------


@dataclass
class MaintenanceSwitch(AgentMiddleware[Any, Any, Any]):
    """Tools named in ``disabled`` answer "unavailable" instead of running."""

    disabled: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        super().__init__()

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        name = request.tool_call["name"]
        if name in self.disabled:
            return ToolMessage(f"{name} is temporarily unavailable (maintenance).",
                               tool_call_id=request.tool_call["id"], status="error")
        return await handler(request)


# -- validation: output ------------------------------------------------------------------------


@wrap_model_call
async def mask_emails(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse:
    """Hide email addresses in the model's answer before anyone sees it."""
    response = await handler(request)
    masked = [
        m.model_copy(update={"content": EMAIL.sub("[email hidden]", m.text)})
        if isinstance(m, AIMessage) and m.text and EMAIL.search(m.text) else m
        for m in response.result
    ]
    return ModelResponse(result=masked, structured_response=response.structured_response)
