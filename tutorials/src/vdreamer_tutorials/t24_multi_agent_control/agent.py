"""Lesson 24: the lesson-22 supervisor with limits on every level.

- each specialist: at most ``max_specialist_model_calls`` model calls per question
  (``ModelCallLimitMiddleware``), so a confused specialist stops instead of looping;
- each delegation: a ``timeout_s`` for the whole sub-conversation (vdreamer
  governance), and failures contained as tool errors the supervisor can explain;
- the supervisor: at most ``max_asks_per_specialist`` questions to any one
  specialist and ``max_delegations`` in total per run (``ToolCallLimitMiddleware``);
- the run: LangGraph's ``recursion_limit`` as the last backstop (client-settable).

::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t24-guarded-supervisor \
        --models config/models.ollama.yaml
    uv run vdreamer run t24-guarded-supervisor --message "Where is A1001 and what's in stock?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode, VDreamerToolMiddleware

from ..t21_specialists.specialists import SPECIALISTS, Specialist, build_specialist
from ..t22_supervisor.agent import supervisor_prompt


@dataclass(frozen=True)
class Limits:
    delegation_timeout_s: float = 60
    max_specialist_model_calls: int = 4
    max_asks_per_specialist: int = 2
    max_delegations: int = 5


def make_guarded_delegate(spec: Specialist, model: BaseChatModel, limits: Limits) -> BaseTool:
    specialist = build_specialist(spec, model, middleware=[
        VDreamerToolMiddleware(),
        # "end": stop the specialist with a message instead of failing the run.
        ModelCallLimitMiddleware(run_limit=limits.max_specialist_model_calls,
                                 exit_behavior="end"),
    ])

    @vdreamer.tool(
        f"ask_{spec.name}",
        description=f"{spec.responsibility} Ask one self-contained question.",
        timeout_s=limits.delegation_timeout_s,
    )
    async def delegate(question: str) -> str:
        result = await specialist.ainvoke({"messages": [{"role": "user", "content": question}]})
        return str(result["messages"][-1].text)

    return delegate


GUARDRAILS = (
    "\nIf a specialist reports an error, times out, or you hit a limit, do not retry "
    "endlessly: tell the customer what you could and could not find out."
)


class GuardedSupervisorAgent(BaseAgent):
    name = "t24-guarded-supervisor"
    mode = Mode.CHAT
    models = ("chat",)

    def __init__(self, limits: Limits | None = None) -> None:
        self.limits = limits or Limits()

    def build(self, ctx: BuildContext) -> Any:
        model, limits = ctx.models["chat"], self.limits
        kwargs = ctx.agent_kwargs()
        per_specialist = [
            ToolCallLimitMiddleware(tool_name=f"ask_{spec.name}",
                                    run_limit=limits.max_asks_per_specialist)
            for spec in SPECIALISTS
        ]
        return create_agent(
            model,
            tools=[make_guarded_delegate(spec, model, limits) for spec in SPECIALISTS],
            system_prompt=supervisor_prompt(SPECIALISTS) + GUARDRAILS,
            middleware=[
                *kwargs.pop("middleware"),
                *per_specialist,
                ToolCallLimitMiddleware(run_limit=limits.max_delegations),
            ],
            **kwargs,
        )


agent = GuardedSupervisorAgent()
