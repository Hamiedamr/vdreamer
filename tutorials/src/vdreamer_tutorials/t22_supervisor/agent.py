"""Lesson 22: a supervisor agent that delegates to the lesson-21 specialists.

Each specialist becomes one tool of the supervisor, ``ask_<name>(question)``,
whose description is the specialist's responsibility. The supervisor decides
whom to ask (possibly several at once) and writes the final answer::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t22-support-supervisor \
        --models config/models.ollama.yaml
    uv run vdreamer run t22-support-supervisor \
        --message "Is the family tent in stock, and can I return it if it's too big?"
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode, VDreamerToolMiddleware

from ..t21_specialists.specialists import SPECIALISTS, Specialist, build_specialist


def make_delegate(spec: Specialist, model: BaseChatModel, timeout_s: float = 90) -> BaseTool:
    """``ask_<name>``: one specialist behind a tool. Each call is a fresh conversation."""
    specialist = build_specialist(spec, model, middleware=[VDreamerToolMiddleware()])

    @vdreamer.tool(
        f"ask_{spec.name}",
        description=f"{spec.responsibility} Ask one self-contained question.",
        timeout_s=timeout_s,
    )
    async def delegate(question: str) -> str:
        result = await specialist.ainvoke({"messages": [{"role": "user", "content": question}]})
        return str(result["messages"][-1].text)

    return delegate


def supervisor_prompt(specialists: tuple[Specialist, ...]) -> str:
    team = "\n".join(f"- ask_{s.name}: {s.responsibility}" for s in specialists)
    return (
        "You coordinate a support team for an outdoor-gear shop. You cannot look anything "
        f"up yourself; ask your specialists:\n{team}\n"
        "Ask independent questions in parallel. Each specialist sees only your question, "
        "so include the order ids and product names it needs. Then answer the customer in "
        "your own words, using only what the specialists said."
    )


class SupportSupervisorAgent(BaseAgent):
    name = "t22-support-supervisor"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> Any:
        model = ctx.models["chat"]
        return create_agent(
            model,
            tools=[make_delegate(spec, model) for spec in SPECIALISTS],
            system_prompt=supervisor_prompt(SPECIALISTS),
            **ctx.agent_kwargs(),
        )


agent = SupportSupervisorAgent()
