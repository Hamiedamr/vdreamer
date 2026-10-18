"""The vdreamer agent: wiring only — no business rules, no prompt text, no data."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from vdreamer import BaseAgent, BuildContext, Mode

from .domain import OrderRepository, demo_orders
from .prompts import system_prompt
from .settings import HelpdeskSettings
from .tools import make_tools


class HelpdeskAgent(BaseAgent):
    name = "t11-helpdesk"
    mode = Mode.CHAT
    models = ("chat",)

    def __init__(self, orders: OrderRepository | None = None,
                 settings: HelpdeskSettings | None = None) -> None:
        self.orders = orders if orders is not None else demo_orders()
        self.settings = settings if settings is not None else HelpdeskSettings()

    def build(self, ctx: BuildContext) -> Any:
        return create_agent(
            ctx.models["chat"],
            tools=make_tools(self.orders, self.settings),
            system_prompt=system_prompt(self.settings),
            **ctx.agent_kwargs(),
        )


agent = HelpdeskAgent()
