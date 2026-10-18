"""Lesson 12: an agent you can configure, follow in logs, and watch in metrics.

    cd tutorials
    echo T12_LOG_LEVEL=DEBUG >> .env
    uv run vdreamer dev --registry vdreamer.yaml --agent t12-observable \
        --models config/models.ollama.yaml
    uv run vdreamer run t12-observable --message "Where is order A1001?"
    curl -s http://127.0.0.1:8000/api/v1/metrics | grep vdreamer_tool
"""

from __future__ import annotations

import time
from typing import Any

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode, RunInfo

from .logs import get_logger
from .settings import ObservabilitySettings

ORDERS = {"A1001": "shipped", "A1002": "processing"}


def make_tools(settings: ObservabilitySettings) -> list[BaseTool]:
    log = get_logger(settings.log_level)

    @vdreamer.tool
    def get_order_status(order_id: str) -> str:
        """Look up the status of an order by its id, for example "A1001"."""
        started = time.monotonic()
        vdreamer.emit_progress("looking up the order", 50)  # for the user
        status = ORDERS.get(order_id.strip().upper(), "unknown")
        elapsed_ms = (time.monotonic() - started) * 1000
        log.info("order lookup order_id=%s status=%s", order_id, status)  # for operators
        if elapsed_ms > settings.slow_tool_ms:
            log.warning("slow order lookup elapsed_ms=%.0f", elapsed_ms)
        return f"Order {order_id}: {status}"

    return [get_order_status]


class ObservableAgent(BaseAgent):
    name = "t12-observable"
    mode = Mode.CHAT
    models = ("chat",)

    def __init__(self, settings: ObservabilitySettings | None = None) -> None:
        # Loaded when the instance is created, which vdreamer does at startup when it
        # imports the agent: an invalid environment stops `dev`/`worker`/`serve` before
        # any run. Tests pass settings explicitly.
        self.settings = settings if settings is not None else ObservabilitySettings.from_env()
        self._started: dict[str, float] = {}

    def build(self, ctx: BuildContext) -> Any:
        settings = self.settings
        log = get_logger(settings.log_level)

        @before_model
        def log_context_size(state: AgentState[Any], runtime: Runtime[Any]) -> None:
            log.debug("model call messages=%d", len(state["messages"]))

        kwargs = ctx.agent_kwargs()
        return create_agent(
            ctx.models["chat"],
            tools=make_tools(settings),
            system_prompt=f"You are the order assistant of {settings.shop_name}.",
            middleware=[*kwargs.pop("middleware"), log_context_size],
            **kwargs,
        )

    async def on_run_start(self, run: RunInfo) -> None:
        self._started[run.run_id] = time.monotonic()
        get_logger(self.settings.log_level).info(
            "run started agent=%s attempt=%d", run.agent_name, run.attempt
        )

    async def on_run_end(self, run: RunInfo, status: str) -> None:
        started = self._started.pop(run.run_id, time.monotonic())
        get_logger(self.settings.log_level).info(
            "run ended status=%s duration_ms=%.0f", status, (time.monotonic() - started) * 1000
        )


agent = ObservableAgent()
