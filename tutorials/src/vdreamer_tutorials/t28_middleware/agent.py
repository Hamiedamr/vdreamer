"""Lesson 28: an order assistant assembled from middleware, in a deliberate order.

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t28-guarded-assistant \
        --models config/models.ollama.yaml
    uv run vdreamer run t28-guarded-assistant --roles "orders:*" \
        --message "Change the address of A1001 to 1 Main St"
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode

from .middleware import AuditEntry, AuditTrail, MaintenanceSwitch, input_guard, mask_emails

ORDERS = {"A1001": {"status": "processing", "contact": "ada@example.com",
                    "address": "5 Old Road"}}
#: What get_order may show: tool results reach clients as ``tool.result`` events too,
#: so a tool returns only fields that may be shown (the email stays in the system).
VISIBLE_FIELDS = ("status", "address")
log = logging.getLogger("vdreamer_tutorials.t28.audit")


@vdreamer.tool
def get_order(order_id: str) -> dict[str, Any]:
    """Look an order up: status and shipping address."""
    order = ORDERS.get(order_id.upper(), {"status": "unknown"})
    return {"order_id": order_id, **{k: v for k, v in order.items() if k in VISIBLE_FIELDS}}


@vdreamer.tool(scope="orders:write", side_effects=True)
def update_shipping_address(order_id: str, address: str) -> str:
    """Change where an order is shipped. Needs the orders:write scope."""
    ORDERS.setdefault(order_id.upper(), {})["address"] = address
    return f"Shipping address of {order_id} is now {address}."


def log_audit(entry: AuditEntry) -> None:
    log.info("tool=%s status=%s user=%s run=%s", entry.tool, entry.status, entry.user_id,
             entry.run_id)


class GuardedAssistantAgent(BaseAgent):
    name = "t28-guarded-assistant"
    mode = Mode.CHAT
    models = ("chat",)

    def __init__(self, audit: Callable[[AuditEntry], None] = log_audit,
                 disabled_tools: set[str] | None = None) -> None:
        self.audit = audit
        self.switch = MaintenanceSwitch(disabled_tools or set())

    def build(self, ctx: BuildContext) -> Any:
        kwargs = ctx.agent_kwargs()
        # The answer is checked (mask_emails) before it is shown: don't stream tokens.
        model = ctx.models["chat"].model_copy(update={"tags": ["nostream"]})
        return create_agent(
            model,
            tools=[get_order, update_shipping_address],
            system_prompt="You help customers with their orders.",
            middleware=[
                *kwargs.pop("middleware"),  # 1. vdreamer governance: authorize, event, contain
                input_guard,                # 2. refuse bad input before any model call
                AuditTrail(self.audit),     # 3. record what tools did
                self.switch,                # 4. tools switched off at run time
                mask_emails,                # 5. check the answer before it is saved or shown
            ],
            **kwargs,
        )


agent = GuardedAssistantAgent()
