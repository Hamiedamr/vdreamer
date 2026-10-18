"""Lesson 8: an inventory agent whose tools are built from an injected dependency.

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t08-inventory \
        --models config/models.ollama.yaml
    uv run vdreamer run t08-inventory --message "Please reserve 2 units of TENT-4P"
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from vdreamer import BaseAgent, BuildContext, Mode, Policy

from .inventory import InMemoryInventory, Inventory
from .tools import make_inventory_tools

SYSTEM_PROMPT = (
    "You manage stock reservations for an outdoor-gear shop. Use check_stock to answer "
    "availability questions. Use reserve_stock only when the customer explicitly asks to "
    "reserve, and confirm the reservation id. If a tool reports an error, explain it."
)


def demo_inventory() -> InMemoryInventory:
    return InMemoryInventory({"TENT-2P": 0, "TENT-4P": 3, "BAG-0C": 12})


class InventoryAgent(BaseAgent):
    name = "t08-inventory"
    mode = Mode.CHAT
    models = ("chat",)
    # Deployment-specific limits belong to the agent, not to the reusable tool:
    # they override the decorators' policies for every matching tool name.
    tool_policies = {"*_stock": Policy(timeout_s=5)}

    def __init__(self, inventory: Inventory | None = None) -> None:
        self.inventory = inventory if inventory is not None else demo_inventory()

    def build(self, ctx: BuildContext) -> Any:
        return create_agent(
            ctx.models["chat"],
            tools=make_inventory_tools(self.inventory),
            system_prompt=SYSTEM_PROMPT,
            **ctx.agent_kwargs(),
        )


agent = InventoryAgent()
