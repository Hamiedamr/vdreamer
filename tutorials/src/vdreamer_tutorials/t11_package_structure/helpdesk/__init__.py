"""helpdesk — an order-support agent, organised as a small application.

Layers, each importing only the ones below it::

    agent.py     the vdreamer BaseAgent: wiring only
    tools.py     tool factory: the agent-facing contract
    prompts.py   prompt text, built from settings
    settings.py  plain settings object
    domain.py    business types and rules: no LangChain, no vdreamer
"""

from .agent import HelpdeskAgent, agent
from .domain import InMemoryOrders, Order, OrderRepository
from .settings import HelpdeskSettings

__all__ = ["HelpdeskAgent", "HelpdeskSettings", "InMemoryOrders", "Order", "OrderRepository",
           "agent"]
