"""Lesson 21: each specialist served on its own, with its own name and its own tests.

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t21-orders \
        --models config/models.ollama.yaml
    uv run vdreamer run t21-orders --message "What is in order A1001?"
"""

from __future__ import annotations

from typing import Any

from vdreamer import BaseAgent, BuildContext, Mode

from .specialists import (
    ORDERS_SPECIALIST,
    POLICIES_SPECIALIST,
    PRODUCTS_SPECIALIST,
    Specialist,
    build_specialist,
)


class SpecialistAgent(BaseAgent):
    """Serves one ``Specialist`` as a standalone vdreamer agent."""

    mode = Mode.CHAT
    models = ("chat",)

    def __init__(self, spec: Specialist) -> None:
        self.spec = spec
        self.name = f"t21-{spec.name}"

    def build(self, ctx: BuildContext) -> Any:
        return build_specialist(self.spec, ctx.models["chat"], **ctx.agent_kwargs())


orders_agent = SpecialistAgent(ORDERS_SPECIALIST)
products_agent = SpecialistAgent(PRODUCTS_SPECIALIST)
policies_agent = SpecialistAgent(POLICIES_SPECIALIST)
