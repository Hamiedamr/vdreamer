"""Lesson 15: long-term memory — facts about a customer that outlive the conversation.

vdreamer gives every agent a LangGraph *store* (``ctx.store``): SQLite in
``vdreamer dev`` (the same ``.db`` file), Postgres in production. Unlike thread
state it is not tied to a conversation. This agent keeps notes per customer,
under a namespace built from the authenticated user id, so one customer's notes
are never visible in another customer's conversation::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t15-customer-memory \
        --models config/models.ollama.yaml
    uv run vdreamer run t15-customer-memory --user ada --message "I always camp in winter"
    uv run vdreamer run t15-customer-memory --user ada --message "Which sleeping bag?"
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.tools import ToolRuntime
from langchain_core.messages import SystemMessage
from langchain_core.tools import ToolException
from langgraph.store.base import BaseStore

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode, VDreamerContext

BASE_PROMPT = (
    "You are the assistant of an outdoor-gear shop. When the customer tells you something "
    "durable about themselves (how they camp, sizes, dislikes), save it with "
    "save_customer_note. Use what you know about them; never mention notes they did not give."
)
MAX_NOTES_IN_PROMPT = 10


def notes_namespace(user_id: str | None) -> tuple[str, ...]:
    """Where one customer's notes live. No user, no memory: never a shared namespace."""
    if not user_id:
        raise ToolException("Long-term memory needs a signed-in customer.")
    return ("customers", user_id, "notes")


def _store(runtime: ToolRuntime) -> BaseStore:
    if runtime.store is None:
        raise ToolException("No long-term store is configured.")
    return runtime.store


@vdreamer.tool(parse_docstring=True)
async def save_customer_note(note: str, runtime: ToolRuntime) -> str:
    """Save a durable fact about the customer for future conversations.

    Args:
        note: One short fact, e.g. "camps in winter, below -5 °C".
    """
    namespace = notes_namespace(vdreamer.current_context().user_id)
    await _store(runtime).aput(namespace, uuid.uuid4().hex, {"note": note})
    return f"Saved: {note}"


@vdreamer.tool
async def forget_customer_notes(runtime: ToolRuntime) -> str:
    """Delete everything remembered about the customer. Use only when they ask."""
    store = _store(runtime)
    namespace = notes_namespace(vdreamer.current_context().user_id)
    items = await store.asearch(namespace, limit=1000)
    for item in items:
        await store.adelete(namespace, item.key)
    return f"Deleted {len(items)} notes."


async def recall_notes(store: BaseStore | None, user_id: str | None) -> list[str]:
    if store is None or not user_id:
        return []
    items = await store.asearch(notes_namespace(user_id), limit=MAX_NOTES_IN_PROMPT)
    return [str(item.value["note"]) for item in sorted(items, key=lambda i: i.created_at)]


@wrap_model_call
async def prompt_with_customer_notes(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse:
    """Load the caller's notes from the store into the system prompt of every model call."""
    context = request.runtime.context
    user_id = context.user_id if isinstance(context, VDreamerContext) else None
    notes = await recall_notes(request.runtime.store, user_id)
    known = ("\nWhat you know about this customer:\n- " + "\n- ".join(notes)) if notes else ""
    return await handler(request.override(system_message=SystemMessage(BASE_PROMPT + known)))


class CustomerMemoryAgent(BaseAgent):
    name = "t15-customer-memory"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> Any:
        kwargs = ctx.agent_kwargs()  # includes store=ctx.store
        return create_agent(
            ctx.models["chat"],
            tools=[save_customer_note, forget_customer_notes],
            middleware=[*kwargs.pop("middleware"), prompt_with_customer_notes],
            **kwargs,
        )


agent = CustomerMemoryAgent()
