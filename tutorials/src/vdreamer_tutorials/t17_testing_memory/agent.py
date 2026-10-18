"""Lesson 17: a deterministic memory desk — the subject of a memory test suite.

No model: commands are parsed with plain string matching, so every test (and a
real `vdreamer dev` restart) is exact. It keeps two memories:

- ``remember for now: <text>`` → this thread only (thread state, checkpointed);
- ``remember forever: <text>`` → this user, every thread (``ctx.store``);
- ``recall`` → both lists; ``forget forever`` → deletes the user's long-term notes.

::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t17-memory-desk
    uv run vdreamer run t17-memory-desk --user ada --message "remember forever: size 44"
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode

HELP = "Say 'remember for now: …', 'remember forever: …', 'recall' or 'forget forever'."


def append(existing: list[str], new: list[str]) -> list[str]:
    return existing + new


class MemoryState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    session_notes: NotRequired[Annotated[list[str], append]]


def user_namespace(user_id: str) -> tuple[str, ...]:
    return ("memory-desk", user_id)


async def long_term_notes(store: BaseStore, user_id: str) -> list[str]:
    items = await store.asearch(user_namespace(user_id), limit=100)
    return [str(i.value["text"]) for i in sorted(items, key=lambda i: i.created_at)]


async def handle(state: MemoryState, runtime: Runtime[Any]) -> dict[str, Any]:
    text = state["messages"][-1].text.strip()
    command = text.lower()
    user_id = vdreamer.current_context().user_id
    store = runtime.store

    def say(reply: str, **update: Any) -> dict[str, Any]:
        return {"messages": [AIMessage(reply)], **update}

    if command.startswith("remember for now:"):
        note = text.split(":", 1)[1].strip()
        return say(f"Noted for this conversation: {note}", session_notes=[note])
    needs_user = command.startswith(("remember forever:", "recall", "forget forever"))
    if needs_user and (store is None or not user_id):
        return say("Long-term memory needs a signed-in user.")
    assert store is not None and user_id is not None
    if command.startswith("remember forever:"):
        note = text.split(":", 1)[1].strip()
        await store.aput(user_namespace(user_id), uuid.uuid4().hex, {"text": note})
        return say(f"Noted for good: {note}")
    if command.startswith("recall"):
        session = state.get("session_notes", [])
        forever = await long_term_notes(store, user_id)
        return say(f"This conversation: {session or 'nothing'}. Always: {forever or 'nothing'}.")
    if command.startswith("forget forever"):
        items = await store.asearch(user_namespace(user_id), limit=100)
        for item in items:
            await store.adelete(user_namespace(user_id), item.key)
        return say(f"Forgot {len(items)} notes.")
    return say(HELP)


class MemoryDeskAgent(BaseAgent):
    name = "t17-memory-desk"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> StateGraph:
        graph = StateGraph(MemoryState)
        graph.add_node("handle", handle)
        graph.add_edge(START, "handle")
        graph.add_edge("handle", END)
        return graph  # vdreamer compiles it with ctx.checkpointer and ctx.store


agent = MemoryDeskAgent()
