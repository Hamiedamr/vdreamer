"""Lesson 14: short-term memory — what an agent remembers within one conversation.

Two kinds, both scoped to the thread:

- the **transcript**: every message, restored by the checkpointer and sent to the
  model on each turn (``create_agent`` does this for you);
- **working memory**: structured facts the agent decided to keep — here the
  customer's preferences — written by a tool into thread state and put into the
  system prompt of every later model call.

::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t14-preferences \
        --models config/models.ollama.yaml
    uv run vdreamer run t14-preferences     # then: "Please keep answers short, in German"
"""

from __future__ import annotations

from typing import Any, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode

BASE_PROMPT = "You are the assistant of an outdoor-gear shop."


class PreferenceState(AgentState[Any]):
    """``create_agent``'s state (messages, …) plus this thread's preferences."""

    preferences: NotRequired[dict[str, str]]


@vdreamer.tool(parse_docstring=True)
def remember_preference(topic: str, value: str, runtime: ToolRuntime) -> Command[Any]:
    """Remember how the customer wants to be served, for the rest of this conversation.

    Args:
        topic: What the preference is about, e.g. "language", "length", "units".
        value: The preference, e.g. "German", "short", "imperial".
    """
    preferences = {**runtime.state.get("preferences", {}), topic.lower(): value}
    note = ToolMessage(f"Remembered: {topic} = {value}", tool_call_id=runtime.tool_call_id)
    return Command(update={"preferences": preferences, "messages": [note]})


@dynamic_prompt
def prompt_with_preferences(request: ModelRequest) -> str:
    """Rebuilt for every model call, from the thread's current state."""
    preferences = request.state.get("preferences", {})
    if not preferences:
        return BASE_PROMPT + " When the customer states a preference, remember it."
    listed = "; ".join(f"{topic}: {value}" for topic, value in sorted(preferences.items()))
    return f"{BASE_PROMPT} Always respect the customer's preferences — {listed}."


class PreferencesAgent(BaseAgent):
    name = "t14-preferences"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> Any:
        kwargs = ctx.agent_kwargs()
        return create_agent(
            ctx.models["chat"],
            tools=[remember_preference],
            state_schema=PreferenceState,
            middleware=[*kwargs.pop("middleware"), prompt_with_preferences],
            **kwargs,
        )


agent = PreferencesAgent()
