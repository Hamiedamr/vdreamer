# Lesson 14 — Short-term conversational memory

> **Outcome:** an assistant that remembers within a conversation — everything said so far, plus preferences it chose to keep ("answer in German", "metric units") — by writing them to thread state with a tool and putting them into every later prompt.

## Learning objectives

- Explain what "short-term memory" is in a vdreamer agent: the thread, restored by the checkpointer.
- Rely on the automatic transcript, and know its limits.
- Add **working memory** to a `create_agent` agent: extend its state (`state_schema`), write to it from a tool (`Command(update=...)` with `ToolRuntime`), read it in a `@dynamic_prompt`.
- Know where short-term memory ends: a new thread starts empty.

## Prerequisites

- Lesson [13](../t13_state_and_history/README.md) (state, reducers, checkpoints).
- Lesson [9](../t09_tool_selection/README.md) (middleware with `create_agent`).

## Concepts

A model has no memory between calls: every call sends everything the model should know. "Memory" is therefore about **what you put in the prompt**, and where you keep it in between.

**Short-term memory** is everything scoped to one conversation (one vdreamer thread). It has two parts:

- **The transcript.** `create_agent` keeps the `messages` list in state; vdreamer's checkpointer restores it at the start of every run on the thread, so the model sees the whole conversation each turn. This is automatic — lessons 2–12 already relied on it.
- **Working memory.** Facts worth keeping in a structured form: preferences, the order being discussed, decisions taken. The transcript contains them too, but buried, and it will eventually be trimmed (lesson 16). Structured facts survive trimming and are cheap to restate in the system prompt.

In `create_agent`, a **tool can update state**: instead of a string it returns a `Command(update={...})`, and it gets the current state and its own call id from an injected `runtime: ToolRuntime` argument (injected parameters are hidden from the model's schema and from vdreamer's `tool.call` events). The update must include a `ToolMessage` answering the call. A `@dynamic_prompt` middleware then builds the system prompt from state on every model call.

**Long-term memory** — facts that outlive a conversation, like a customer's preferences next week — is a different store with different rules (lesson 15).

## What comes from where

| Piece | Provided by |
| --- | --- |
| Transcript persistence per thread (checkpointer), thread ids from the API | vdreamer |
| `state_schema=` (extending `AgentState`), `@dynamic_prompt`, `ToolRuntime` | LangChain (`langchain.agents`, `langchain.tools`) |
| `Command(update=...)` | LangGraph |
| `remember_preference`, the prompt builder | this lesson's code |

## How it works

```
turn 1: "From now on, please answer in German"
   model ─► remember_preference(topic="language", value="German")
   tool  ─► Command(update={preferences: {language: German}, messages: [ToolMessage("Remembered…")]})
   model ─► "Alles klar, …"                               ← checkpointed with the thread
turn 2: "How much is the 2-person tent?"
   restored: messages (4), preferences {language: German}
   dynamic_prompt ─► "… Always respect the customer's preferences — language: German."
   model ─► "Das Zelt kostet 149 Euro."
```

## Step by step

**1. Extend the agent's state.**

```python
# file: agent.py
class PreferenceState(AgentState[Any]):
    """``create_agent``'s state (messages, …) plus this thread's preferences."""

    preferences: NotRequired[dict[str, str]]
```

**2. A tool that writes to state.** `runtime` is injected — the model never sees or fills it:

```python
# file: agent.py
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
```

Copy, then update (`{**old, key: value}`): never mutate the dict you read from state.

**3. A prompt rebuilt from state on every call.**

```python
# file: agent.py
@dynamic_prompt
def prompt_with_preferences(request: ModelRequest) -> str:
    """Rebuilt for every model call, from the thread's current state."""
    preferences = request.state.get("preferences", {})
    if not preferences:
        return BASE_PROMPT + " When the customer states a preference, remember it."
    listed = "; ".join(f"{topic}: {value}" for topic, value in sorted(preferences.items()))
    return f"{BASE_PROMPT} Always respect the customer's preferences — {listed}."
```

**4. Wire both into `create_agent`.**

```python
# file: agent.py
        return create_agent(
            ctx.models["chat"],
            tools=[remember_preference],
            state_schema=PreferenceState,
            middleware=[*kwargs.pop("middleware"), prompt_with_preferences],
            **kwargs,
        )
```

## The complete example

[`agent.py`](agent.py) — `PreferenceState`, `remember_preference`, `prompt_with_preferences`, `PreferencesAgent` (`t14-preferences`).

## Run it

From `tutorials/`:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t14-preferences --models config/models.ollama.yaml
uv run vdreamer run t14-preferences
```

At `you>`: `From now on, please answer in German and keep it short`, then a few questions. Quit (Ctrl+C) and start `vdreamer run` again: a new thread — the preferences are gone. Reattach to the old thread with `vdreamer run t14-preferences --thread <thread_id>` and they are back.

## Expected behavior

```
-> remember_preference({"topic": "language", "value": "German"}) [custom]
<- ok  (2 ms)
assistant>
Alles klar, ich antworte ab jetzt auf Deutsch.
```

The result summary is empty: vdreamer summarises a tool's `ToolMessage` result, and this tool returns a `Command` (state update) instead. The `ToolMessage` inside it still reaches the model. Later answers follow the preference even after many turns, because it is restated in every system prompt.

## Test it

[`test_t14_short_term_memory.py`](../../../../tests/tutorials/test_t14_short_term_memory.py) checks the transcript is resent, that a remembered preference appears in the *next turn's* system prompt, that preferences accumulate and update, that a new thread starts empty, and the flow through the gateway with `vdreamer run --thread`:

```python
# file: tests/tutorials/test_t14_short_term_memory.py
    assert (await session.state()).values["preferences"] == {"language": "German"}
    system = model.prompts[2][0]
    assert isinstance(system, SystemMessage)
    assert system.text == f"{BASE_PROMPT} Always respect the customer's preferences — " \
                          "language: German."
```

```bash
uv run pytest tests/tutorials/test_t14_short_term_memory.py -v
```

## Common mistakes and debugging

- **`Command` without a `ToolMessage`** — LangChain refuses it (`Expected to have a matching ToolMessage in Command.update …`); vdreamer contains that as a `tool_error`, and the state is not updated. Always include one with `runtime.tool_call_id`.
- **Writing to state from a tool by mutation** (`runtime.state["preferences"][k] = v`) — not saved. Return a `Command(update=...)`.
- **A custom state key `create_agent` does not know** — the update for it is silently dropped, with no error. Declare every key in `state_schema`, and test that it is actually saved (as `test_a_remembered_preference_shapes_every_later_prompt` does).
- **Expecting memory across conversations** — each `vdreamer run` (each new thread) starts empty. Reattach with `--thread`, or use long-term memory (lesson 15).
- **Remembering secrets or sensitive data** — thread state is stored in every checkpoint. Do not keep what you would not want in your database.
- **Pydantic serializer `UserWarning`s in test output** for tools with an injected `ToolRuntime` — they come from library internals serializing vdreamer's run context, are harmless, and also appear in the repository's own test suite.

## Design trade-offs

- **Transcript only vs. working memory.** The transcript is free and complete but long and, once trimmed, forgetful. Working memory costs a tool call (or an extraction step, lesson 13) and a design decision about what to keep.
- **Model-decided vs. code-decided memory.** Here the model decides when to call `remember_preference`; it may miss one. Rule-based extraction (lesson 13) is reliable for patterns, the model for open-ended facts.
- **System prompt vs. messages.** Restating facts in the system prompt keeps them prominent; appending reminders to the conversation would grow it with every turn.

## Exercises

1. Add `forget_preference(topic)` and test it.
2. Cap working memory at five preferences (oldest dropped) and test the cap.
3. Remember the order the customer is asking about (`current_order_id`) and have `get_order_status` (lesson 2) default to it when the model omits the id.

Next: [Lesson 15 — Persistent long-term memory](../t15_long_term_memory/README.md).
