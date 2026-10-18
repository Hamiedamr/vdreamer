# Lesson 13 — Agent state and conversation history

> **Outcome:** a support agent that keeps typed facts per conversation — the customer's email, the topics raised, a turn count — updated by reducers, checkpointed after every step, isolated per thread, and visible to operators through the API but never streamed to clients.

## Learning objectives

- Design a state schema beyond `messages`, and choose for each key how updates combine: **overwrite**, **append**, or a **custom reducer**.
- Understand what a checkpointer stores (a snapshot after every step) and how vdreamer maps threads to it.
- Read a thread's current state and its history (`aget_state`, `aget_state_history`, `GET /api/v1/threads/{id}`).
- Keep deterministic bookkeeping out of the model, and feed the model state-derived facts.

## Prerequisites

- Lessons [1](../t01_serving_an_agent/README.md) (threads, `add_messages`) and [3](../t03_react_loop/README.md) (a hand-written graph, state persisting across turns).

## Concepts

A LangGraph **state** is a `TypedDict`. Nodes never mutate it; each returns a *partial update*, and LangGraph merges the update into the state key by key:

| Key | Declared as | An update… |
| --- | --- | --- |
| `customer_email` | `str \| None` (no reducer) | **replaces** the value: the latest write wins |
| `messages` | `Annotated[list[AnyMessage], add_messages]` | **appends** messages (or replaces one with the same id; `RemoveMessage` deletes) |
| `topics` | `Annotated[list[str], merge_unique]` | is combined by **your function** `(old, new) → merged` |

Reducers matter most when several nodes write the same key in one step (parallel branches, lesson 20): without a reducer that is an error; with one, the writes are combined.

The **checkpointer** saves a snapshot of the whole state after every superstep, keyed by `thread_id`. vdreamer owns that mapping: each API thread is a LangGraph thread, and the checkpointer is SQLite in `vdreamer dev` and Postgres in production. That is what makes three things work: the next run on a thread continues from the saved state; a crashed run resumes from its last step (lesson 29); and an interrupted run waits for approval with nothing in memory (lesson 25).

Two things that follow:

- **State is per thread, not per user.** A new conversation starts empty. Facts that should outlive a conversation belong in long-term memory (lesson 15).
- **State is private.** Clients see events (messages, tool calls, progress…), not state keys. Operators can read a thread's current `values` with `GET /api/v1/threads/{id}`; your client can too, if you build it to — but nothing is streamed.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `TypedDict` state, `Annotated[..., reducer]`, `add_messages`, checkpoints, `aget_state`/`aget_state_history` | LangGraph |
| Durable checkpointer per deployment (SQLite in dev, Postgres in production), thread ↔ checkpoint mapping | vdreamer |
| `GET /api/v1/threads/{id}` (runs + current `values`) | vdreamer gateway |
| The extract/respond design, `merge_unique` | this lesson's code |

## How it works

Two runs on one thread:

```
run 1 "Hi, I'm ada@example.com, a refund question"
  extract → {turn: 1, topics: ["refunds"], customer_email: "ada@example.com"}
  respond → system prompt: "Turn 1 … Customer email: ada@example.com. Topics so far: refunds."
  checkpoints: input, after extract, after respond, final
run 2 "When will it be delivered?"
  (state restored from the last checkpoint)
  extract → {turn: 2, topics: ["delivery"]}           ← merge_unique: ["refunds", "delivery"]
  respond → "Turn 2 … Topics so far: refunds, delivery."
```

## Step by step

**1. Declare the state with a policy per key.**

```python
# file: agent.py
class SupportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # reducer: append (or replace by id)
    topics: NotRequired[Annotated[list[str], merge_unique]]  # reducer: union
    customer_email: NotRequired[str | None]  # no reducer: the latest write wins
    turn: NotRequired[int]
```

`NotRequired` keys may be absent until a node writes them; read them with `state.get(...)`.

**2. Write the reducer** — a pure function of the old value and the update:

```python
# file: agent.py
def merge_unique(existing: list[str], new: list[str]) -> list[str]:
    """A reducer: keep every topic once, in the order first raised."""
    return existing + [topic for topic in new if topic not in existing]
```

**3. Do bookkeeping deterministically.** Counting turns or spotting an email needs no model; a plain node is cheaper, faster and testable:

```python
# file: agent.py
def extract(state: SupportState) -> dict[str, Any]:
    """Deterministic bookkeeping: no model needed to count turns or spot an email."""
    latest = state["messages"][-1].text
    update: dict[str, Any] = {
        "turn": state.get("turn", 0) + 1,
        "topics": [topic for word, topic in TOPIC_WORDS.items() if word in latest.lower()],
    }
    if found := EMAIL.search(latest):
        update["customer_email"] = found.group(0).lower()
    return update
```

**4. Give the model the facts, not just the transcript.** The system prompt is rebuilt every turn from state:

```python
# file: agent.py
def facts(state: SupportState) -> str:
    """What the model is told about this thread, from state rather than re-reading history."""
```

## The complete example

[`agent.py`](agent.py) — `SupportState`, `merge_unique`, `extract`, `facts`, and `SupportThreadAgent` (`t13-support-thread`: `extract → respond`).

## Run it

From `tutorials/`:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t13-support-thread --models config/models.ollama.yaml
uv run vdreamer run t13-support-thread
```

At the `you>` prompt, send a few messages (`Hi, I'm ada@example.com, a refund question`, then `When will it be delivered?`). Then read the thread's state (the thread id is printed by `vdreamer run`):

```bash
curl -s -H "Authorization: Bearer tutorial-token-change-me-0123456789" -H "X-User-Id: $USER" \
     http://127.0.0.1:8000/api/v1/threads/<thread_id> | python -m json.tool
```

## Expected behavior

After two messages the thread's `values` hold:

```json
{"messages": [...4 messages...], "topics": ["refunds", "delivery"],
 "customer_email": "ada@example.com", "turn": 2}
```

None of `topics`, `customer_email` or `turn` appears in the `vdreamer run` output: they are state, not events. A new `vdreamer run` (a new thread) starts again at turn 1 with no email.

## Test it

[`test_t13_state_and_history.py`](../../../../tests/tutorials/test_t13_state_and_history.py) tests the reducer and `extract` as plain functions, then three runs on one thread (overwrite, append and merge all at once), isolation between threads, the checkpoint history, and the gateway's thread endpoint. The history test shows the four checkpoints each run leaves behind:

```python
# file: tests/tutorials/test_t13_state_and_history.py
    history = [snap async for snap in session.runner.compiled.aget_state_history(config)]
    # Per run: the input, extract, respond, and the final state — newest first.
    assert [snap.next for snap in history[:4]] == [(), ("respond",), ("extract",), ("__start__",)]
```

The multi-turn test also found a bug in the first version of the email pattern, which kept a sentence's full stop (`ada.lovelace@example.com.`); the fix has its own assertion now.

```bash
uv run pytest tests/tutorials/test_t13_state_and_history.py -v
```

## Common mistakes and debugging

- **A list key without a reducer** — each write replaces the whole list, so "topics so far" only ever holds the latest turn's. Decide per key: overwrite, append, or merge.
- **Mutating state in place** (`state["topics"].append(...)`) — the change may not be saved and breaks reducers. Return an update.
- **`KeyError` on a key not written yet** — use `NotRequired` and `state.get(...)` with a default.
- **Counters that never reset** — state persists across turns on a thread (lesson 3's step budget). Reset in an entry node or derive per-turn values.
- **Putting large data in state** — every checkpoint stores the whole state, after every step. Keep state small; put big results in artifacts (lesson 5) or a store (lesson 15).
- **Expecting state in the client** — it is not streamed. Emit what users should see (messages, progress, artifacts).

## Design trade-offs

- **Extracted facts vs. re-reading the transcript.** Facts in state are cheap and exact for the model to use; but you must decide what to extract. The transcript is complete but grows (lesson 16).
- **Rule-based vs. model-based extraction.** A regex is free and deterministic but brittle (as the bug showed); a model call (with structured output, lesson 18) handles phrasing variety at a cost. Many agents use both: rules first, model for the rest.
- **Checkpoint every step** (`durability="sync"`, vdreamer's default) is what makes crash recovery and approvals safe; `"async"` trades a little durability for latency (lesson 29).

## Exercises

1. Add `order_ids: Annotated[list[str], merge_unique]`, filled from `A\d{4}` patterns, and include them in `facts`.
2. Add a `"reset"` command: when the user writes `/reset`, an entry node clears `topics` (hint: a reducer cannot *remove* — make the update replace, or use a sentinel).
3. Using `aget_state_history`, write a test that finds the checkpoint taken just after `extract` in the first run and checks its `turn`.

Next: [Lesson 14 — Short-term conversational memory](../t14_short_term_memory/README.md).
