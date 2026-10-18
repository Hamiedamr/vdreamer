# Lesson 15 — Persistent long-term memory

> **Outcome:** an assistant that remembers facts about each customer across conversations and restarts — saved by a tool into vdreamer's store under a per-user namespace, loaded into every prompt, isolated between customers, and deletable on request.

## Learning objectives

- Tell short-term memory (thread state, lesson 14) from long-term memory (a **store** shared across threads).
- Use the store vdreamer provides (`ctx.store`, wired by `ctx.agent_kwargs()`): `aput`, `asearch`, `adelete` from tools and middleware.
- Namespace memory by the **authenticated** user so customers can never read each other's notes.
- Load memories into the prompt with an async `@wrap_model_call` middleware.
- Know what the store is in each deployment and what that means for durability.

## Prerequisites

- Lessons [14](../t14_short_term_memory/README.md) (`ToolRuntime`, prompt middleware) and [9](../t09_tool_selection/README.md) (`request.runtime.context`, roles from the caller).

## Concepts

Thread state ends with the conversation. Some facts should not: a customer's shoe size, that they camp in winter, that they prefer metric units. Those belong in **long-term memory**: a key–value store organised in **namespaces** (tuples like `("customers", "ada", "notes")`), independent of threads.

vdreamer gives every agent a LangGraph store as `ctx.store` — the same database as the run state: SQLite in the `vdreamer dev` file, Postgres (`AsyncPostgresStore`) for `vdreamer serve`/`worker`. `ctx.agent_kwargs()` passes it to `create_agent`, which makes it available to tools as `runtime.store` and to middleware as `request.runtime.store`. vdreamer's own runtime is asynchronous, so use the async methods (`aput`, `asearch`, `adelete`).

**Namespace by identity, from a trusted source.** The namespace must come from the authenticated user (`VDreamerContext.user_id`, set by the gateway from `X-User-Id`, which your trusted backend asserts), never from anything the model or the user typed — otherwise "remember that I am lin@example.com" becomes a way to read someone else's memory. No user, no memory: a run without a user id must not fall back to a shared namespace.

**Writing vs. reading.** The model decides *what* to save (a tool call). Your code decides *when to read*: here, middleware loads the customer's notes into the system prompt before every model call — the model does not have to remember to look.

**Memory is data you are responsible for.** It persists, it is about people, and users may ask you to delete it. Provide a way to forget (here a tool; in production also an admin path), and do not store what you would not want to keep.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `ctx.store` — SQLite in `vdreamer dev`, Postgres in production — wired via `agent_kwargs()` | vdreamer |
| `VDreamerContext.user_id` from the authenticated caller | vdreamer gateway |
| `BaseStore` API: `aput`, `asearch`, `adelete`, namespaces, `InMemoryStore`, `AsyncSqliteStore` | LangGraph |
| `runtime.store` in tools, `request.runtime.store` in middleware, `@wrap_model_call` | LangChain |
| Namespacing scheme, notes tools, prompt loading | this lesson's code |

## How it works

```
conversation 1 (user ada, thread monday)
  model ─► save_customer_note("camps in winter, below -5 °C")
  tool  ─► store.aput(("customers", "ada", "notes"), <uuid>, {"note": …})
conversation 2 (user ada, thread tuesday)               conversation 3 (user lin)
  middleware ─► store.asearch(("customers","ada","notes"))  middleware ─► asearch(("customers","lin","notes")) → []
  system prompt += "What you know about this customer: - camps in winter, below -5 °C"
```

## Step by step

**1. One function decides the namespace** — and refuses when there is no user:

```python
# file: agent.py
def notes_namespace(user_id: str | None) -> tuple[str, ...]:
    """Where one customer's notes live. No user, no memory: never a shared namespace."""
    if not user_id:
        raise ToolException("Long-term memory needs a signed-in customer.")
    return ("customers", user_id, "notes")
```

**2. A tool that writes**, with the user id from the run context, not from the model:

```python
# file: agent.py
@vdreamer.tool(parse_docstring=True)
async def save_customer_note(note: str, runtime: ToolRuntime) -> str:
    """Save a durable fact about the customer for future conversations.

    Args:
        note: One short fact, e.g. "camps in winter, below -5 °C".
    """
    namespace = notes_namespace(vdreamer.current_context().user_id)
    await _store(runtime).aput(namespace, uuid.uuid4().hex, {"note": note})
    return f"Saved: {note}"
```

**3. Middleware that reads before every model call:**

```python
# file: agent.py
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
```

`recall_notes` caps the list (`MAX_NOTES_IN_PROMPT`): memory that grows without bound would eventually crowd out the conversation.

**4. Pass the store through.** `ctx.agent_kwargs()` already contains `store=ctx.store`; nothing else is needed.

## The complete example

[`agent.py`](agent.py) — `notes_namespace`, `save_customer_note`, `forget_customer_notes`, `recall_notes`, `prompt_with_customer_notes`, and `CustomerMemoryAgent` (`t15-customer-memory`).

## Run it

From `tutorials/`. `--user` sets the caller's identity (`X-User-Id`):

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t15-customer-memory --models config/models.ollama.yaml
uv run vdreamer run t15-customer-memory --user ada --once --message "I always camp in winter, often below -5"
uv run vdreamer run t15-customer-memory --user ada --once --message "Which sleeping bag should I buy?"
uv run vdreamer run t15-customer-memory --user lin --once --message "Which sleeping bag should I buy?"
```

Stop and restart `vdreamer dev` between the first two commands: the note survives, because the store lives in `./.vdreamer/dev.db`.

## Expected behavior

The first run saves a note:

```
-> save_customer_note({"note": "camps in winter, often below -5 °C"}) [custom]
<- ok Saved: camps in winter, often below -5 °C (3 ms)
```

Ada's second conversation — a new thread — gets a winter-appropriate recommendation without being asked again; Lin's gets a generic one.

## Test it

[`test_t15_long_term_memory.py`](../../../../tests/tutorials/test_t15_long_term_memory.py) covers a note crossing threads, isolation between users, the no-user refusal, deleting only the caller's notes, persistence across a "restart" with the same `AsyncSqliteStore` file `vdreamer dev` uses, and two users through the real gateway:

```python
# file: tests/tutorials/test_t15_long_term_memory.py
    model = scripted(call("save_customer_note", note="shoe size 44"), "Noted.", "Hello!")
    session = AgentSession(agent, models={"chat": model}, store=InMemoryStore())
    await session.run("I wear size 44", user_id="ada", thread_id="a")
    await session.run("Hi", user_id="lin", thread_id="b")

    assert model.prompts[2][0].text == BASE_PROMPT
```

```bash
uv run pytest tests/tutorials/test_t15_long_term_memory.py -v
```

## Common mistakes and debugging

- **A namespace from model arguments** (`save_note(user_email, note)`) — a prompt injection away from reading another customer's memory. Take identity from `vdreamer.current_context()` / `request.runtime.context` only.
- **A shared fallback namespace** (`user_id or "anonymous"`) — every anonymous user then shares one memory. Refuse instead.
- **Sync store calls** (`store.put`, `store.search`) inside vdreamer runs — the dev and production stores are async; call `aput`/`asearch` from `async` tools and middleware.
- **`runtime.store` is `None`** — the agent was built without `**ctx.agent_kwargs()` (or a hand-written graph was not compiled by vdreamer). In tests, pass `store=InMemoryStore()` to `AgentSession`.
- **Unbounded recall** — loading every note into every prompt. Cap it, summarise old notes, or retrieve only relevant ones (with a vector index, outside vdreamer's core).
- **Storing everything the user says** — memory should be a few durable facts, not a second transcript.

## Design trade-offs

- **Prompt injection of memories vs. a recall tool.** Loading notes into every prompt is reliable and costs tokens every call; a `recall_notes` tool costs a round trip and the model may forget to call it. For a handful of short facts, load them.
- **Model-decided writes.** Flexible, but the model may save trivia or miss something important. Constrain with the prompt and the tool description; for critical facts, extract deterministically (lesson 13).
- **Semantic search.** LangGraph stores can index values with embeddings for similarity search. vdreamer leaves retrieval design to agents (it is not core); this lesson keeps exact namespaces.
- **Retention.** Long-term memory needs a deletion path and a retention policy, like any personal data.

## Exercises

1. Store notes with a `category` (`"sizes"`, `"habits"`) and let `prompt_with_customer_notes` group them.
2. Make `save_customer_note` skip exact duplicates (search first, then put).
3. Add `MAX_NOTES_PER_CUSTOMER = 50` and drop the oldest on overflow; test it.
4. Write a script that exports one customer's notes from the dev store (hint: `AsyncSqliteStore.from_conn_string(".vdreamer/dev.db")`).

Next: [Lesson 16 — Managing context growth and conversation boundaries](../t16_context_management/README.md).
