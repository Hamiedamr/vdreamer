# Lesson 23 — Handoffs, shared state and communication

> **Outcome:** a support desk without a supervisor: the active specialist talks to the customer directly and hands the conversation to a colleague with `transfer_to_<name>(note)`; notes travel through shared state; the next turn continues with whoever was speaking; and the client sees each message exactly once.

## Learning objectives

- Implement **handoffs**: a specialist ends its turn and names who continues.
- Share state between agents: a common state schema, notes for the next specialist, the active agent across turns.
- Wrap agents as graph nodes that return only what changed, and know why that matters under vdreamer's streaming.
- Compare handoffs with the supervisor pattern (lesson 22).

## Prerequisites

- Lessons [21](../t21_specialists/README.md) (specialists), [22](../t22_supervisor/README.md) (supervisor), [14](../t14_short_term_memory/README.md) (`Command(update=...)`, `ToolRuntime`, `@dynamic_prompt`) and [13](../t13_state_and_history/README.md) (shared state, routing).

## Concepts

In the supervisor pattern every message passes through a coordinator. With **handoffs**, specialists pass the conversation along themselves: the orders specialist realises the customer wants a different product and hands over to the products specialist, who answers directly. Fewer model calls per turn, a more natural conversation — and less central control.

Three mechanisms make it work:

1. **A handoff tool** per colleague, `transfer_to_<name>(note)`. It returns `Command(update={...})` that records the target and the note in shared state, and it is `return_direct=True`, so the current specialist's turn ends immediately instead of calling its model again.
2. **Shared state** (`DeskState`) that every specialist reads and writes: the messages, `active_agent` (who answers next), `handoff_to` (set during a step), `handoff_notes` (what colleagues said). A `@dynamic_prompt` adds the notes to each specialist's system prompt — that is how agents *communicate*.
3. **Graph routing**: after each specialist node, a conditional edge goes to `handoff_to` or ends the turn; at the start of a turn, to `active_agent`.

**Return only what changed.** A `create_agent` graph used directly as a node returns its *whole* state, including every earlier message. vdreamer streams every AI message that appears in a node's update, so earlier answers would be sent to the client again on every turn (a test in this lesson shows it). Each specialist therefore runs inside a small node function that returns only new messages and the shared keys.

LangChain's documentation also shows handoffs that jump with `Command(goto=…, graph=Command.PARENT)` from inside a sub-agent. Under vdreamer that pattern works, but with the whole-state return described above; the explicit routing here keeps the client stream clean and the control flow visible in the graph.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `Command(update=...)`, `StateGraph`, conditional routing, `END` | LangGraph |
| `create_agent(state_schema=...)`, `ToolRuntime`, `@dynamic_prompt`, `return_direct` | LangChain |
| Governance of every specialist tool call (`VDreamerToolMiddleware`), namespaced events, streaming of AI messages in updates | vdreamer |
| The handoff protocol and the node wrapper | this lesson's code |

## How it works

```
turn 1: "I ordered A1001 but I need a bigger tent"       active_agent: (none) → orders
  orders   ─► transfer_to_products(note="Has order A1001 (TENT-2P); wants a bigger tent.")
             state: handoff_to=products, active_agent=products, handoff_notes=[note]
  products ─► (prompt + "Notes from colleagues: …") search_catalog("family tent") ─► answer
turn 2: "How much is it?"                                 active_agent: products
  products ─► "EUR 289."                                   (only this message is streamed)
```

## Step by step

**1. The shared state.**

```python
# file: agent.py
class DeskState(AgentState[Any]):
    """Shared by every specialist: the conversation plus how it is being handled."""

    active_agent: NotRequired[str]  # who answers the next message
    handoff_to: NotRequired[str | None]  # set by a transfer during this step
    handoff_notes: NotRequired[list[str]]  # what each specialist told the next
```

**2. A handoff tool per colleague**, described by the colleague's responsibility:

```python
# file: agent.py
    @vdreamer.tool(
        f"transfer_to_{target.name}",
        description=f"Hand the customer over to the {target.name} specialist: "
                    f"{target.responsibility} Write a note with what they need to know.",
        return_direct=True,
    )
    def handoff(note: str, runtime: ToolRuntime) -> Command[Any]:
        done = ToolMessage(f"Transferred to {target.name}.", tool_call_id=runtime.tool_call_id)
        notes = [*runtime.state.get("handoff_notes", []), note]
        return Command(update={"messages": [done], "handoff_to": target.name,
                               "handoff_notes": notes})
```

**3. Communicate through the prompt.**

```python
# file: agent.py
@dynamic_prompt
def prompt_with_notes(request: ModelRequest) -> str:
    """Each specialist's own prompt, plus the notes colleagues left for it."""
    notes = request.state.get("handoff_notes", [])
    shared = "\nNotes from colleagues: " + " | ".join(notes) if notes else ""
    return str(request.system_prompt) + shared
```

**4. Run each specialist on the shared state, return only what changed.**

```python
# file: agent.py
    async def node(state: DeskState) -> dict[str, Any]:
        out = await agent.ainvoke({**state, "handoff_to": None})
        known = {message.id for message in state["messages"]}
        target = out.get("handoff_to")
        return {
            "messages": [m for m in out["messages"] if m.id not in known],
            "handoff_to": target,
            "active_agent": target or name,
            "handoff_notes": out.get("handoff_notes", []),
        }
```

**5. Route.** After a specialist: to the colleague, or end the turn. At the start of a turn: to the active specialist.

```python
# file: agent.py
def route_after(state: DeskState) -> str:
    return state.get("handoff_to") or END


def route_first(state: DeskState) -> str:
    return state.get("active_agent", FIRST_CONTACT)
```

## The complete example

[`agent.py`](agent.py) — `DeskState`, `make_handoff`, `prompt_with_notes`, `build_desk_agent`, `as_node`, the routers, and `SupportDeskAgent` (`t23-support-desk`). The specialists' tools and prompts come from [lesson 21](../t21_specialists/specialists.py).

## Run it

From `tutorials/`:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t23-support-desk --models config/models.ollama.yaml
uv run vdreamer run t23-support-desk
```

Try `I ordered A1001 but I need a bigger tent`, then `How much is it?`, then `And can I return the old one?`.

## Expected behavior

```
-> transfer_to_products({"note": "Has order A1001 (TENT-2P, 2-person); wants a bigger tent."}) [custom]
<- ok  (2 ms)
-> search_catalog({"query": "family tent"}) [custom]
<- ok [{"sku": "TENT-4P", "name": "Family tent, 4-person", "price_eur": 289.0, "stock": 3}] (1 ms)
assistant>
Our Family tent, 4-person (TENT-4P) is in stock: 3 left at EUR 289.
```

The next message goes straight to the products specialist. A returns question should be handed on to the policies specialist with a note.

## Test it

[`test_t23_handoffs.py`](../../../../tests/tutorials/test_t23_handoffs.py) checks the handoff (target, shared note, the note in the next specialist's prompt), continuity across turns with nothing re-sent, the full handoff mesh, and — as documentation — what happens without the node wrapper:

```python
# file: tests/tutorials/test_t23_handoffs.py
    await session.run("Can I return things?")
    second = await session.run("Even if I used it once?")

    # Only the new answer reaches the client: no earlier message is sent again.
    assert [p["text"] for p in second.payloads("message.completed")] == ["Yes, unused only."]
```

```bash
uv run pytest tests/tutorials/test_t23_handoffs.py -v
```

## Common mistakes and debugging

- **The specialist keeps talking after a transfer** — the handoff tool is not `return_direct`, so the agent calls its model again and answers anyway. End the turn on transfer.
- **Earlier answers appear again on every turn** — a sub-agent graph added directly as a node returns the whole conversation. Return only new messages.
- **The conversation snaps back to the first specialist** — `active_agent` is not saved. Record who is speaking after every node.
- **Notes duplicated** — a list key with an appending reducer receives the sub-agent's full list back. Overwrite the list (as here) or deduplicate.
- **Endless ping-pong** (A → B → A …) — two specialists each think the other is responsible. LangGraph's recursion limit eventually fails the run; lesson 24 adds a per-turn handoff budget.

## Design trade-offs

- **Handoffs vs. supervisor.** Handoffs: one specialist answers, fewer calls, conversational. Supervisor: one coordinator composes answers from several specialists, more control and consistency. Mixed questions ("status of A1001 *and* the returns policy") suit a supervisor better.
- **Notes vs. the full transcript.** Every specialist here sees the whole conversation *and* the notes. Notes make the reason for a transfer explicit; the transcript keeps details. For long threads, combine with summaries (lesson 16).
- **Explicit routing vs. `Command.PARENT` jumps.** Explicit routing shows every transfer in the graph and keeps streaming clean; parent jumps need less code inside the graph and more care with what reaches clients.

## Exercises

1. Add a `transfer_to_human` handoff that ends the turn with a message and sets `active_agent` to `"human"` (and make `route_first` stop the agents answering).
2. Clear `handoff_notes` when the customer starts a new topic (hint: a `/new` command, or a classifier).
3. Count handoffs per turn and refuse a third one — then compare with lesson 24's approach.

Next: Lesson 24 — Handling failures and controlling multi-agent execution.
