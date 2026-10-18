# Lesson 22 — A supervisor coordinating specialists

> **Outcome:** one supervisor agent that answers any shop question by delegating to the lesson-21 specialists — each wrapped as a tool described by its responsibility — asking several in parallel and composing one answer, with every nested tool call governed and visible to clients under its own namespace.

## Learning objectives

- Build the **supervisor** (agents-as-tools) pattern with `create_agent`: one tool per specialist.
- Describe specialists to the supervisor from data, and keep each delegation self-contained.
- Know what clients see from nested agents (`ns`), and choose what to show.
- Test multi-agent systems deterministically with one script per agent (`RoutedChatModel`).

## Prerequisites

- Lesson [21](../t21_specialists/README.md) (the specialists), lesson [7](../t07_multiple_tools/README.md) (parallel tool calls) and lesson [6](../t06_testing_and_traces/README.md) (scripted models).

## Concepts

A **supervisor** is an agent whose tools are other agents. The customer talks to the supervisor; the supervisor decides whom to ask, asks (in parallel when questions are independent), and writes the final answer. The specialists never talk to the customer directly.

Wrapping a specialist as a tool gives you everything you already know about tools:

- **Selection by description** — the tool's description is the specialist's one-sentence responsibility (lesson 21), so the supervisor chooses exactly as a ReAct agent chooses tools.
- **Governance** — the delegation is a governed call: `tool.call`/`tool.result` events, a timeout for the whole sub-conversation, and containment if the specialist fails.
- **Parallel fan-out** — two `ask_*` calls in one step run concurrently (lesson 7).

**Context isolation.** Each delegation starts a fresh conversation containing only the supervisor's question. Specialists cannot be confused by unrelated history, and they cannot leak it. The price: the supervisor must write *self-contained* questions ("Is the family tent (TENT-4P) in stock?", not "is it in stock?"). The prompt says so.

**What clients see.** The nested specialist's events — its tool calls, results and its answer — stream to the client too, each tagged with a namespace (`ns`, e.g. `["tools:…"]`) that identifies the nested run; the supervisor's own events have `ns == []`. A client can show nested work in a collapsible panel, or hide it. `vdreamer run` prints everything, so you will see the specialists' answers before the supervisor's. To hide specialists' text entirely, give them a `nostream` copy of the model (lesson 16) — their governed tool calls stay visible.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `create_agent`, tools that call other agents, parallel tool execution | LangChain / LangGraph |
| Governed delegations (events, timeout, containment), `VDreamerToolMiddleware` for the nested specialists | vdreamer |
| Nested events tagged with `ns` | vdreamer (the stream driver keeps LangGraph's namespaces) |
| `make_delegate`, the supervisor prompt; `RoutedChatModel` in tests | this lesson / the series' test support |

## How it works

```
customer: "Is the family tent in stock, and can I return it if it's too big?"
supervisor ─► ask_products("Is the family tent (TENT-4P) in stock?")  ┐ parallel
           ─► ask_policies("What is the returns policy?")             ┘
                products specialist ─► search_catalog("family tent") ─► "TENT-4P: 3 in stock, EUR 289."
                policies specialist ─► get_policy("returns")         ─► "Unused items … within 30 days …"
supervisor ─► "Yes — 3 family tents are in stock (EUR 289), and unused items can be returned within 30 days."
```

## Step by step

**1. A specialist behind a tool.** The specialist is built once (when the supervisor's graph is built), with governance middleware and no state; each call is a fresh conversation:

```python
# file: agent.py
def make_delegate(spec: Specialist, model: BaseChatModel, timeout_s: float = 90) -> BaseTool:
    """``ask_<name>``: one specialist behind a tool. Each call is a fresh conversation."""
    specialist = build_specialist(spec, model, middleware=[VDreamerToolMiddleware()])

    @vdreamer.tool(
        f"ask_{spec.name}",
        description=f"{spec.responsibility} Ask one self-contained question.",
        timeout_s=timeout_s,
    )
    async def delegate(question: str) -> str:
        result = await specialist.ainvoke({"messages": [{"role": "user", "content": question}]})
        return str(result["messages"][-1].text)

    return delegate
```

**2. A supervisor prompt rendered from the same data.**

```python
# file: agent.py
def supervisor_prompt(specialists: tuple[Specialist, ...]) -> str:
    team = "\n".join(f"- ask_{s.name}: {s.responsibility}" for s in specialists)
```

**3. The supervisor itself is an ordinary `create_agent` agent.**

```python
# file: agent.py
        model = ctx.models["chat"]
        return create_agent(
            model,
            tools=[make_delegate(spec, model) for spec in SPECIALISTS],
            system_prompt=supervisor_prompt(SPECIALISTS),
            **ctx.agent_kwargs(),
        )
```

## The complete example

[`agent.py`](agent.py) — `make_delegate`, `supervisor_prompt`, `SupportSupervisorAgent` (`t22-support-supervisor`). The specialists come from [lesson 21](../t21_specialists/specialists.py).

## Run it

From `tutorials/`:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t22-support-supervisor --models config/models.ollama.yaml
uv run vdreamer run t22-support-supervisor \
    --message "Is the family tent in stock, and can I return it if it's too big?"
```

## Expected behavior

```
-> ask_products({"question": "Is the family tent (TENT-4P) in stock?"}) [custom]
-> ask_policies({"question": "What is the returns policy?"}) [custom]
-> search_catalog({"query": "family tent"}) [custom]
-> get_policy({"topic": "returns"}) [custom]
<- ok [{"sku": "TENT-4P", "name": "Family tent, 4-person", "price_eur": 289.0, "stock": 3}] (3 ms)
<- ok Unused items can be returned within 30 days of delivery. (3 ms)
assistant>
TENT-4P: 3 in stock, EUR 289.                      ← the products specialist (ns: tools:…)

assistant>
Unused items can be returned within 30 days of delivery.   ← the policies specialist

<- ok TENT-4P: 3 in stock, EUR 289. (20 ms)
<- ok Unused items can be returned within 30 days of delivery. (19 ms)
assistant>
Yes — 3 family tents are in stock (EUR 289), and unused items can be returned within 30 days of delivery.
```

(The arrows are annotations.) With a real model, the supervisor's wording and whether it asks in parallel will vary.

## Test it

[`test_t22_supervisor.py`](../../../../tests/tutorials/test_t22_supervisor.py) gives each agent its own script with `RoutedChatModel` — a call goes to the script whose key appears in that call's system prompt — so parallel delegations stay deterministic:

```python
# file: tests/tutorials/test_t22_supervisor.py
    # Each specialist saw only the supervisor's question, not the customer's message.
    products_prompt = routes[PRODUCTS].prompts[0]
    assert [m.text for m in products_prompt if isinstance(m, HumanMessage)] == [
        "Is the family tent (TENT-4P) in stock?"
    ]
```

Other tests check the tool descriptions, the namespaces of nested events, a failing inner lookup contained for the supervisor, and the gateway.

```bash
uv run pytest tests/tutorials/test_t22_supervisor.py -v
```

## Common mistakes and debugging

- **Vague delegations** ("is it in stock?") — the specialist has no context. Tell the supervisor to write self-contained questions, and test what specialists receive.
- **The supervisor answers from its own knowledge** — give it no lookup tools of its own, and say "using only what the specialists said".
- **A delegation timeout shorter than the specialist's work** — the specialist's tools and model calls all run inside `timeout_s`. Budget for the whole sub-conversation.
- **Building the specialist inside the tool function** — rebuilt on every call. Build once in `make_delegate`.
- **Duplicate-looking answers in the client** — nested messages are streamed with their `ns`. Render or hide by namespace, or silence the specialists' text with a `nostream` model.

## Design trade-offs

- **Supervisor vs. handoffs (lesson 23).** A supervisor keeps control and composes one answer, at the cost of an extra model call per turn and a "telephone game" between agents. Handoffs let a specialist talk to the customer directly — fewer calls, less control.
- **Fresh context per delegation** is simple and safe; passing the conversation (or a summary) helps follow-up questions but costs tokens and isolation.
- **One model for all** is simplest to deploy; a smaller model for specialists with narrow jobs can cut cost noticeably.

## Exercises

1. Hide specialists' answers from clients with `background_model` (lesson 16); keep their tool calls visible. Update the namespace test.
2. Add the `delivery` specialist from lesson 21's exercise; the supervisor prompt and tools update from data alone.
3. Let the supervisor pass a short `context` string with each question and measure (with the scripted prompts) what the specialist now sees.

Next: [Lesson 23 — Handoffs, shared state and communication](../t23_handoffs/README.md).
