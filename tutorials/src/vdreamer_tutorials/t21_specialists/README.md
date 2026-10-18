# Lesson 21 — Specialised agents with clear responsibilities

> **Outcome:** three specialist agents — orders, products, policies — each with one area of responsibility, its own tools and prompt, a one-sentence description for other agents, served and tested on its own; the building blocks of the multi-agent systems in lessons 22–24.

## Learning objectives

- Decide when one agent should become several, and draw boundaries between them.
- Describe a specialist as data (`Specialist`: name, responsibility, prompt, tools) and build it with one factory.
- Give every tool exactly one owner, and test that boundary.
- Serve the same specialist standalone (with `ctx.agent_kwargs()`) or nested (with governance only).

## Prerequisites

- Lessons [7](../t07_multiple_tools/README.md)–[9](../t09_tool_selection/README.md) (tool sets, contracts, selection) and [11](../t11_package_structure/README.md) (package structure).

## Concepts

One agent with every tool works until it does not: past a dozen tools the model picks wrong more often, the prompt becomes a patchwork of rules for unrelated areas, and every change risks every behavior. **Specialists** split the problem:

- **One responsibility each**, stated in one sentence. If you need "and" twice, split again.
- **Their own tools.** A tool has exactly one owner, so each specialist's choice is small and unambiguous.
- **Their own prompt**, including what to do with out-of-scope requests ("say it is outside your area" — not "try anyway").
- **Independently testable and deployable.** Each specialist is a complete agent; you can serve it alone, test it alone, and swap its model alone.

The `responsibility` sentence is written for *other agents*: in lesson 22 a supervisor reads it to decide whom to ask.

A specialist built with `create_agent` can run in two ways:

- **Standalone** — served by vdreamer as its own agent (`t21-orders`): pass `**ctx.agent_kwargs()` so it gets vdreamer's checkpointer, store, run context and tool governance.
- **Nested** — called by another agent (lessons 22–24): pass only the governance middleware (`VDreamerToolMiddleware`). It then keeps no state between calls; the parent's thread holds the conversation, and the specialist's tool calls are still governed and reported, tagged with the nesting namespace (`ns`).

## What comes from where

| Piece | Provided by |
| --- | --- |
| `create_agent(..., name=...)`, prompts, tools | LangChain |
| One `BaseAgent` class serving several specialists (instance `name`) | vdreamer (plain attributes) |
| `ctx.agent_kwargs()`, `VDreamerToolMiddleware`, governance of nested tool calls | vdreamer |
| The `Specialist` description and boundaries | this lesson's design |

## How it works

```
Specialist(name, responsibility, prompt, tools) ──build_specialist(spec, model, **kwargs)──► create_agent graph

  orders    "Answers questions about existing orders…"      tools: get_order
  products  "Answers questions about products…"             tools: search_catalog
  policies  "Quotes the shop's returns, shipping and warranty policies."   tools: get_policy

served alone:   SpecialistAgent(ORDERS_SPECIALIST) → "t21-orders"   (**ctx.agent_kwargs())
nested (22–24): build_specialist(spec, model, middleware=[VDreamerToolMiddleware()])
```

## Step by step

**1. Describe a specialist as data.**

```python
# file: specialists.py
@dataclass(frozen=True)
class Specialist:
    name: str
    responsibility: str  # one sentence: what it does, used by supervisors (lesson 22)
    prompt: str
    tools: tuple[BaseTool, ...]
```

**2. Write each one with its boundary in the prompt.**

```python
# file: specialists.py
ORDERS_SPECIALIST = Specialist(
    name="orders",
    responsibility="Answers questions about existing orders: status, contents and totals.",
    prompt="You handle questions about existing orders only. Look orders up with get_order. "
           "If asked about anything else, say it is outside your area.",
    tools=(get_order,),
)
```

**3. One factory for all of them.**

```python
# file: specialists.py
def build_specialist(spec: Specialist, model: BaseChatModel, **agent_kwargs: Any) -> Any:
```

**4. Serve each one.** One `BaseAgent` class, three instances; the agent's `name` is an instance attribute here:

```python
# file: agent.py
class SpecialistAgent(BaseAgent):
    """Serves one ``Specialist`` as a standalone vdreamer agent."""

    mode = Mode.CHAT
    models = ("chat",)

    def __init__(self, spec: Specialist) -> None:
        self.spec = spec
        self.name = f"t21-{spec.name}"

    def build(self, ctx: BuildContext) -> Any:
        return build_specialist(self.spec, ctx.models["chat"], **ctx.agent_kwargs())
```

## The complete example

- [`specialists.py`](specialists.py) — sample data, the three tools, the three `Specialist`s, `build_specialist`.
- [`agent.py`](agent.py) — `SpecialistAgent` and the instances `t21-orders`, `t21-products`, `t21-policies`.

## Run it

From `tutorials/`, one specialist at a time:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t21-orders --models config/models.ollama.yaml
uv run vdreamer run t21-orders --message "What is in order A1001?"
uv run vdreamer run t21-orders --message "What is your returns policy?"   # outside its area
```

## Expected behavior

```
-> get_order({"order_id": "A1001"}) [custom]
<- ok {"order_id": "A1001", "status": "shipped", "items": ["TENT-2P"], "total_eur": 149.0} (1 ms)
```

Asked about returns, the orders specialist has no policy tool and should say the question is outside its area.

## Test it

[`test_t21_specialists.py`](../../../../tests/tutorials/test_t21_specialists.py) tests the boundaries themselves — one owner per tool, a stated boundary per prompt — and each specialist alone (it is offered only its own tools; it cannot run another's):

```python
# file: tests/tutorials/test_t21_specialists.py
def test_every_tool_has_exactly_one_owner() -> None:
    owners: dict[str, str] = {}
    for spec in SPECIALISTS:
        for tool in spec.tools:
            assert tool.name not in owners, f"{tool.name} owned by {owners[tool.name]} too"
            owners[tool.name] = spec.name
```

```bash
uv run pytest tests/tutorials/test_t21_specialists.py -v
```

## Common mistakes and debugging

- **Overlapping responsibilities** ("orders" and "returns" both look up orders) — a supervisor cannot choose well, and both drift. Give each tool one owner; let specialists *share data* through tools, not responsibilities.
- **Splitting too early** — two specialists with one tool each and a supervisor in front cost an extra model call per request. Split when one agent's prompt or toolbox has become hard to manage, not before.
- **No out-of-scope instruction** — a specialist asked something else will improvise an answer. Tell it how to decline.
- **Passing `**ctx.agent_kwargs()` to a nested specialist** — it would share the parent's checkpointer and thread config, mixing its own conversation state into the parent's thread. Nested specialists get only the governance middleware and stay stateless; the parent thread holds the conversation.

## Design trade-offs

- **Specialists vs. dynamic tool selection (lesson 9).** Both shrink the choice a model faces. Selection keeps one agent and one prompt; specialists also separate prompts, models and tests, at the cost of coordination (lessons 22–24).
- **Same model or different models.** Specialists can use different aliases (a small fast model for policies, a strong one for orders). More aliases means more to serve and probe.
- **Data in code.** `Specialist` as data makes the set easy to list, test and render (a supervisor's prompt in lesson 22); a class per specialist would give each more room for custom behavior.

## Exercises

1. Add a `delivery` specialist owning a `track_parcel` tool; the ownership test should still pass.
2. Give the products specialist its own model alias (`products`) and declare it in both config files.
3. Write a test asserting every specialist's responsibility mentions at least one of its tools' subjects.

Next: [Lesson 22 — A supervisor coordinating specialists](../t22_supervisor/README.md).
