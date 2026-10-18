# Lesson 11 — Structuring an agent application as a Python package

> **Outcome:** an order helpdesk organised as a small package — domain, settings, prompts, tools, agent — where each layer imports only the layers below it, each is tested on its own, and an architecture test keeps it that way.

## Learning objectives

- Split an agent application into modules with one responsibility each, and know what belongs where.
- Keep business rules free of LangChain and vdreamer, so they are reusable and trivially testable.
- Make the vdreamer `BaseAgent` a thin wiring layer, configured through its constructor.
- Enforce the dependency direction with a test that reads the imports.
- Expose a deliberate public surface (`__init__.py`, `__all__`) and register the agent by name.

## Prerequisites

- Lessons [8](../t08_tool_contracts/README.md) (factories, protocols) and [6](../t06_testing_and_traces/README.md) (layered tests).
- Python packages and relative imports.

## Concepts

The lessons so far kept everything in one `agent.py`. That is right for a 50-line agent and wrong for a real one, where the same rules serve several agents, prompts are reviewed by non-programmers, and tests should not need a model to check a business rule.

Structure the package around **what changes for different reasons**:

| Module | Holds | Changes when |
| --- | --- | --- |
| `domain.py` | business types and rules (`Order`, `can_cancel`), repository interfaces | the business changes |
| `settings.py` | one immutable settings object | deployment needs differ (lesson 12 loads it from the environment) |
| `prompts.py` | prompt text, built from settings | you tune the agent's behavior |
| `tools.py` | the agent-facing contract: tools over the domain | what the model may do changes |
| `agent.py` | the `BaseAgent`: picks model aliases, assembles tools, prompt and middleware | wiring changes |

Dependencies point **downwards only**: `agent → tools/prompts → settings/domain`. The domain imports no LangChain or vdreamer, so it can be reused by a batch job or a REST endpoint, and its tests run in microseconds. Nothing imports `agent.py` except the package's `__init__`, which is what vdreamer loads.

The `BaseAgent` subclass keeps no logic: its constructor takes the dependencies (repository, settings) with sensible defaults, and `build(ctx)` assembles the graph. vdreamer creates the graph once per worker from the registered *instance*, so different instances (a demo, a test, production) differ only in what you pass in.

## What comes from where

| Piece | Provided by |
| --- | --- |
| Package layout, layering, `Protocol`, dataclasses | Python / this lesson's design |
| `BaseAgent`, `build(ctx)`, registry `vdreamer.yaml`, `vdreamer validate` | vdreamer |
| `create_agent`, `ToolException` | LangChain |
| The import-rule test (`ast`) | this lesson's tests |

## How it works

```
vdreamer.yaml: t11-helpdesk → vdreamer_tutorials.t11_package_structure.helpdesk:agent
                                         │
helpdesk/__init__.py ── exports agent, HelpdeskAgent, HelpdeskSettings, Order, …
   agent.py     HelpdeskAgent(orders, settings).build(ctx) ─┐
   tools.py     make_tools(orders, settings)  ◄─────────────┤
   prompts.py   system_prompt(settings)       ◄─────────────┘
   settings.py  HelpdeskSettings                ◄── prompts, tools
   domain.py    Order, OrderRepository, InMemoryOrders, can_cancel  ◄── tools
```

## Step by step

**1. The domain knows nothing about agents.**

```python
# file: helpdesk/domain.py
class OrderRepository(Protocol):
    def get(self, order_id: str) -> Order | None: ...

    def for_customer(self, email: str) -> list[Order]: ...
...
def can_cancel(order: Order) -> bool:
    """The rule: only orders that have not shipped can be cancelled."""
    return order.status == "processing"
```

**2. Settings are one frozen object, passed down explicitly** — no module reads the environment on its own:

```python
# file: helpdesk/settings.py
@dataclass(frozen=True)
class HelpdeskSettings:
    shop_name: str = "Trailhead Outfitters"
    max_orders_listed: int = 5
```

**3. Prompts are functions of settings**, in one reviewable place:

```python
# file: helpdesk/prompts.py
def system_prompt(settings: HelpdeskSettings) -> str:
```

**4. Tools translate between the model and the domain.** The rule itself (`can_cancel`) stays in the domain; the tool only adapts it:

```python
# file: helpdesk/tools.py
    @vdreamer.tool(name="can_cancel")
    def can_cancel_tool(order_id: str) -> str:
        """Check whether an order can still be cancelled."""
        order = _get(order_id)
        verdict = "can" if can_cancel(order) else "cannot"
        return f"Order {order.order_id} {verdict} be cancelled (status: {order.status})."
```

**5. The agent only wires.**

```python
# file: helpdesk/agent.py
    def __init__(self, orders: OrderRepository | None = None,
                 settings: HelpdeskSettings | None = None) -> None:
        self.orders = orders if orders is not None else demo_orders()
        self.settings = settings if settings is not None else HelpdeskSettings()

    def build(self, ctx: BuildContext) -> Any:
        return create_agent(
            ctx.models["chat"],
            tools=make_tools(self.orders, self.settings),
            system_prompt=system_prompt(self.settings),
            **ctx.agent_kwargs(),
        )
```

**6. Publish a deliberate surface** in `helpdesk/__init__.py` (`__all__`) and register the instance: `t11-helpdesk: vdreamer_tutorials.t11_package_structure.helpdesk:agent`.

## The complete example

[`helpdesk/`](helpdesk/__init__.py) — `domain.py`, `settings.py`, `prompts.py`, `tools.py`, `agent.py` and `__init__.py`. The capstone (lesson 31) uses the same layout at a larger scale.

## Run it

From `tutorials/`:

```bash
uv run vdreamer validate --registry vdreamer.yaml t11-helpdesk --models config/models.ollama.yaml --no-probe
uv run vdreamer dev --registry vdreamer.yaml --agent t11-helpdesk --models config/models.ollama.yaml
uv run vdreamer run t11-helpdesk --message "Can I still cancel order A1002?"
```

In your own project the layout is the same with your package name: `vdreamer dev --agent mypackage:agent`, or a `vdreamer.yaml` registry next to it.

## Expected behavior

`validate` passes with `tools.governed: 3 tool(s)`. The cancellation question produces:

```
-> can_cancel({"order_id": "A1002"}) [custom]
<- ok Order A1002 can be cancelled (status: processing). (1 ms)
```

## Test it

[`test_t11_package_structure.py`](../../../../tests/tutorials/test_t11_package_structure.py) tests each layer with nothing above it — the domain with no LangChain, the prompt with custom settings, the tools over a custom repository — then the assembled agent and the gateway. And it enforces the architecture:

```python
# file: tests/tutorials/test_t11_package_structure.py
#: module → the sibling modules and outside libraries it may import.
ALLOWED = {
    "domain": {"pydantic"},
    "settings": set(),
    "prompts": {"settings"},
    "tools": {"domain", "settings", "langchain_core", "vdreamer"},
    "agent": {"domain", "settings", "prompts", "tools", "langchain", "vdreamer"},
}
```

If someone imports `vdreamer` into `domain.py` "just for a helper", this test fails and says which module broke the rule.

```bash
uv run pytest tests/tutorials/test_t11_package_structure.py -v
```

## Common mistakes and debugging

- **Business rules inside tool functions** — they can then only be tested through a tool call, and a second agent has to copy them. Keep the rule in the domain; the tool adapts it.
- **Circular imports** (`tools.py` imports `agent.py` to read a setting) — a sign the dependency points upward. Pass the value down instead.
- **Reading `os.environ` in many modules** — configuration becomes invisible and untestable. Read it once (lesson 12), into the settings object.
- **Per-request data on the agent instance** (`self.current_user = …`) — one instance and one compiled graph serve every run and every user, concurrently. Per-run data comes from the run context and state.
- **A huge `__init__.py`** that imports everything — slow imports and accidental public API. Export the few names users need.
- **`agent spec … could not be imported`** — the registry points at the package; check it exposes `agent` in `__init__.py`.

## Design trade-offs

- **Five files for a small agent** is more structure than it needs; the payoff comes with the second agent, the second developer, or the first prompt review. Start in one file and split along these seams when it grows.
- **Protocols vs. abstract base classes.** Protocols need no inheritance (any matching class works, including test fakes); ABCs check at instantiation. Protocols suit dependency injection well.
- **Settings objects vs. globals.** Explicit objects make every dependency visible in a constructor signature, at the cost of passing them around.

## Exercises

1. Add a `cancel_order` side-effecting tool whose rule (`can_cancel`) stays in the domain; give `InMemoryOrders` a `cancel` method and test it without any agent.
2. Write a second agent, `t11-helpdesk-readonly`, that reuses `make_tools` but drops `can_cancel`. What did you need to change in the other modules? (Ideally nothing.)
3. Extend the import-rule test so that `tools.py` may not import `langchain` (only `langchain_core`).

Next: [Lesson 12 — Configuration, logging and observability](../t12_config_and_observability/README.md).
