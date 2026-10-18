# Lesson 8 — Designing reusable tools with clear contracts

> **Outcome:** an inventory agent whose tools are built by a factory from an injected `Inventory` dependency, return typed results, raise actionable errors, and make a stock reservation that is safe to retry — the same tools reusable by any agent and testable with any implementation.

## Learning objectives

- Separate a tool into three layers: the **domain** (plain Python), the **contract** (schema + result model + errors), and the **adapter** (the `@vdreamer.tool` function).
- Inject dependencies into tools with a factory and a `typing.Protocol` instead of module globals.
- Declare side effects (`side_effects=True`) and make them idempotent with vdreamer's per-call idempotency key.
- Set deployment-specific limits with the agent's `tool_policies` instead of editing reusable tools.
- Write a contract checklist you can apply to every tool.

## Prerequisites

- Lessons [4](../t04_custom_tool/README.md) (tool schemas), [5](../t05_tool_errors/README.md) (`ToolException`, policies) and [7](../t07_multiple_tools/README.md) (tool sets).
- `typing.Protocol` and pydantic models.

## Concepts

The tools in lessons 2–7 read module-level dictionaries. That works for a demo and fails for reuse: two agents cannot use different stores, tests must monkeypatch globals, and the tool's logic is tangled with its data source. A reusable tool is a **contract** in front of a **dependency**:

```
Inventory (Protocol)          ← what the tools need: level(sku), reserve(sku, units, key)
   ▲ implemented by
InMemoryInventory / WarehouseApi / a test fake
   ▲ injected into
make_inventory_tools(inventory) → [check_stock, reserve_stock]   ← the contract the model sees
   ▲ used by
InventoryAgent(inventory) — or any other agent
```

**A tool contract** has five parts, and each one should be explicit:

1. **Name and purpose** — one job, `verb_noun`, read-only or not.
2. **Inputs** — a pydantic schema with constraints and descriptions (validated before the tool runs).
3. **Output** — a typed model (`StockLevel`, `Reservation`) serialized to JSON; field names carry units (`available_units`).
4. **Errors** — `ToolException` with a message that says what to do next; domain exceptions translated, never leaked.
5. **Effects** — whether it changes anything (`side_effects=True`), and what happens if it runs twice.

**Idempotency.** vdreamer checkpoints every step and retries a run after a worker crash. If the crash happened after a tool changed something but before the step was saved, the same tool call runs again. For a reservation that would double-book stock. vdreamer helps: for a tool with `side_effects=True`, governance sets `current_context().idempotency_key` to `"{run}:{tool_call_id}"`. The model's tool-call id is checkpointed with the step, so **a retry of the same call gets the same key**; the domain stores results by key and returns the first one on a repeat. (For an external API, pass the key on to it — many payment and booking APIs accept an `Idempotency-Key` header for exactly this.)

**Agent policies over tool policies.** A reusable tool declares sensible defaults (`timeout_s=10`). The agent that deploys it may need different limits for its environment. `BaseAgent.tool_policies` maps tool names or globs (`"*_stock"`) to a `Policy`; its non-default fields override the decorator's, for every matching tool, without touching the tool's code. The same works for third-party tools you cannot edit at all.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `Protocol`, dependency injection through a factory | Python / this lesson's design |
| `args_schema`, `ToolException` | LangChain |
| `side_effects=True`, `current_context().idempotency_key`, `tool_policies`, `Policy` | vdreamer |
| Same idempotency key on a retried call (checkpointed tool-call ids) | vdreamer + LangGraph checkpoints |
| `BuildContext().tools.govern(...)`, `bind_context(...)` (used in tests) | vdreamer public API |

## How it works

A reservation, and the retry that does not double it:

```
model: reserve_stock(sku="TENT-4P", units=2)            tool_call_id = call_7
governance: side_effects=True → idempotency_key = "run_1:call_7"
inventory.reserve("TENT-4P", 2, key="run_1:call_7") → R-0001 (stored under the key)
   ✗ worker crashes before the step is checkpointed
worker retries the run → the same checkpointed tool call → the same key
inventory.reserve(..., key="run_1:call_7") → R-0001 again; stock unchanged
```

## Step by step

**1. The domain: types and an interface, no agent code.** `StockLevel` and `Reservation` are the result contract; `Inventory` lists exactly what the tools need.

```python
# file: inventory.py
class StockLevel(BaseModel):
    """What ``check_stock`` returns: a stable, typed result."""

    sku: str
    available_units: int
    reserved_units: int
...
class Inventory(Protocol):
    """The operations the inventory tools need, and nothing more."""

    def level(self, sku: str) -> StockLevel | None: ...

    def reserve(self, sku: str, units: int, idempotency_key: str) -> Reservation: ...
```

**2. Idempotency lives in the domain.** The implementation remembers each key's result:

```python
# file: inventory.py
    def reserve(self, sku: str, units: int, idempotency_key: str) -> Reservation:
        if idempotency_key in self._by_key:  # a retry of a call that already happened
            return self._by_key[idempotency_key]
```

**3. Inputs validated once, shared by both tools.**

```python
# file: tools.py
SKU = Field(pattern=r"^[A-Z0-9]+-[A-Z0-9]+$", description='A product SKU, e.g. "TENT-4P".')


class CheckStockInput(BaseModel):
    sku: str = SKU


class ReserveStockInput(BaseModel):
    sku: str = SKU
    units: int = Field(gt=0, le=10, description="How many units to reserve (1 to 10).")
```

**4. The factory: thin adapters over the injected dependency.** The tools close over `inventory`; nothing is global.

```python
# file: tools.py
def make_inventory_tools(inventory: Inventory) -> list[BaseTool]:
    """``check_stock`` (read-only) and ``reserve_stock`` (side effect) over ``inventory``."""
```

The writing tool declares its effect, takes the key from the run context, and translates domain errors into actionable messages:

```python
# file: tools.py
    @vdreamer.tool(args_schema=ReserveStockInput, side_effects=True, timeout_s=10)
    def reserve_stock(sku: str, units: int) -> dict[str, Any]:
        """Reserve units of a product for the customer. Changes stock: call it only when
        the customer explicitly asks to reserve, and only once per request."""
        key = vdreamer.current_context().idempotency_key
        assert key is not None  # side_effects=True: governance always sets one
        try:
            reservation = inventory.reserve(sku, units, idempotency_key=key)
        except KeyError:
            raise ToolException(f"Unknown SKU {sku}. Check the SKU with the customer.") from None
        except InsufficientStock as exc:
            raise ToolException(f"Cannot reserve: {exc}. Offer fewer units.") from None
        return reservation.model_dump()
```

**5. The agent receives the dependency and sets its own limits.**

```python
# file: agent.py
class InventoryAgent(BaseAgent):
    name = "t08-inventory"
    mode = Mode.CHAT
    models = ("chat",)
    # Deployment-specific limits belong to the agent, not to the reusable tool:
    # they override the decorators' policies for every matching tool name.
    tool_policies = {"*_stock": Policy(timeout_s=5)}

    def __init__(self, inventory: Inventory | None = None) -> None:
        self.inventory = inventory if inventory is not None else demo_inventory()
```

`build()` passes `make_inventory_tools(self.inventory)` to `create_agent`. In production, the registered instance would be built with a real warehouse client.

## The complete example

- [`inventory.py`](inventory.py) — the domain: result models, `Inventory`, `InMemoryInventory`.
- [`tools.py`](tools.py) — input schemas and `make_inventory_tools`.
- [`agent.py`](agent.py) — `InventoryAgent` (`t08-inventory`).

## Run it

From `tutorials/`:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t08-inventory --models config/models.ollama.yaml
uv run vdreamer run t08-inventory --message "Please reserve 2 units of TENT-4P"
```

Then ask for 5 more and see the error the model relays. The demo inventory lives in the worker process: restarting `vdreamer dev` resets it.

## Expected behavior

```
-> reserve_stock({"sku": "TENT-4P", "units": 2}) [custom]
<- ok {"reservation_id": "R-0001", "sku": "TENT-4P", "units": 2} (1 ms)
```

Asking for more than is left gives `<- error Error (tool_error): Cannot reserve: only 1 units of TENT-4P available. Offer fewer units.`

## Test it

[`test_t08_tool_contracts.py`](../../../../tests/tutorials/test_t08_tool_contracts.py) tests the domain without any agent code, the contract of the tools over a fresh inventory, the agent, and the gateway. The retry test calls the same governed tool call twice inside one bound run context — what a crash-and-retry does — and checks that stock moved once:

```python
# file: tests/tutorials/test_t08_tool_contracts.py
    inventory = InMemoryInventory({"TENT-4P": 3})
    (_, reserve) = BuildContext().tools.govern(make_inventory_tools(inventory))
    tool_call = {"type": "tool_call", "id": "call_7", "name": "reserve_stock",
                 "args": {"sku": "TENT-4P", "units": 2}}
    context = VDreamerContext(run_id="run_1", thread_id="t", agent_name="t08-inventory")

    with bind_context(context):
        first = reserve.invoke(tool_call)
        retry = reserve.invoke(tool_call)

    assert first.content == retry.content  # the same reservation, returned twice
```

```bash
uv run pytest tests/tutorials/test_t08_tool_contracts.py -v
```

## Common mistakes and debugging

- **`RuntimeError: no active vdreamer run context`** when calling `reserve_stock.invoke(...)` in a script — the tool reads the run context. Bind one (`with vdreamer.bind_context(VDreamerContext(...)):`) or test the domain directly.
- **`idempotency_key` is `None`** — the tool is not declared `side_effects=True`, so governance does not set a per-call key. (Outside a side-effect tool, `idempotency_key` is the run's own key, if the run was started with one.)
- **A new key per retry** (`uuid4()` inside the tool) — defeats the purpose. The key must come from something that survives the crash: the checkpointed tool-call id that vdreamer uses.
- **Leaking domain exceptions** (`KeyError: 'TENT-9'`) — the model gets a cryptic message. Catch domain errors in the adapter and raise `ToolException` with the next step.
- **Mutable state shared between agent instances** (a module-level inventory) — tests interfere and two agents see each other's data. Inject it.
- **`tool_policies` that seem to have no effect** — patterns are matched case-sensitively against tool names (`fnmatch`), and fields left at their default never override. Check the effective policy as the test does: `BuildContext(tool_policies=...).tools.govern(tools)[i].policy`.

## Design trade-offs

- **Factories vs. classes with methods.** A factory with closures keeps each tool a plain function with a docstring the model reads. A class (`InventoryTools(inventory).as_tools()`) groups related tools more visibly. Both work; pick one per codebase.
- **Pydantic results vs. dicts.** Models document and validate the shape at the cost of a little code; `model_dump()` keeps the wire format plain JSON.
- **Idempotency in the domain vs. in the tool.** Storing results per key in the domain (or the external system) survives process restarts if the store is durable; a dictionary in the tool would not. The in-memory demo shows the shape, not the durability.
- **Read/write split.** Separate `check_stock` and `reserve_stock` tools let you gate only the writing one (approvals, lesson 27; roles, lesson 28), and keep read tools safe to run in parallel.

## Exercises

1. Write a `FakeInventory` in a test that records calls, and use it to assert `reserve_stock` passes the key through unchanged.
2. Add `release_reservation(reservation_id)`, side-effecting and idempotent. What should a second release of the same id return?
3. Register a second agent, `t08-inventory-strict`, with `tool_policies = {"reserve_stock": Policy(requires_roles=("staff",))}` and see what a run without that role gets (preview of lesson 28).
4. Make `InMemoryInventory` thread-safe (two concurrent reservations must not oversell) and test it with a thread pool.

Next: Lesson 9 — Tool selection and execution control.
