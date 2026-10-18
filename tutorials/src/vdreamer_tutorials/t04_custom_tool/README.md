# Lesson 4 — Defining and integrating a custom tool

> **Outcome:** a shipping-quote tool whose name, description and typed arguments form a precise contract for the model, inspected as the JSON schema the model actually receives, integrated into an agent and tested on its own.

## Learning objectives

- Turn a Python function into a tool with `@vdreamer.tool`, and know which parts become the tool's **contract**: name, description, argument schema.
- Use type hints (`float`, `bool`, `Literal[...]`, defaults) to constrain what the model can send.
- Put per-argument descriptions into the schema with `parse_docstring=True`.
- Inspect the exact schema a model receives (`convert_to_openai_tool`) and test the tool without any model.
- Return structured data (a `dict`) and know how it reaches the model.

## Prerequisites

- [Lesson 2](../t02_minimal_react_agent/README.md) (`create_agent`, `@vdreamer.tool`) and [Lesson 3](../t03_react_loop/README.md) (tool calls and `ToolMessage`s).
- Python type hints, including `Literal`.

## Concepts

When an agent starts, each tool is converted into a **function schema** — a name, a description and a JSON Schema of its parameters — and sent to the model with every request. The model chooses tools by reading those schemas, and generates arguments that (usually) follow them. So a tool definition is an interface for a reader that cannot see your code:

- **The name** says what it does: `estimate_shipping`, not `calc2`.
- **The description** says *when* to use it and what it returns.
- **The parameters** say exactly what to send. Types narrow the space: `Literal["domestic", "eu", "international"]` becomes an `enum` the model picks from, instead of free text you would have to parse. A default (`express: bool = False`) makes an argument optional.
- **Per-argument descriptions** carry units and examples ("kilograms, for example 2.5"). By default LangChain leaves the docstring's `Args:` section inside the description text; `parse_docstring=True` moves each entry into its parameter's schema, where models use it best.

A tool's **return value** goes back to the model as the `ToolMessage` content: strings as they are, other values serialized (a `dict` becomes JSON). Structured results are easier for the model to quote accurately than prose.

`@vdreamer.tool` is LangChain's `@tool` with one addition: a governance **policy** stored in `tool.metadata["vdreamer"]` (timeout, redaction, result size, approval…). The tool remains a plain LangChain `BaseTool` — usable in any LangChain code, testable by calling `.invoke()`.

## What comes from where

| Piece | Provided by |
| --- | --- |
| Function → `StructuredTool`, `parse_docstring`, schema generation | LangChain (`langchain.tools.tool`) |
| `convert_to_openai_tool` (the schema sent to models) | LangChain (`langchain_core`) |
| `@vdreamer.tool` and the policy in `metadata["vdreamer"]` | vdreamer |
| `Literal`, defaults, docstrings | Python |

## How it works

```
@vdreamer.tool(parse_docstring=True)          what the model receives with every request
def estimate_shipping(weight_kg: float,  ──►  {"name": "estimate_shipping",
        destination: Destination,               "description": "Estimate the price and …",
        express: bool = False) -> dict:          "parameters": {"properties": {
    """Estimate the price … parcel.                 "weight_kg": {"type": "number", "description": …},
    Args:                                           "destination": {"enum": ["domestic","eu","international"], …},
        weight_kg: …                                "express": {"type": "boolean", "default": false, …}},
    """                                           "required": ["weight_kg", "destination"]}}

model reply: tool_calls=[{"name": "estimate_shipping", "args": {"weight_kg": 2.5, "destination": "eu", "express": true}}]
    ──► the function runs ──► ToolMessage('{"destination": "eu", …, "cost_eur": 27.45, "delivery_days": 2}')
```

## Step by step

**1. Model the choices as types.** A `Literal` alias documents the allowed destinations once and becomes an `enum` in the schema:

```python
# file: agent.py
Destination = Literal["domestic", "eu", "international"]
```

**2. Write the function with a docstring for the model.** The first line becomes the description; each `Args:` entry becomes its parameter's description.

```python
# file: agent.py
@vdreamer.tool(parse_docstring=True)
def estimate_shipping(
    weight_kg: float, destination: Destination, express: bool = False
) -> dict[str, Any]:
    """Estimate the price and delivery time of shipping one parcel.

    Args:
        weight_kg: Parcel weight in kilograms, for example 2.5.
        destination: "domestic" (same country), "eu" (another EU country) or "international".
        express: Use the express service: faster, but more expensive.
    """
    base, per_kg, days = RATES[destination]
    cost = base + per_kg * weight_kg
    if express:
        cost *= EXPRESS_SURCHARGE
        days = max(1, days // 2)
    return {
        "destination": destination,
        "weight_kg": weight_kg,
        "express": express,
        "cost_eur": round(cost, 2),
        "delivery_days": days,
    }
```

The result echoes its inputs (`destination`, `weight_kg`, `express`) so the model — and anyone reading the trace — can see what was actually priced.

**3. Inspect the contract.** This is what the model receives:

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
convert_to_openai_tool(estimate_shipping)
```

```json
{"type": "function", "function": {
  "name": "estimate_shipping",
  "description": "Estimate the price and delivery time of shipping one parcel.",
  "parameters": {"type": "object", "required": ["weight_kg", "destination"], "properties": {
    "weight_kg":   {"type": "number", "description": "Parcel weight in kilograms, for example 2.5."},
    "destination": {"type": "string", "enum": ["domestic", "eu", "international"],
                    "description": "\"domestic\" (same country), \"eu\" (another EU country) or \"international\"."},
    "express":     {"type": "boolean", "default": false,
                    "description": "Use the express service: faster, but more expensive."}}}}}
```

Without `parse_docstring=True`, the three argument descriptions would stay inside `description` as raw text and the parameters would carry types only.

**4. Integrate it.** Pass the tool to the agent and say in the system prompt when to use it — including the mapping the model must make from what users say ("Spain") to what the tool accepts (`"eu"`):

```python
# file: agent.py
SYSTEM_PROMPT = (
    "You quote shipping prices for an online shop. Always use the estimate_shipping tool; "
    "never guess a price. Countries in the EU other than the shop's own are 'eu'. "
    "State the price in euros and the delivery time in days."
)
```

## The complete example

[`agent.py`](agent.py) — the rate table, `estimate_shipping`, and `ShippingQuoteAgent` (`t04-shipping-quote`), a `create_agent` agent exactly as in lesson 2.

## Run it

From `tutorials/`:

```bash
uv run python -c "
from vdreamer_tutorials.t04_custom_tool.agent import estimate_shipping
print(estimate_shipping.invoke({'weight_kg': 2.5, 'destination': 'eu', 'express': True}))"

uv run vdreamer dev --registry vdreamer.yaml --agent t04-shipping-quote --models config/models.ollama.yaml
uv run vdreamer run t04-shipping-quote --message "How much to send 2.5 kg to Spain, fast?"
```

## Expected behavior

The direct call prints `{'destination': 'eu', 'weight_kg': 2.5, 'express': True, 'cost_eur': 27.45, 'delivery_days': 2}` — no model involved. Through the agent (model wording varies):

```
-> estimate_shipping({"weight_kg": 2.5, "destination": "eu", "express": true}) [custom]
<- ok {"destination": "eu", "weight_kg": 2.5, "express": true, "cost_eur": 27.45, "delivery_days": 2} (1 ms)
assistant>
Express shipping of 2.5 kg to Spain costs EUR 27.45 and takes about 2 days.
```

## Test it

A tool is a contract plus behavior, and both are testable without a model. [`test_t04_custom_tool.py`](../../../../tests/tutorials/test_t04_custom_tool.py) pins the schema the model sees, prices a table of cases, checks the `ToolMessage` form, and runs the agent with a scripted model:

```python
# file: tests/tutorials/test_t04_custom_tool.py
def test_what_the_model_sees() -> None:
    spec = convert_to_openai_tool(estimate_shipping)["function"]

    assert spec["name"] == "estimate_shipping"
    assert spec["description"] == "Estimate the price and delivery time of shipping one parcel."
    params = spec["parameters"]
    assert params["required"] == ["weight_kg", "destination"]
```

Pinning the schema in a test turns an accidental contract change (a renamed argument, a lost description) into a failing test instead of a quietly worse agent.

```bash
uv run pytest tests/tutorials/test_t04_custom_tool.py -v
```

## Common mistakes and debugging

- **A docstring written for humans only** ("Helper for the quote flow") — the model has no other information. Say what the tool does, when to use it, and what the arguments mean, with units.
- **`str` where a `Literal` belongs** — free-text arguments invite "Spain", "ES", "europe" and parsing code. Constrain the input and let the system prompt (or the description) explain the mapping.
- **No docstring at all** — LangChain refuses to build the tool (`ValueError: Function must have a docstring if description not provided`). Pass `description=` or write one.
- **Forgetting `parse_docstring=True`** — still works, but the argument descriptions stay in the description text, which small models follow less reliably. With it, a malformed docstring (for example, no blank line before `Args:`) raises `ValueError: Found invalid Google-Style docstring.` when the module is imported. An argument you forget to list is not an error: it simply gets no description.
- **Returning objects that do not serialize** (a database row, a `datetime` inside a custom class) — the model receives an unhelpful `repr`. Return `str`, numbers, lists and dicts of those.
- **Testing only through the agent** — slow and indirect. Call `tool.invoke({...})` directly for the logic; keep agent tests for the wiring.

## Design trade-offs

- **Rich schema vs. prompt length.** Every tool's schema is sent with every model call. Good descriptions cost tokens; with many tools that adds up (lesson 9 shows how to offer only the relevant ones).
- **Strict types vs. forgiving tools.** An `enum` rejects anything unexpected (lesson 5 shows how vdreamer reports that to the model). A forgiving `str` accepts more but pushes ambiguity into your code. Prefer strict inputs with clear descriptions.
- **Dict vs. sentence results.** JSON is precise and easy to test; a sentence can be friendlier for small models. Whatever you choose, keep it consistent across tools (lesson 8).
- **Logic inside the tool.** Fine for a small function. When the logic grows or needs dependencies (a database, an HTTP client), keep it in a plain class and make the tool a thin adapter (lesson 8).

## Exercises

1. Add an `insured: bool = False` argument that adds 2% of a `declared_value_eur` (another argument). Update the schema test first.
2. Remove `parse_docstring=True`, print `convert_to_openai_tool(estimate_shipping)` again, and compare.
3. Rename the tool with `@vdreamer.tool("quote_shipping", parse_docstring=True)` and see which test fails first.
4. Return a sentence instead of a dict and ask yourself which result a model would quote more reliably.

Next: [Lesson 5 — Tool inputs, outputs, errors and validation](../t05_tool_errors/README.md).
