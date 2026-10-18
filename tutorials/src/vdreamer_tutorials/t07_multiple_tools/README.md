# Lesson 7 — An agent with multiple tools

> **Outcome:** a shop assistant with four tools — catalog search, stock, order status, store policy — that picks the right one, chains two when one's output is the next one's input, and batches independent lookups into one step.

## Learning objectives

- Give one agent several tools and make the choice between them unambiguous for the model.
- Recognise the three multi-tool patterns: **selection** (one of several), **chaining** (the output of one feeds the next), **fan-out** (independent calls in the same step).
- Guide tool use through descriptions and the system prompt, and keep tool results small and structured.
- Test the tool *set* as a whole, not just each tool.

## Prerequisites

- Lessons [3](../t03_react_loop/README.md) (the loop, parallel calls), [4](../t04_custom_tool/README.md) (tool contracts) and [6](../t06_testing_and_traces/README.md) (testing with scripted models).

## Concepts

With one tool, the model only decides *whether* to call it. With several, it also decides *which*, in *what order*, and *with what arguments* — based only on the tools' names, descriptions and schemas, plus the system prompt. Most multi-tool failures are therefore design failures you can fix in text:

- **Overlap.** Two tools that could both answer ("`find_item`" and "`search_products`") make the choice a coin toss. Give each tool one clear job.
- **Missing links.** `check_stock` needs a SKU, but customers say "the 2-person tent". If nothing tells the model where SKUs come from, it guesses one. Say it in the description ("by its SKU (from search_products)") and in the prompt ("if you only have a product name, search first").
- **Prompt-only rules.** Rules like "never quote policy from memory" belong in the tool description too, because that is what the model reads when it decides.

Three patterns appear in every multi-tool agent:

| Pattern | Example | In the trace |
| --- | --- | --- |
| Selection | "What is your returns policy?" → `get_policy("returns")` | one tool call |
| Chaining | "Is the 2-person tent in stock?" → `search_products` → `check_stock("TENT-2P")` | two steps; the second uses the first one's result |
| Fan-out | "Do you have TENT-4P and BAG-0C, and is shipping free?" → three calls at once | one step, several `tool.call`s |

Fan-out saves round trips: one model call instead of three. Most tool-calling models do it when calls are independent; `create_agent` runs the batch's calls concurrently.

## What comes from where

| Piece | Provided by |
| --- | --- |
| Several tools in `create_agent(tools=[...])`, concurrent execution of one step's calls | LangChain |
| Choosing tools, chaining, fan-out | the model (guided by your descriptions and prompt) |
| One `tool.call`/`tool.result` pair per call, governance on each | vdreamer |
| `Literal` → `enum` (`get_policy(topic)`), argument descriptions | Python typing, LangChain schema generation |

## How it works

"Is the 2-person tent in stock?":

```
reason ─► search_products(query="2-person tent")
            └─► [{"sku": "TENT-2P", "name": "Trail tent, 2-person", "price_eur": 149.0}]
reason ─► check_stock(sku="TENT-2P")          ← the SKU comes from the observation above
            └─► "TENT-2P: out of stock, expected back on 2026-10-20."
reason ─► "The Trail tent, 2-person (EUR 149) is out of stock right now; …"
```

"Do you have TENT-4P and BAG-0C, and is shipping free?":

```
reason ─► check_stock("TENT-4P") ┐
          check_stock("BAG-0C")  ├─ one step, run concurrently
          get_policy("shipping") ┘
reason ─► "Both are in stock, and the order ships free."
```

## Step by step

**1. One job per tool, with the link to other tools in its description.**

```python
# file: agent.py
@vdreamer.tool(parse_docstring=True)
def check_stock(sku: str) -> str:
    """Check how many units of a product are in stock, by its SKU (from search_products).

    Args:
        sku: The product SKU, for example "TENT-4P".
    """
```

**2. Make the "wrong input" answer helpful.** If the model passes a product name as a SKU anyway, the tool says what to do instead of just "not found":

```python
# file: agent.py
    key = sku.strip().upper()
    if key not in STOCK:
        return f"Unknown SKU {key!r}. Use search_products to find the right SKU."
```

**3. Return compact, structured results.** Search returns only what the next step needs (SKU, name, price) — not the whole catalog record — and caps the number of matches:

```python
# file: agent.py
@vdreamer.tool(parse_docstring=True)
def search_products(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Search the product catalog by keywords. Returns SKU, name and price per match.

    Use this first when the customer names a product rather than a SKU.
```

**4. Close off free text where you can.** A `Literal` turns the policy topic into an enum, so "privacy" is rejected by the schema (`invalid_args`) instead of raising `KeyError` in your code:

```python
# file: agent.py
def get_policy(topic: Literal["returns", "shipping", "warranty"]) -> str:
    """Quote the shop's official policy on one topic. Never paraphrase policy from memory.
```

**5. Describe the toolbox in the system prompt**, including the order of operations and the invitation to batch:

```python
# file: agent.py
SYSTEM_PROMPT = """You are the shop assistant of an outdoor-gear store.
Tools:
- search_products: find products and their SKUs from a description.
- check_stock: stock for one SKU. If you only have a product name, search first.
- get_order_status: the status of an existing order.
- get_policy: returns, shipping and warranty rules. Quote it; do not rely on memory.
Use as few tool calls as needed. Independent lookups can be requested together.
Never invent products, prices, stock levels or policies."""
```

## The complete example

[`agent.py`](agent.py) — the sample catalog, stock and policies; the four tools; `TOOLS`; and `ShopAssistantAgent` (`t07-shop-assistant`).

## Run it

From `tutorials/`:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t07-shop-assistant --models config/models.ollama.yaml
uv run vdreamer run t07-shop-assistant
```

Try: `Is the 2-person tent in stock?` (chaining), `Do you have TENT-4P and BAG-0C, and is shipping free?` (fan-out), `What is your warranty on boots?` (selection), `Where is order A1002?`.

## Expected behavior

Chaining (tool lines exact; wording varies):

```
-> search_products({"query": "2-person tent"}) [custom]
<- ok [{"sku": "TENT-2P", "name": "Trail tent, 2-person", "price_eur": 149.0}] (1 ms)
-> check_stock({"sku": "TENT-2P"}) [custom]
<- ok TENT-2P: out of stock, expected back on 2026-10-20. (1 ms)
assistant>
The Trail tent, 2-person (EUR 149) is out of stock right now; it is expected back on 20 October.
```

For the fan-out question, a capable model issues the three calls in one step (three `->` lines before any answer); a weaker one may do them one by one — slower, but still correct.

## Test it

[`test_t07_multiple_tools.py`](../../../../tests/tutorials/test_t07_multiple_tools.py) tests each tool alone, then the **set**: unique names, distinct descriptions, and a description on every argument — the properties that keep tool choice unambiguous as the toolbox grows:

```python
# file: tests/tutorials/test_t07_multiple_tools.py
def test_the_tool_set_is_unambiguous() -> None:
    specs = [convert_to_openai_tool(tool)["function"] for tool in TOOLS]
    names = [spec["name"] for spec in specs]
    assert len(set(names)) == len(names)
    assert len({spec["description"] for spec in specs}) == len(specs)
```

The chaining test checks that the SKU used in step 2 is the one the model *observed* in step 1:

```python
# file: tests/tutorials/test_t07_multiple_tools.py
    found = json.loads([m for m in model.prompts[1] if isinstance(m, ToolMessage)][-1].text)
    assert found[0]["sku"] == result.tool_calls[1][1]["sku"] == "TENT-2P"
```

```bash
uv run pytest tests/tutorials/test_t07_multiple_tools.py -v
```

## Common mistakes and debugging

- **The model guesses a SKU** (`check_stock("TENT-2")`) — the link between tools is not described. Add "(from search_products)" to the argument or description, and make the not-found answer point to the right tool.
- **The model calls the same tool repeatedly with slightly different arguments** — the result was not useful (too big, too vague, or an error it cannot act on). Look at the `tool.result` summaries in the trace.
- **Policy or prices answered without a tool call** — the model "knew" an answer. Say "never invent / never from memory" in both the prompt and the tool's description, and test with a live model (lesson 6) because only a real model shows this behavior.
- **Fan-out expected but calls come one by one** — model-dependent. Do not rely on it for correctness, only for speed.
- **Two tools mutate the same data in one fan-out step** — they run concurrently. Keep tools that change data out of parallel batches, or make them safe to run concurrently (lesson 8).

## Design trade-offs

- **Few general tools vs. many specific ones.** `run_sql(query)` is flexible but hard to use safely and correctly; `check_stock(sku)` is narrow but reliable. Prefer specific tools for agents that face customers; keep general ones for expert users, behind approvals (lesson 27).
- **Every tool costs prompt space** on every model call, and more tools mean more chances to pick the wrong one. Beyond about 10–20 tools, offer only the relevant subset per request (lesson 9) or split responsibilities across agents (lessons 21–22).
- **Structured vs. text results.** Lists of dicts are precise and easy to chain (the SKU is a field); short sentences are friendlier when the result is the final answer. This agent uses both on purpose.

## Exercises

1. Add `get_product_details(sku)` returning weight and materials. Make sure its description does not overlap with `search_products`, and extend the tool-set test if needed.
2. Make `search_products` match "2 person tent" (without the hyphen) too, with a test.
3. Write a scripted test for "Where is A1001 and what is your returns policy?" expecting a two-call fan-out.
4. With a live model, ask "Do you have tents?" and look at whether it searches or asks a clarifying question. Adjust the prompt and compare.

Next: [Lesson 8 — Designing reusable tools with clear contracts](../t08_tool_contracts/README.md).
