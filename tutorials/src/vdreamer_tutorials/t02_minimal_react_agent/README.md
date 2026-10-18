# Lesson 2 — A minimal ReAct agent

> **Outcome:** an order-support agent that uses a language model and one tool, `get_order_status`, served by vdreamer against a local Ollama (or any vLLM/OpenAI-compatible server), and tested without one.

## Learning objectives

- Build a ReAct agent with LangChain's `create_agent`: a model, a list of tools, a system prompt.
- Connect a model through a **model alias**: the agent asks for `"chat"`; `models.yaml` decides which server and model that is.
- Wire a prebuilt agent into vdreamer with `**ctx.agent_kwargs()`, and know what it supplies.
- Watch the reason–act–observe cycle in the event stream (`tool.call`, `tool.result`, the answer).

## Prerequisites

- [Lesson 1](../t01_serving_an_agent/README.md): `BaseAgent`, `build(ctx)`, `vdreamer dev` / `vdreamer run`.
- For a live run: [Ollama](https://ollama.com) running locally with a tool-calling model, or an OpenAI-compatible server (vLLM, llama.cpp `llama-server`, LM Studio, …). The tests need neither.

## Concepts

A language model on its own can only produce text. A **ReAct agent** ("reason + act") gives it tools: the model *reasons* about the request, *acts* by asking for a tool call (a function name plus JSON arguments), *observes* the tool's result, and repeats until it can answer. The model never runs code itself; your program executes the tool and feeds the result back.

LangChain's `create_agent` builds that loop as a LangGraph graph: a **model** node calls the LLM with the conversation and the tools' schemas; if the reply contains tool calls, a **tools** node runs them and appends their results as `ToolMessage`s; then control returns to the model. When the model replies without tool calls, the graph ends. Lesson 3 builds the same loop by hand.

The system prompt is where you set the agent's job and its rules ("never invent an order status"). The tool's **name, docstring and parameter types** are what the model sees when deciding whether and how to call it — write them for the model, not just for humans (lesson 4).

vdreamer never talks to cloud model providers. Agents declare **aliases** (`models = ("chat",)`), and `models.yaml` maps each alias to a self-hosted endpoint. The same agent code runs against a laptop Ollama in development and a vLLM cluster in production.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `create_agent` (the ReAct loop), `system_prompt` | LangChain (`langchain.agents`) |
| `@vdreamer.tool` — LangChain's `@tool` plus a governance policy | vdreamer (the tool is a plain LangChain `BaseTool`) |
| `models = ("chat",)`, `ctx.models["chat"]`, `models.yaml` | vdreamer (builds a `ChatOllama` / `ChatOpenAI` model) |
| `ctx.agent_kwargs()` → checkpointer, store, run context, tool-governance middleware | vdreamer |
| `tool.call` / `tool.result` events, secret redaction, error containment | vdreamer governance |

## How it works

```
user message ─► model node ──(tool calls?)── no ──► answer (AIMessage) ─► END
                   ▲               │ yes
                   │               ▼
                   └──── tools node: get_order_status(order_id="A1001")
                         → ToolMessage("Order A1001: shipped on …")
```

Each step reaches the client as an event: the model's tool-call message, `tool.call` (the name and arguments), `tool.result` (status and a summary), and finally the answer as `token.delta`s and `message.completed`.

`ctx.agent_kwargs()` returns exactly four keyword arguments for `create_agent`:

- `checkpointer` and `store` — vdreamer's durable thread state and long-term store (lessons 13–15);
- `context_schema=VDreamerContext` — the run's identity (run id, thread, user, roles) for tools and middleware;
- `middleware=[VDreamerToolMiddleware(...)]` — runs every tool call through vdreamer's governance pipeline (lesson 5): events, redaction, timeouts, error containment.

## Step by step

**1. Write the tool.** `@vdreamer.tool` is a drop-in for LangChain's `@tool`: the function name becomes the tool name, the docstring its description, the type hints its JSON schema.

```python
# file: agent.py
@vdreamer.tool
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an order by its id, for example "A1001"."""
    key = order_id.strip().upper()
    status = ORDERS.get(key)
    if status is None:
        return f"No order with id {order_id!r} was found."
    return f"Order {key}: {status}"
```

Returning a clear "not found" message (instead of raising) lets the model tell the customer what happened. Lesson 5 covers real errors.

**2. Give the agent its job.**

```python
# file: agent.py
SYSTEM_PROMPT = (
    "You are the order-support assistant of an online shop. "
    "Use the get_order_status tool to answer questions about orders. "
    "If the customer gives no order id, ask for it. Never invent an order status."
)
```

**3. Build the agent around a model alias.**

```python
# file: agent.py
class OrderStatusAgent(BaseAgent):
    name = "t02-order-status"
    mode = Mode.CHAT
    models = ("chat",)  # an alias from models.yaml, available as ctx.models["chat"]

    def build(self, ctx: BuildContext) -> Any:
        return create_agent(
            ctx.models["chat"],
            tools=[get_order_status],
            system_prompt=SYSTEM_PROMPT,
            **ctx.agent_kwargs(),  # checkpointer, store, run context, tool governance
        )
```

`create_agent` returns a compiled graph wired to `ctx.checkpointer`, so vdreamer serves it as is.

**4. Point the alias at a model.** [`tutorials/config/models.ollama.yaml`](../../../config/models.ollama.yaml):

```yaml
aliases:
  chat:
    provider: ollama
    base_url: http://127.0.0.1:11434
    model: gemma4:31b-cloud
    temperature: 0
```

`gemma4:31b-cloud` runs on ollama.com, so prompts leave your machine. To stay local, `ollama pull gemma4:31b` (or another tool-calling model such as `qwen3:8b`) and change `model`. For vLLM or another OpenAI-compatible server, use [`config/models.vllm.yaml`](../../../config/models.vllm.yaml) (`provider: vllm`, the server's `/v1` URL; a key, if any, is named by `api_key_env` and read from `.env` — never written in the YAML).

## The complete example

[`agent.py`](agent.py): the tool, the prompt and the agent above, plus a dictionary of three sample orders (`A1001`–`A1003`) standing in for the shop's order system.

## Run it

From `tutorials/` (with `.env` from lesson 1). Check the agent and the config without contacting the model:

```bash
uv run vdreamer validate --registry vdreamer.yaml t02-order-status \
    --models config/models.ollama.yaml --no-probe
```

With Ollama running (`ollama serve`, and the model pulled or signed in for `-cloud`):

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t02-order-status --models config/models.ollama.yaml
uv run vdreamer run t02-order-status --message "Where is my order A1001?"     # second terminal
```

## Expected behavior

`validate` passes with `tools.governed: 1 tool(s)` and `model:chat: ollama http://127.0.0.1:11434 (not contacted)`. A run looks like this (the tool lines are exact; the model's wording will differ):

```
[thread thread_8a55…]
[run running]
-> get_order_status({"order_id": "A1001"}) [custom]
<- ok Order A1001: shipped on 2026-10-01, arriving 2026-10-04 (3 ms)
assistant>
Your order A1001 shipped on 1 October and should arrive on 4 October.

[run completed]
```

`[custom]` is the tool's source: your own code, as opposed to an MCP server. Ask "Where is my parcel?" without an id and the model should ask for one, with no tool call. Ask about `A9999` and the tool answers "No order with id … was found", which the model relays.

## Test it

The tests replace the `chat` alias with a `ScriptedChatModel` whose replies are fixed in advance: first a tool call, then an answer. What they verify is everything *around* the model — that the tool runs, the events are right, and the second model call **saw the tool's result**:

```python
# file: tests/tutorials/test_t02_minimal_react_agent.py
    model = scripted(
        call("get_order_status", order_id="A1001"),
        "Your order A1001 shipped on October 1st and arrives on October 4th.",
    )
    result = await AgentSession(agent, models={"chat": model}).run("Where is order A1001?")

    assert result.status == "completed"
    assert result.tool_calls == [("get_order_status", {"order_id": "A1001"})]
...
    observation = second_prompt[-1]
    assert isinstance(observation, ToolMessage)
    assert observation.text == "Order A1001: shipped on 2026-10-01, arriving 2026-10-04"
```

[`test_t02_minimal_react_agent.py`](../../../../tests/tutorials/test_t02_minimal_react_agent.py) also tests the tool on its own, the no-tool path, and the full gateway with `vdreamer run`. A live check against a real model is opt-in (`test_tutorials_live_ollama.py`, lesson 6).

```bash
uv run pytest tests/tutorials/test_t02_minimal_react_agent.py -v
```

## Common mistakes and debugging

- **`dev: unknown model alias(es) ['chat']: no models.yaml was given`** — pass `--models config/models.ollama.yaml` (or set `VDREAMER_MODELS`).
- **`dev: model 'chat': ollama endpoint http://127.0.0.1:11434 unreachable`** — vdreamer probes every model at startup and refuses to start without it. Start Ollama, or check `base_url`. `validate --no-probe` checks the file without the network.
- **The model never calls the tool** — some small models do not support tool calling at all; pick one that does. Then look at the tool's docstring and the system prompt: the model only knows what they say.
- **`msg.content` is a list, not a string** — vdreamer builds chat models with LangChain's `output_version="v1"`, so `content` is a list of content blocks (text, reasoning, tool calls). Use `msg.text` for the string. This matters as soon as your own code reads model output (lesson 3).
- **Passing `checkpointer=InMemorySaver()` like the LangChain docs** — use `**ctx.agent_kwargs()` instead. `validate` fails `checkpointer` (`compiled the graph with its own InMemorySaver`) and, without vdreamer's middleware, `tools.governed` (`ungoverned tools ['get_order_status']`); `dev`, `worker` and `serve` refuse such an agent. Ungoverned tools would get no redaction, timeout or error containment.
- **Calling `ctx.models["other"]`** — an alias you did not declare in `models` is a `KeyError`: agents only get what they declare.

## Design trade-offs

- **`create_agent` vs. a hand-written loop.** The prebuilt loop is short, correct and supports LangChain middleware (lessons 9, 10, 16, 28). A hand-written graph (lesson 3) gives full control over each step. Start with `create_agent`; move to a graph when you need control flow the loop does not have.
- **Returning "not found" vs. raising.** A string result keeps the conversation going and lets the model explain. Raising is right for genuine failures the model cannot fix (lesson 5).
- **Aliases over model names in code.** One more file to manage, but swapping models (or testing with a scripted one) never touches agent code.
- **`temperature: 0`.** More repeatable answers and tool choices — useful while learning and testing; raise it for more varied phrasing.

## Exercises

1. Add `A1004` with status `"cancelled: refunded on 2026-09-30"` and ask about it.
2. Change the system prompt so the agent always answers in one sentence, and compare runs.
3. Write a test where the model calls the tool for an unknown order id and then apologises. Assert on the `tool.result` summary.
4. Run the same agent against a vLLM server by switching to `--models config/models.vllm.yaml` — no code change.

Next: [Lesson 3 — Inside the reasoning-and-action loop](../t03_react_loop/README.md).
