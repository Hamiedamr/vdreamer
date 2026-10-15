# Playground — first-contact agents and LangChain docs examples

Small agents for trying vdreamer for the first time, plus examples from the LangChain/LangGraph docs served unchanged except for where the model comes from. Each agent lives in `src/playground/` and is listed in `vdreamer.yaml`.

| Agent | Mode | Model | What it shows |
| --- | --- | --- | --- |
| `hello` | chat | none (a fake streaming model) | token streaming, `progress`/`log`, a governed tool with redaction, a `table` artifact, `require_approval` |
| `broken` | headless | none | contained tool failures (`tool_error`, `timeout`, oversized result → artifact), then a node error |
| `calculator` | chat | `chat` | [LangGraph quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart), Graph API |
| `calculator-functional` | chat | `chat` | the same quickstart, Functional API (`@entrypoint`/`@task`) |
| `joke-chain` | headless | `chat` | [workflows: prompt chaining](https://docs.langchain.com/oss/python/langgraph/workflows-agents) |
| `router` | both | `chat` | [workflows: routing](https://docs.langchain.com/oss/python/langgraph/workflows-agents) with structured output |
| `hitl-tools` | chat | `chat` | [LangChain human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop): `create_agent` + `HumanInTheLoopMiddleware` |

## Run it

From the repo root (the package is a uv workspace member, so it is importable after `uv sync`). The token is any string of 16+ characters:

```
export VDREAMER_INTERNAL_TOKEN=dev-token-0123456789
cd examples/playground
uv run vdreamer dev --registry vdreamer.yaml --agent hello
uv run vdreamer run hello --message "hi"              # second terminal
```

Open `http://127.0.0.1:8000/api/v1/docs` for the Swagger UI.

The model-backed agents need a `chat` alias. `config/models.ollama.yaml` points it at Ollama's `gemma4:31b-cloud` (the `-cloud` tag runs on ollama.com — use `gemma4:31b` to stay local); `config/models.yaml` is the vLLM production shape:

```
uv run vdreamer dev --registry vdreamer.yaml --agent calculator --models config/models.ollama.yaml
uv run vdreamer run calculator --message "Add 3 and 4, then multiply by 10."

uv run vdreamer dev --registry vdreamer.yaml --agent router --models config/models.ollama.yaml
uv run vdreamer run router --trigger --input '{"input": "Write me a joke about cats"}'

uv run vdreamer dev --registry vdreamer.yaml --agent hitl-tools --models config/models.ollama.yaml
uv run vdreamer run hitl-tools --message "Save 'hello' to notes.txt"
```

Everything above is also covered offline, with the model scripted: `uv run pytest tests/acceptance/test_playground.py` (from the repo root).

One-shot runs without a server validate the input, print every event, and say why a run failed:

```
uv run vdreamer dev --registry vdreamer.yaml --agent broken --input '{"message": "go"}'
```

## What changed from the docs

- `init_chat_model("…")` / `model="gpt-5.5"` → `ctx.models["chat"]`. vdreamer serves self-hosted models only; aliases come from `models.yaml`.
- `build()` returns the `StateGraph` uncompiled; vdreamer compiles it with its own checkpointer. `create_agent` gets `**ctx.agent_kwargs()` instead of `checkpointer=InMemorySaver()`.
- A Functional API `@entrypoint` receives the run input (`{"messages": [...]}`) rather than a bare list, and takes `checkpointer=ctx.checkpointer`.
- Tools go through `ctx.tools.govern([...])` (or the `create_agent` middleware from `agent_kwargs()`), so clients see `tool.call`/`tool.result`.
- Node code reads `msg.text` where the docs read `msg.content`: vdreamer builds chat models with `output_version="v1"` (PRD §2.4), so `content` is a list of content blocks, and the prompt-chaining gate (`"?" in joke`) would never pass on it.
- The router asks for `with_structured_output(Route, method="function_calling")`. The default `json_schema` method relies on Ollama's `format` field, which `-cloud` models ignore (they answer in prose and parsing fails); tool calling works on every tool-capable model.
- Final state keys (`output`, `final_joke`) are not protocol events: clients see the LLM calls as messages, plus any artifacts, logs and progress the agent emits.
