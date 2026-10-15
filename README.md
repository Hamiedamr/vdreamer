# vdreamer

Serve your LangGraph agents as a production REST/SSE API — without writing a server.

You write an agent as a small Python class around a LangGraph graph (a `StateGraph`, a `create_agent` agent, or a functional `@entrypoint`). vdreamer runs it: threads and runs, streaming over SSE, persistence and crash recovery, tool governance, human approvals, artifacts, and a generated OpenAPI contract for the apps that call it. It is built for self-hosted models (vLLM, Ollama) and air-gapped deployments, and it sends nothing anywhere you didn't configure.

- **Your code stays plain LangChain/LangGraph** (1.x). Tools are ordinary LangChain tools.
- **One command for local development** (`vdreamer dev`, SQLite, no infrastructure), two for production (`vdreamer serve` + `vdreamer worker`, Postgres).
- **A stable client protocol**: REST under `/api/v1`, typed SSE events, Swagger UI, and exportable schemas for your frontend or backend team.

## Install

Python 3.11+. Add vdreamer to your project from the GitHub release (or your internal package index):

```bash
pip install "vdreamer @ git+https://github.com/Hamiedamr/vdreamer@v0.1.0"
# or, with uv:
uv add "vdreamer @ git+https://github.com/Hamiedamr/vdreamer@v0.1.0"
```

Optional extras, added as `vdreamer[postgres,minio]` and so on:

| Extra | For |
|---|---|
| `postgres` | `vdreamer serve` / `worker` (production state store) |
| `minio` | artifact storage in production |
| `sandbox` | table artifacts and charts (pyarrow, pandas, matplotlib, plotly), `ctx.sandbox` |
| `mcp` | tools from MCP servers |
| `sql`, `qdrant`, `neo4j` | database and vector-store resources |
| `fastembed` | local embedding models |

For an offline (air-gapped) install, see [docs/operations.md](docs/operations.md).

## Quickstart

**1. Write an agent** in your project, e.g. `my_agent.py`:

```python
from langchain.agents import create_agent

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode


@vdreamer.tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It's always sunny in {city}!"


class WeatherAgent(BaseAgent):
    name = "weather"
    mode = Mode.CHAT
    models = ("chat",)  # an alias from models.yaml

    def build(self, ctx: BuildContext):
        return create_agent(
            ctx.models["chat"],
            tools=[get_weather],
            system_prompt="You are a helpful weather assistant.",
            **ctx.agent_kwargs(),  # checkpointer, store, run context, tool governance
        )


agent = WeatherAgent()
```

**2. Point the model alias at your model server** in `models.yaml`. For Ollama:

```yaml
aliases:
  chat:
    provider: ollama
    base_url: http://127.0.0.1:11434
    model: qwen3:8b
```

Or for any OpenAI-compatible server (vLLM, llama.cpp `llama-server`, LM Studio, LocalAI, TGI, …), use `provider: vllm` with the server's `/v1` URL:

```yaml
aliases:
  chat:
    provider: vllm
    base_url: http://127.0.0.1:8000/v1
    model: Qwen/Qwen3-8B          # the model id the server lists at /v1/models
    api_key_env: LLM_API_KEY      # optional: the *name* of the env var holding the key
```

Keys never go in the file: `export LLM_API_KEY=...` and the alias reads it at startup. Leave `api_key_env` out for servers that need no key.

**3. Check it, then serve it** from your project folder. The token is any secret of 16+ characters; your client sends it as a bearer token.

```bash
export VDREAMER_INTERNAL_TOKEN=change-me-to-a-long-secret
vdreamer validate my_agent:agent --models models.yaml
vdreamer dev --agent my_agent:agent --models models.yaml
```

`vdreamer dev` runs the API and a worker in one process, keeping everything in `./.vdreamer/dev.db` (add `.vdreamer/` to your `.gitignore`). Add `--reload` to restart on code changes.

**4. Talk to it.** From another terminal, with the same token:

```bash
vdreamer run weather --message "What's the weather in Cairo?"
```

```
[thread thread_6830a875…]
[run running]
-> get_weather({"city": "Cairo"}) [custom]
<- ok It's always sunny in Cairo! (0 ms)
assistant>
It's always sunny in Cairo!
[run completed]
```

Or open **http://127.0.0.1:8000/api/v1/docs** (Swagger UI; click *Authorize* and paste the token), or use plain HTTP:

```bash
H=(-H "Authorization: Bearer $VDREAMER_INTERNAL_TOKEN" -H "X-User-Id: alice" -H "Content-Type: application/json")
curl "${H[@]}" -X POST http://127.0.0.1:8000/api/v1/agents/weather/threads
# → {"thread_id": "thread_…"}
curl "${H[@]}" -X POST http://127.0.0.1:8000/api/v1/threads/<thread_id>/runs \
     -d '{"input": {"messages": [{"role": "user", "content": "Weather in Paris?"}]}}'
# → 202 {"run_id": "run_…", "status": "queued"}
curl -N "${H[@]}" http://127.0.0.1:8000/api/v1/threads/<thread_id>/events
# → SSE: run.status, tool.call, tool.result, token.delta, message.completed, …
```

## Writing agents

An agent is a `BaseAgent` subclass with a `build()` method that returns a LangGraph graph. Return a `StateGraph` uncompiled (vdreamer compiles it with its checkpointer), a `create_agent(...)` agent with `**ctx.agent_kwargs()`, or a functional `@entrypoint(checkpointer=ctx.checkpointer)`.

| Attribute | Default | Meaning |
|---|---|---|
| `name` | — | the agent's name in the API (`/agents/{name}`) |
| `mode` | `Mode.CHAT` | `CHAT` (threads and messages), `HEADLESS` (triggered by systems), or `BOTH` |
| `input_schema` | `None` | a pydantic model the input is validated against; required for `HEADLESS`/`BOTH` |
| `models`, `resources` | `()` | aliases this agent may use, read in `build()` as `ctx.models[...]` / `ctx.resources[...]` |
| `access_roles` | `()` | roles allowed to use the agent (empty = any user) |
| `expose_thinking` | `True` | stream the model's reasoning to clients |
| `tool_policies` | `{}` | governance policies for tools by name or glob, e.g. third-party toolkits |
| `durability` | `"sync"` | checkpoint every step (crash-safe); `"async"` trades that for speed |

`build(ctx)` gets a `BuildContext`: `ctx.models`, `ctx.resources`, `ctx.tools` (`govern([...])`, `mcp("server")`), `ctx.checkpointer`, `ctx.store`, and `ctx.agent_kwargs()` for `create_agent`. Inside a running node or tool, `vdreamer.current_context()` gives the run, thread, user and roles.

`vdreamer validate pkg.module:agent` checks all of this (and probes the models with `--models`); `dev`, `worker` and `serve` refuse an agent that fails it.

### What clients see

Only these reach a client — plain state keys don't:

- **Messages** — every `AIMessage` your graph produces streams as `token.delta` events and ends with `message.completed`. Return an `AIMessage` for anything a user should read.
- **Progress and logs** — `vdreamer.emit_progress("parsing", 40)` (a percentage, 0–100) and `vdreamer.emit_log("…")`.
- **Artifacts** — `vdreamer.emit_artifact("results", ArtifactKind.TABLE, rows)` (or `await aemit_artifact(...)` in async code, or `return Artifact(...)` from a tool) stores tables, charts, JSON, HTML or files and announces them; clients page, preview and download them. Tables need the `sandbox` extra.
- **Citations** — `vdreamer.emit_citation(message_id, source, locator, snippet)` binds sources to an answer.
- **Tool calls**, **approvals**, and the run's **status**.

Chat models from `models.yaml` return content as a list of content blocks (LangChain's `output_version="v1"`): read `message.text` for the string, not `message.content`.

### Tools

Decorate tools with `@vdreamer.tool` (a drop-in for LangChain's `@tool`) to set a policy, and run them through governance: `create_agent` with `**ctx.agent_kwargs()` does it for you; in a `StateGraph`, call `ctx.tools.govern([...])`. A governed call is authorized, redacted in events, run under a timeout, and contained if it fails — the error goes back to the model and the run continues:

```python
@vdreamer.tool(redact=["api_key"], timeout_s=30, max_result_chars=8000)
def search(query: str, api_key: str) -> str: ...
```

Arguments named `password`, `token`, `secret`, `api_key` or `authorization` are always masked, and so are secrets an exception echoes. Results longer than `max_result_chars` are stored as an artifact and the model gets a truncated preview. Mark tools that change things with `side_effects=True` (they get an idempotency key in `current_context()`), and gate them with `approval=ApprovalSpec(kind="…", approver_role="…")`.

**MCP servers** are declared in `resources.yaml` and selected per agent:

```yaml
netinv: {type: mcp, url: https://mcp.netinv.internal/mcp, bearer_env: NETINV_TOKEN, include: ["get_*"]}
```

```python
tools = ctx.tools.mcp("netinv")
```

### Human approvals

Pause a run for a person's decision from any node:

```python
decision = vdreamer.require_approval(
    "sql_review", {"sql": sql}, actions=("approve", "reject", "edit", "feedback"),
    approver_role="dba",  # optional: who may decide (otherwise the user who started the run)
)
if decision["action"] == "approve": ...
```

The run waits in `awaiting_approval` (it survives restarts) until someone resolves it with `POST /api/v1/interrupts/{id}/resolve` — or in the terminal, where `vdreamer run` prompts inline and `vdreamer run <agent> --inbox` lists what an approver may decide. The decision carries `resolved_by`. LangChain's `HumanInTheLoopMiddleware` and MCP elicitation work through the same endpoint. For a full approve / edit / re-plan / reject flow, `vdreamer.ApprovalGate` builds the subgraph for you.

### Headless agents

Agents with `mode = Mode.HEADLESS` (or `BOTH`) are started by other systems:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agents/<name>/trigger \
     -H "Authorization: Bearer <service token>" -H "Idempotency-Key: order-1234" \
     -H "Content-Type: application/json" \
     -d '{"input": {...}, "callback": "https://erp.internal/hooks/agent"}'
```

The `Idempotency-Key` makes retries safe (the same key returns the same run). Service tokens are configured with `VDREAMER_SERVICE_TOKENS=erp:<token>`, and callback hosts must be listed in `CALLBACK_ALLOWLIST`. Callbacks fire when the run completes, fails or waits for approval; they are HMAC-signed and retried with backoff. From the terminal: `vdreamer run <name> --trigger --input '{...}'`.

Files come in as `vdreamer.FileRef` fields in the input schema (keys under `inputs/` in object storage). In a node, `files = vdreamer.current_context().files`; read with `files.open(ref)`, and unpack a ZIP safely (in a sandboxed process, with size and path checks) with `files.safe_extract(ref, allow=["*.jpg"])`.

## Models and resources

Agents name aliases; `models.yaml` and `resources.yaml` map them to real endpoints, so the same agent runs against a laptop Ollama and a production vLLM cluster. Pass them with `--models` / `--resources` or `VDREAMER_MODELS` / `VDREAMER_RESOURCES`.

```yaml
# models.yaml
aliases:
  # provider vllm = any OpenAI-compatible chat completions server
  reasoner: {provider: vllm, base_url: http://10.20.0.5:8000/v1, model: qwen3-32b,
             api_key_env: VLLM_KEY, thinking: reasoning_content}
  local:    {provider: ollama, base_url: http://10.20.0.7:11434, model: llama3, thinking: think_tags}
  dense:    {provider: fastembed, kind: embedding, model: BAAI/bge-small-en-v1.5}
fastembed: {cache_dir: /opt/models/fastembed}
```

```yaml
# resources.yaml
vectors:   {type: qdrant, url: http://10.20.0.9:6333}
graph:     {type: neo4j, uri: bolt://10.20.0.8:7687, user_env: NEO4J_USER, password_env: NEO4J_PASSWORD, read_only: true}
warehouse: {type: sql, url_env: DWH_URL, read_only: true}   # plus an async driver: aiosqlite, asyncpg, psycopg
```

Secrets are never written in these files — only the names of environment variables that hold them (`*_env`). Models and resources open once when a worker starts; a worker that can't reach them refuses to start. `thinking` extracts a model's reasoning so clients can show it.

## Running in production

```bash
export VDREAMER_PG_DSN=postgresql://vdreamer@db/vdreamer
vdreamer migrate                                   # once, and after upgrades
vdreamer serve  --agent my_agent:agent             # the API: stateless, run as many as you like
vdreamer worker --agent my_agent:agent --models models.yaml --resources resources.yaml
```

The API and the workers talk only through Postgres; only workers reach model servers and resources. Artifacts go to MinIO/S3 (`VDREAMER_MINIO_ENDPOINT`, `_ACCESS_KEY`, `_SECRET_KEY`). The repository's `Dockerfile` builds one image for both roles (`docker run <image> serve …` / `worker …`); layer your agent code on top of it.

For air-gapped hosts, `vdreamer bundle` builds a hash-pinned offline install set on a connected machine, and `vdreamer doctor --offline` proves a host makes no outbound connections. [docs/operations.md](docs/operations.md) covers the install, the image, every environment variable, and upgrades.

## For client developers

`vdreamer export-schema --out vdreamer-schema` writes the OpenAPI 3.1 spec, the event JSON Schema, each agent's input schema, TypeScript types, and a conformance kit of recorded streams to test a client against. [docs/protocol.md](docs/protocol.md) explains how to consume the event stream, reconnect with `Last-Event-ID`, and resolve approvals. Calls carry the bearer token plus `X-User-Id` (and optionally `X-User-Roles`); your web backend asserts who the user is.

## CLI

| Command | Does |
|---|---|
| `vdreamer dev` | API + worker in one process on SQLite; `--input JSON` runs once and prints every event |
| `vdreamer run <agent>` | terminal client: chat, trigger, resolve approvals, reattach with `--thread` |
| `vdreamer validate <spec>` | check an agent and its model/resource aliases |
| `vdreamer serve` / `worker` / `migrate` | production processes and schema setup (Postgres) |
| `vdreamer export-schema` | the client contract: OpenAPI, event schema, TypeScript types, conformance kit |
| `vdreamer bundle` | offline install set for air-gapped hosts |
| `vdreamer doctor --offline` | check an install and assert zero egress |

Agents are named as `module:attribute` from your project folder, or by name through a registry file (`--registry vdreamer.yaml` with `agents: {weather: my_agent:agent}`) or an `--agents-dir`. Every command has `--help`.

## License

Apache License 2.0 — see [LICENSE](LICENSE). vdreamer bundles Swagger UI (Apache-2.0) and plotly.js (MIT) for offline use; see [NOTICE](NOTICE).

Contributing or working on vdreamer itself: see [CONTRIBUTING.md](CONTRIBUTING.md).
