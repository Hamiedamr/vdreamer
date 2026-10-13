# PRD: LangGraph Agent Serving VDreamer (`vdreamer`)

`protocol_version: 1.0` · Draft v4 · Refs: D1–D19, A1–A23 (discovery); A24–A45 new (§7.4).

---

## 1. System Summary

**Problem.** Each LangGraph agent re-implements REST/SSE, reconnects, sessions, HITL, artifacts, triggers, tools, and model wiring, inconsistently and fragilely.

**Product.** A Python SDK + CLI serving `BaseAgent` implementations as a production REST/SSE API with zero server code (D1), fully compatible with LangChain/LangGraph 1.x.

**Users.** AI developers only. No frontend code ships; client apps integrate via the protocol contract (§6).

**Design rule.** Core ships generic primitives only; domain logic (retrieval, SQL semantics, validation rules) lives in agent code. A capability enters core only if ≥2 reference agents need it and it can't be built cleanly on existing primitives.

**Reference agents (acceptance examples, not features).** Core is P0-complete when these build without touching vdreamer internals (§7.3):

| ID | Reference agent | Mode | Core primitives exercised |
|---|---|---|---|
| UC1 | SQL & Data Analysis | Chat | `ctx.resources` (Neo4j, SQL), `require_approval` with `edit`, table/image/html artifacts, sandbox |
| UC2 | Telecom Site & Antenna Validation | Headless | `/trigger`, `FileRef` + `safe_extract`, VLM alias, `json` artifact, role-gated approval, callback |
| UC3 | Multi-PDF Research & Comparison | Chat + headless | fastembed aliases (dense/sparse/rerank), `ctx.resources` (Qdrant), `emit_citation`, `thinking.delta` |

---

## 2. Architecture & Runtime Model

### 2.1 Components

| Component | Process | Responsibility |
|---|---|---|
| Gateway | `vdreamer serve` (stateless, N replicas) | Auth, REST, SSE replay, interrupt resolution, artifact proxy |
| Worker | `vdreamer worker` (N replicas) | Executes graphs, normalizes events, uploads artifacts, hosts fastembed models |
| Sandbox | Worker subprocess (D19) | Generated code under rlimits |
| Postgres | External | Checkpoints, store, threads, runs, events, interrupts, artifact metadata, queue, outbox |
| MinIO | External | Artifact bytes, inbound files |
| Model servers | External (vLLM/Ollama) | Chat/VLM endpoints (OpenAI-compatible) |
| Agent resources | External, agent-declared | Qdrant, Neo4j, SQL DBs, internal MCP servers, via `resources.yaml` (A35) |

One image, two entrypoints. `vdreamer dev` runs everything in one process on SQLite and a local filesystem store (D11).

### 2.2 Developer Contract (D1, A5)

```python
from vdreamer import BaseAgent, Mode, BuildContext, FileRef
from pydantic import BaseModel

class SiteInput(BaseModel):
    site_id: str
    sender: str
    package: FileRef                  # MinIO key under inputs/ (A17)

class SiteValidator(BaseAgent):
    name = "site-validator"           # route segment
    mode = Mode.HEADLESS              # CHAT | HEADLESS | BOTH
    models = ["vision", "reasoner"]   # models.yaml aliases (D17)
    resources = []                    # resources.yaml aliases (A35)
    input_schema = SiteInput          # required for HEADLESS/BOTH (D16)
    expose_thinking = True
    access_roles = ["netops"]         # agent RBAC via X-User-Roles; empty = any user

    def build(self, ctx: BuildContext):   # any LangGraph graph (§2.4)
        ...
    async def on_run_start(self, run): ...           # optional hooks
    async def on_run_end(self, run, status): ...
```

- **`BuildContext`:** `models`, `resources`, `tools` (§2.7), `checkpointer`, `store`, `settings`, `agent_kwargs()`.
- **Runtime helpers** (`emit_artifact`, `emit_citation`, `emit_progress`, `emit_log`, `require_approval`, `ctx.sandbox`, `ctx.files`, `ctx.idempotency_key`) resolve the current run via context variables and work in any node or tool without parameters.
- **Lifecycle:** `build()` runs once per worker; graph cached.
- **Discovery** (D2): `vdreamer serve pkg.module:agent`, `--agents-dir ./agents`, or a `vdreamer.yaml` registry.

### 2.3 CLI

| Command | Function |
|---|---|
| `dev` · `serve` · `worker` | Single-process hot reload · gateway · worker |
| `run <agent>` | Terminal client: starts a run, pretty-prints events, resolves interrupts inline (A45) |
| `validate` | Aliases resolve; headless agents declare `input_schema` and interrupt `approver_role` (A16); approval routes wired; vdreamer checkpointer used; no ungoverned tools |
| `migrate` | VDreamer schema plus checkpointer/store setup |
| `export-schema` | OpenAPI 3.1, event JSON Schema, input schemas, TS types, conformance kit (§6.1) |
| `doctor --offline` | Probes all backends, model files, MCP servers, sandbox; prints LangChain/LangGraph versions; asserts zero egress |
| `bundle` | Offline wheelhouse and model cache (§5.3) |

### 2.4 LangChain / LangGraph Compatibility (A30, A31, A41–A43)

**Baseline:** `langchain>=1.4,<2` (needed by `langchain.mcp`), `langgraph>=1.2,<2` (v3 event streaming, node timeouts, `RunControl`, stream transformers), `langgraph-checkpoint-postgres`, Python ≥3.11. Both are LTS 1.x lines under semver. CI runs the reference agents against the pinned floor and the latest 1.x minor (A41).

**Accepted graphs.** `build()` may return a `StateGraph` (vdreamer compiles it), any compiled graph, a functional-API `@entrypoint`, `create_agent(...)`, or `create_deep_agent(...)`. For prebuilt agents, `ctx.agent_kwargs()` supplies `{checkpointer, store, context_schema=VDreamerContext, middleware=[VDreamerToolMiddleware()]}`.

**Native primitives used, never reimplemented:**

| Concern | Primitive |
|---|---|
| Persistence | `AsyncPostgresSaver`; `durability="sync"` by default, overridable per agent (A42) |
| Long-term memory | `AsyncPostgresStore`, injected as `ctx.store` |
| Run context | Runtime `context=VDreamerContext`; `ToolRuntime` inside tools |
| Cancel, graceful shutdown | `RunControl.request_drain()`, which leaves a resumable checkpoint |
| Node timeouts, recovery | `add_node(timeout=…, error_handler=…)`, developer-owned; `NodeTimeoutError` and `NodeError` map to `run.status.error` |
| HITL | `interrupt()` / `Command(resume=…)`; `HumanInTheLoopMiddleware` (§4.5) |
| Reasoning | Standard `reasoning` content blocks; models are built with `output_version="v1"` |
| Tools | `BaseTool`, `ToolRuntime`, `@wrap_tool_call`, `langchain.mcp` (§2.7) |

**Event adapter (A31, A43).** One module with two drivers that emit identical protocol events, verified by golden-event tests. Select with `VDREAMER_STREAM_DRIVER`.
- **`v2` (default, stable):** `astream(stream_mode=["messages","updates","custom"], subgraphs=True, version="v2")`. v2 reports interrupts on `values` parts, which this driver doesn't subscribe to, so pending interrupts are read from `aget_state(config).interrupts` after the stream ends.
- **`v3` (opt-in, beta):** `astream_events(version="v3")`. Maps the `messages` projections (text, reasoning, tool calls, usage), lifecycle, subgraphs, and custom events; interrupts come from the run's `interrupts`. Becomes default once stable and passing the golden-event suite.
- **Both:** vdreamer-enveloped custom data becomes artifact/citation/progress/log events; other custom data becomes `log`; `ns` is preserved for subgraphs.
- **Tool events:** governed tools (§2.7) are the single emitter of `tool.call`/`tool.result`. Ungoverned tool messages are mapped with default redaction.

### 2.5 REST Contract (`/api/v1`)

| Method & Path | Semantics |
|---|---|
| `GET /agents` · `/agents/{name}` | Registry: mode, input JSON Schema |
| `POST /agents/{name}/threads` | `201 {thread_id}` |
| `GET /threads` · `/threads/{id}` · `DELETE /threads/{id}` | Owner-scoped list, history from checkpoint, soft delete |
| `POST /threads/{id}/runs` | `{input, config?}` → `202 {run_id}`; `409 {active_run_id, reason}` (D12, A11); `?wait_for_idle=true` queues behind `running` only |
| `GET /runs/{id}` · `POST /runs/{id}/cancel` | Status; cooperative cancel at next superstep |
| `GET /runs/{id}/events` · `GET /threads/{id}/events` | SSE with `Last-Event-ID`; the thread variant joins the active or latest run (reattach after refresh) |
| `POST /agents/{name}/trigger` | Headless (D15): `Idempotency-Key` required; body validated; optional `callback` (A26); `202 {run_id}`; a replayed key returns `200` with the original run (A18) |
| `GET /interrupts` · `/interrupts/{id}` · `POST /interrupts/{id}/resolve` | Approver inbox (role-filtered); §4 |
| `GET /artifacts/{id}` · `/preview` · `/raw` · `/download` | §3.4 |
| `GET /protocol/schema.json` · `/openapi.json` · `/healthz` · `/readyz` · `/metrics` | Contracts, ops |

Errors share one body, `{error: {code, message, details?}}`, with stable codes (e.g., `run_conflict`).

### 2.6 Inter-service Boundaries

- **Clients:** client app → Web Backend → gateway with `Bearer <internal_token>` + `X-User-Id`/`X-User-Roles` (D13); token checked first, two valid for rotation (A14). Upstream services use service tokens (A16).
- **Gateway ↔ worker:** Postgres only (queue rows, `LISTEN/NOTIFY` `{run_id, seq}`, A7). Only workers reach models, MinIO writes, and resources.

### 2.7 Tool Integration (A37–A40)

**One type.** Every tool, whatever its source, is a LangChain `BaseTool`. vdreamer adds no base class, so tools stay portable to plain LangChain code.

| Source | How developers provide it |
|---|---|
| Custom | Plain `@tool` works unchanged. `@vdreamer.tool(redact=[…], timeout_s=…, side_effects=True, approval=…)` is `@tool` plus policy stored in `tool.metadata["vdreamer"]`. |
| LangChain ecosystem | Any `BaseTool` or LangChain toolkit, passed as-is. Policies are attached by name or glob via `BaseAgent.tool_policies = {"sql_db_*": Policy(redact=["query"], side_effects=True)}`. |
| Internal MCP | Declared in `resources.yaml` (`type: mcp`). Discovered tools are named `{server}_{tool}` and selected with `ctx.tools.mcp("netinv", include=[…])`. |

**Runtime access in custom tools:** either a native `runtime: ToolRuntime[VDreamerContext]` parameter (`context` holds run, thread, user, roles, `idempotency_key`; plus `state`, `stream_writer`, `tool_call_id`), or the implicit helpers called directly.

**Governance.** `ctx.tools.govern(tools)` wraps each tool in a proxy that keeps its name and schema; `VDreamerToolMiddleware` (`@wrap_tool_call`) does the same for `create_agent`. Each governed call runs:
1. **Authorize:** tool scope plus optional `requires_roles` against the initiator's roles.
2. **Emit `tool.call`**, redacting policy keys plus defaults (password, token, secret, api_key, authorization).
3. **Approve (optional):** `approval=Policy.approval(kind, approver_role, scope)` calls `require_approval`, a per-tool gate with no graph changes.
4. **Execute** under `timeout_s` (default 60), with `ctx.idempotency_key` for `side_effects=True`.
5. **Normalize:** results over `max_result_chars` (8,000) become a `json`/`file` artifact; the model gets a truncated body plus `artifact_id`. MCP structured/multimodal content maps the same way.
6. **Contain errors:** failures become `ToolMessage(status="error")` with `error.code` (`tool_error|timeout|denied|transport|invalid_args`); the run continues unless `fail_run=True`.
7. **Emit `tool.result`** with `source` and `duration_ms`.

**MCP lifecycle** (A38, A39):

```yaml
# resources.yaml
netinv: {type: mcp, url: https://mcp.netinv.internal/mcp, bearer_env: NETINV_TOKEN,
         startup: required, refresh_s: 300, include: ["get_*", "search_*"], identity: service}
pdfkit: {type: mcp, command: /opt/mcp/pdfkit/bin/server, args: ["--stdio"], env: {LOG_LEVEL: info}, startup: optional}
```

- **Connect:** one `langchain.mcp.MCPAdapter` per server at worker boot; `url` → Streamable HTTP, `command` → stdio; auth on the `fastmcp.Client`.
- **Discover:** `list_tools()` cached per worker, refreshed every `refresh_s`; changes apply to new runs, each recording its tool-snapshot hash.
- **Failures:** a down `required` server fails `/readyz`; an `optional` one is excluded and logged.
- **Elicitation:** surfaces as a LangGraph interrupt and is normalized like any other interrupt source (§4.5).
- **Identity:** `service` bearer (P0); `forward` initiator identity headers (P1).

---

## 3. Session, State & Artifact Management

### 3.1 Identity

- `thread_id` = session.
- `run_id` = one execution = one user turn = `question_id` (A10).
- `/trigger` creates a new thread unless a `thread_id` is passed.

### 3.2 Run Lifecycle

`queued → running → (awaiting_approval → running)* → completed | failed | rejected | expired | cancelled`

- **Claiming:** `FOR UPDATE SKIP LOCKED` with 10 s lease heartbeat.
- **Cancel and shutdown:** cancel calls `RunControl.request_drain()` and the run → `cancelled`. On worker SIGTERM, every active run is drained and requeued without consuming an attempt.
- **Recovery:** expired lease → requeue with `attempt+1` (max 3), resume from checkpoint; the in-flight node re-executes, so side effects use `ctx.idempotency_key` (A6).
- **Approvals:** `awaiting_approval` frees the worker; resolution enqueues a resume.

### 3.3 Event Log & Protocol

**Storage.** `run_events(run_id, seq, type, ns, payload jsonb, ts)`, PK `(run_id, seq)` (A3); SSE `id: {run_id}:{seq}`. Replay sends `seq > N`, then tails live; any replica serves any reconnect.

**Envelope:**

```json
{"v":"1.0","type":"tool.call","run_id":"r_…","thread_id":"t_…","agent":"sql-analyst","seq":42,"ts":"…","ns":[],"payload":{}}
```

| Type | Payload |
|---|---|
| `run.status` | `{status, attempt, error?: {code, message}}` |
| `token.delta` / `thinking.delta` | `{message_id, text}`; thinking suppressed if `expose_thinking=false` |
| `message.completed` | `{message_id, role, text, thinking?, citation_ids[], usage?}` |
| `tool.call` | `{tool_call_id, name, source: custom\|langchain\|mcp:{server}, args}` (redacted, A33) |
| `tool.result` | `{tool_call_id, status: ok\|error, error?: {code, message}, summary, artifact_ids[], duration_ms}`; full output is never streamed |
| `artifact.created` | §3.4 |
| `citation` | `{citation_id, message_id, source: {artifact_id?, uri?, title}, locator: {page?, …}, snippet (≤300 chars), score?}` |
| `interrupt.requested` / `interrupt.resolved` | §4 |
| `progress` | `{label, pct?}` |
| `log` | `{level, message}` |

- **Heartbeat** every 15 s; `X-Accel-Buffering: no` (A22). **Deltas** batched at emission (50 ms / 32 tokens); post-run compaction may drop deltas superseded by `message.completed` (A4).
- **Thinking extraction** (D18, A20): standard `reasoning` content blocks first. For providers that don't emit them, per-alias fallbacks apply: the vLLM `reasoning_content` field, then a stateful `<think>` splitter (tags may span chunks).

### 3.4 Artifacts (D6–D8)

```python
emit_artifact(name: str, kind: ArtifactKind, data: Any, metadata: dict | None = None, renderer: str | None = None) -> ArtifactRef
return Artifact(kind=..., data=..., name=...)
```

| `kind` | Accepted `data` | Stored as | SSE preview |
|---|---|---|---|
| `table` | `pyarrow.Table`, `DataFrame`, row iterator | Parquet (A24) | schema + 10 rows + `row_count` |
| `image` | bytes/path (png, jpg, svg) | as-is | `{width, height}` |
| `html` | string; may embed `artifact://{id}` | as-is | `{title, summary?}` |
| `json` | Pydantic model or dict (≤1 MB) | JSON | full body if ≤64 KB |
| `file` | bytes/path + mime | as-is | `{filename, mime}` |

- **Rules:** no auto-detection (unsupported data → `TypeError`); `renderer` is an optional UI hint (e.g., `verdict.v1`); write order is bytes (`artifacts/{thread_id}/{run_id}/{artifact_id}.{ext}`) → metadata row → event.
- **Tables:** written incrementally as Parquet row groups. `/preview?offset&limit&sort` (limit ≤1000); `/download?format=csv|xlsx|parquet` converts while streaming (XLSX >1,048,575 rows → `422`); `TABLE_MAX_ROWS` (5M) truncates with `metadata.truncated`.
- **HTML:** `artifact://` links are rewritten to `/raw` URLs. Served under a CSP with no external origins and `connect-src 'none'`. Chart JS is vendored at `/static/vendor/` (A13).
- **Access & retention:** gateway-proxied (A8) for thread owner, granted approver, or `artifact:read:all`. Lifecycle deletes after 30 days; missing objects → `410` (A9).

`artifact.created` payload: `{artifact_id, question_id, session_id, kind, renderer?, name, mime_type, size_bytes, preview, view_url, download_url, metadata}`.

### 3.5 Files & Sandbox

**Inputs.** `FileRef` fields carry MinIO keys under `inputs/`; the gateway validates the prefix (A17). Agents read them via `ctx.files.open(ref)`.

**`ctx.files.safe_extract(ref, allow=[…]) -> Manifest`** (A34), in the sandbox: caps sizes, entry count, compression ratio, nesting ≤2; rejects traversal, absolute paths, symlinks, encryption; decodes names UTF-8 → CP437 → CP1256. Returns `[{path, object_key, mime, size, sha256}]`; disallowed types listed as skipped.

**`ctx.sandbox.run(code, inputs, timeout_s=60)`** (D19, A19): dedicated uid, rlimits (CPU, 2 GB memory, file size, nproc), wiped scratch dir, empty env, network off where namespaces permit (containers in P1). Files in `./out/` become artifacts. Pinned libs: pandas, pyarrow, matplotlib, plotly, numpy.

### 3.6 Models (D17, A21, A23, A36)

```yaml
# models.yaml (names illustrative)
aliases:
  reasoner: {provider: vllm, base_url: http://10.20.0.5:8000/v1, model: <chat-model>, api_key_env: VLLM_KEY,
             max_tokens: 8192, temperature: 0.2, thinking: reasoning_content}
  vision:   {provider: vllm, base_url: http://10.20.0.6:8000/v1, model: <vlm>, modalities: [text, image]}
  local:    {provider: ollama, base_url: http://10.20.0.7:11434, model: <model>, thinking: think_tags}
  dense:    {provider: fastembed, kind: embedding, model: <multilingual-dense>}
  sparse:   {provider: fastembed, kind: sparse,    model: <multilingual-sparse>}
  rerank:   {provider: fastembed, kind: rerank,    model: <multilingual-reranker>}
fastembed: {cache_dir: /opt/models/fastembed, local_files_only: true, providers: [CPUExecutionProvider]}
```

- vLLM/Ollama aliases become LangChain chat models. fastembed aliases load once per worker from `cache_dir` as native `TextEmbedding`/`SparseTextEmbedding`/`TextCrossEncoder` objects.
- Agents reference aliases only; `validate`/`doctor` fail on unknown aliases, unreachable endpoints, or missing model files.

### 3.7 Resources (A35)

```yaml
# resources.yaml
vectors:  {type: qdrant, url: https://10.20.1.5:6333, api_key_env: QDRANT_KEY}
semantic: {type: neo4j,  uri: bolt://10.20.1.6:7687, user_env: NEO4J_USER, password_env: NEO4J_PASS, read_only: true}
warehouse:{type: sql,    url_env: DWH_URL, read_only: true}
```

Workers open pooled async clients at start and inject them via `ctx.resources[alias]`; `read_only` forces read sessions/credentials. A new type is one adapter class (`connect`, `probe`, `close`). Query APIs are not wrapped. MCP entries are covered in §2.7.

---

## 4. Human-in-the-Loop (HITL) Protocol

### 4.1 SDK

```python
decision = require_approval(
    kind="sql_review",                  # renderer hint
    payload={"sql": sql, "dry_run": stats, "lineage": nodes},
    actions=["approve", "reject", "edit", "feedback"],
    approver_role=None,                 # required in headless runs (A16)
    scope=None,                         # e.g. "db:write:prod"
    edit_schema=SqlEdit,                # validates edit payloads
    ttl_hours=24,
)
```

- Wraps LangGraph `interrupt(value)`; the gateway enforces the serialized policy (A15). Resume restarts the node from the top, so pre-work belongs in a preceding node.
- **`ApprovalGate`:** a prebuilt subgraph, `prepare → gate → {approve|edit: execute, reject: on_reject, feedback: replan}`. `validate` flags unwired routes.

### 4.2 Record & Events

- **Record:** `interrupts(interrupt_id, run_id, kind, scope, approver_role, actions[], payload, status: pending|resolved|expired|cancelled, version, expires_at, resolved_by, resolution)`.
- **`interrupt.requested`:** `{interrupt_id, kind, scope, approver_role, actions[], payload, edit_schema (JSON Schema), expires_at}`.
- **`interrupt.resolved`:** `{interrupt_id, action, resolved_by}`.

### 4.3 Resolution

`POST /interrupts/{id}/resolve` with body `{action, payload?, text?, expected_version}`. For multi-action interrupts, send `{decisions: [{action, payload?, text?}], expected_version}`, one entry per action and in order.

| Action | Effect |
|---|---|
| `approve` | `Command(resume={"action":"approve"})` |
| `edit` | `payload` is validated against `edit_schema`, passed in the resume value, and written to state before execution |
| `feedback` | `{text}` → the gate routes to `replan` |
| `reject` | Chat: routes to `on_reject` and the conversation continues. Headless: run → `rejected` |
| `respond` | `{text}` returned to the agent as the tool's answer ("ask user" tools) |

- **Responses:** `200`; `403 forbidden_role`; `409` on a stale version, where the first resolver wins (A12); `410` expired; `422` invalid edit.
- **Authority:** the initiator approves by default. If `approver_role` is set, that role is required. Scopes matching `HITL_STRICT_SCOPES` require an approver who is not the initiator (four-eyes, A32).

### 4.4 Expiry & Notification (D10)

- **Expiry:** a leader-elected sweeper expires overdue interrupts; run → `expired`, thread unlocks.
- **Notifications:** headless or role-gated interrupts go via the outbox to an HMAC-signed webhook and/or SMTP relay, with deep link `{WEB_BASE_URL}/review/{interrupt_id}`. At-least-once; receivers dedup on `event_id`.
- **Run callbacks:** the outbox also delivers `/trigger` callbacks (`completed|failed|interrupt`) to URLs on `CALLBACK_ALLOWLIST` (A26).

### 4.5 Interrupt Sources (A44)

Every LangGraph interrupt is normalized into the same record, event, and resolve endpoint, whatever raised it:

| Source | `kind` | Actions | Resume value built by the gateway |
|---|---|---|---|
| `require_approval` | developer-defined | declared | `{action, payload?, text?}` |
| `HumanInTheLoopMiddleware` (`interrupt_on`) | `tool_review` | each action's `allowed_decisions` (approve, edit, reject, respond) | `{"decisions": [...]}`, in action order |
| MCP elicitation (`langchain.mcp`) | `mcp_elicitation` | edit, reject | `{"responses": {...}}` |
| Raw `interrupt(value)` | `generic` | respond | resolver payload, verbatim |

Policy fields (`approver_role`, `scope`, TTL) can be attached to non-vdreamer sources via `BaseAgent.interrupt_policies`, keyed by `kind` or tool name.

---

## 5. Air-Gapped & Enterprise Topology

### 5.1 Reference Topology

| Segment | Hosts | Inbound from |
|---|---|---|
| VM-WEB | Client app + Web Backend (LDAP), outside vdreamer | Users |
| VM-GW | Gateway | VM-WEB, upstream triggers |
| VM-WRK | Workers, sandbox, fastembed models | none |
| VM-LLM | vLLM/Ollama | VM-WRK |
| VM-DB | Postgres | VM-GW, VM-WRK |
| VM-OBJ | MinIO | VM-GW, VM-WRK, upstream (`inputs/` write-only) |
| Agent resources | Qdrant, Neo4j, SQL DBs, MCP servers | VM-WRK |

Internal DNS only, with no default route. The gateway binds to the internal interface only (A14).

### 5.2 Zero-Egress

- **Offline libs:** LangSmith tracing forced off; `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; fastembed `local_files_only` (A36); bundled `TIKTOKEN_CACHE_DIR`; vendored chart JS, no CDNs.
- **Tool egress** (A40): VM network policy is the authoritative control. In addition, every declared endpoint (model, resource, MCP `url`) must match `NETWORK_ALLOWLIST` (internal CIDRs/hostnames) at config load and at connect. stdio MCP `command` values must be absolute paths to preinstalled binaries, and `validate` rejects downloading launchers (`npx`, `uvx`, `pip`). stdio servers run as worker subprocesses with rlimits and only their declared env.
- **Telemetry:** none, and no update checks. `doctor --offline` runs in CI with egress blocked.

### 5.3 Packaging

- **Python:** pins `langchain>=1.4,<2` and `langgraph>=1.2,<2` (A41). `vdreamer bundle` → `wheelhouse/` + hashed lock; `pip install --no-index --find-links`. Extras: `[postgres]`, `[minio]`, `[fastembed]`, `[qdrant]`, `[neo4j]`, `[mcp]` (`langchain[mcp]`), `[sandbox]`.
- **Models:** `vdreamer bundle --models` pre-fetches fastembed aliases on a connected build host; ship baked into the image or as a read-only volume.
- **OCI:** `vdreamer:<ver>` (gateway/worker) and `vdreamer-sandbox:<ver>` (P1), pushed to the internal registry.
- **Protocol artifacts:** each release attaches the `export-schema` output and conformance kit (§6.1). No npm packages.
- **vLLM prerequisites** (A21): `--enable-auto-tool-choice`, a matching `--tool-call-parser`, and `--reasoning-parser` for thinking models.

### 5.4 Config, Secrets, Observability

- **Config:** env vars; YAML files reference secrets via `*_env` keys only (A23). Logs exclude identity headers beyond `user_id` and redacted args.
- **Metrics:** queue depth, run latency, SSE connections, model latency/tokens per alias, tool latency/errors per source, MCP server health, sandbox timeouts, outbox lag.

---

## 6. Frontend Integration Spec

The contract any client must follow; vdreamer ships no client code.

### 6.1 Protocol Contract & Conformance Kit

- **Artifacts:** `export-schema` publishes OpenAPI 3.1, the event JSON Schema, per-agent input schemas, and generated TS types.
- **Conformance kit:** recorded SSE streams (chat, mid-run reconnect, tool overflow, all interrupt sources, artifacts, citations), each with the expected client state.
- **Normative client behavior:**
  - Send `X-Protocol-Version: 1`; reject a major-version mismatch.
  - Consume SSE with fetch (D14). Reconnect with backoff + jitter, resuming via `Last-Event-ID`, or use `/threads/{id}/events` after a page refresh.
  - Deduplicate by `seq`. Ignore unknown event types. Treat unknown artifact kinds/renderers as downloads.
  - Fold events into `{status, messages[{id, text, thinking, citations[]}], toolCalls[], artifacts[], interrupts[], progress, logs}`.

### 6.2 Web Backend Proxy

- Proxies `/api/v1/**` with internal token and identity headers; no SSE buffering; idle timeout > 2× heartbeat; forwards `Last-Event-ID`. Artifact URLs are same-origin with the session cookie, so links need no JS-attached headers (A25).

### 6.3 Rendering Hints (non-normative)

| Kind | Suggested rendering | Source |
|---|---|---|
| `table` | Server-paged grid, CSV/XLSX export | `/preview`, `/download` |
| `image` | `<img>` | `/raw` |
| `html` | `<iframe sandbox="allow-scripts">`, no `allow-same-origin` | `/raw` |
| `json` | Tree, or a view keyed by `renderer` | preview, `/raw` |
| `file` | Download; PDFs at `#page=N` | `/raw`, `/download` |

Citations bind to `message_id` and open `source` at `locator.page`.

### 6.4 Interrupt Handling Obligations

- Offer only `actions[]`. Build edit forms from `edit_schema`. For `tool_review`, submit `decisions[]` in action order.
- Send `expected_version`; refresh on `409`. Block new prompts during `running`/`awaiting_approval` unless using `wait_for_idle`.

---

## 7. Scope Boundaries

### 7.1 Functional Matrix

| Capability | P0 | P1 |
|---|---|---|
| `BaseAgent`, routing, hooks, full CLI | ● | |
| Protocol v1, resumable SSE, heartbeat, delta coalescing, thinking extraction | ● | |
| LangChain/LangGraph 1.x: v2 driver (v3 opt-in), `create_agent`/Deep Agents/functional API, store, `RunControl`, HITL middleware, interrupt normalization | ● | |
| Postgres queue, leases, crash recovery, idempotency | ● | |
| Artifacts: 5 kinds, Parquet tables, CSV/XLSX export, HTML rewriting + CSP, `renderer` hint | ● | |
| `emit_citation` + `citation` event | ● | |
| Governed tool pipeline (`govern`, middleware, `@vdreamer.tool`, `tool_policies`, result overflow → artifact) | ● | |
| MCP: Streamable HTTP + stdio, discovery, refresh, elicitation mapping, service identity | ● | |
| MCP `forward` identity; legacy SSE transport | | ● |
| `require_approval`, `ApprovalGate`, RBAC, four-eyes, TTL, outbox notifications | ● | |
| `/trigger`, `FileRef`, `safe_extract`, run callbacks | ● | |
| `models.yaml` (vLLM, Ollama, fastembed), `resources.yaml` (Qdrant, Neo4j, SQL, MCP), subprocess sandbox | ● | |
| Protocol artifacts + conformance kit; `vdreamer run` terminal client | ● | |
| Prometheus metrics, JSON logs | ● | |
| Reference agents UC1–UC3 in `examples/` | ● | |
| Container sandbox; Redis adapter; OTLP; quotas; `vdreamer new`; time-travel API | | ● |

### 7.2 Non-Goals

- **External concerns:** LLM serving or fine-tuning; end-user auth/LDAP (D13); mail fetching (D15); cron (schedulers call `/trigger`).
- **Domain logic in core:** retrieval pipelines, chunking, ingestion, and vector-schema management all live in agent code.
- **Semantic layer:** building or syncing the Neo4j graph.
- **MCP:** prompts, resources, sampling; exposing vdreamer agents as MCP servers.
- **Platform scope:** generated server code (D1); LangGraph Platform API compatibility; non-LangChain frameworks; cloud model providers.
- **Frontend:** any client application, UI component, or client SDK (Angular or otherwise). Client teams build against §6.

### 7.3 Reference Agents (`examples/`)

**UC1 SQL & Data Analysis.** `schema_link` (Cypher via `ctx.resources["semantic"]`) → `generate_sql` → `dry_run` (EXPLAIN) → `ApprovalGate(kind="sql_review")` with lineage → execute on `ctx.resources["warehouse"]` → `table` → sandboxed chart → `html` report. Non-SELECT adds `approver_role` + `db:write`.

**UC2 Site & Antenna Validation.** Mail worker uploads the ZIP → `/trigger` (`package: FileRef`) → `safe_extract` → VLM inspection + MCP tools → `json` verdict (`renderer="verdict.v1"`) → `ApprovalGate(approver_role="site_approver")` → callback.

**UC3 Multi-PDF Research** (EN + AR). *Ingest* (headless): sandbox parse → page-offset chunks → `dense` + `sparse` → Qdrant upsert. *Chat* (`create_agent`): hybrid `query_points` (RRF) → `rerank` → answer with reasoning → `emit_citation` per chunk → comparison `table`.

### 7.4 New Assumptions

- **A24:** Parquet tables. **A25:** Same-origin artifact URLs. **A26:** Callbacks to allowlisted URLs only. **A27–A29:** Retired.
- **A30:** Python ≥3.11. **A31:** All LangGraph streaming specifics live in one adapter module. **A32:** `HITL_STRICT_SCOPES` four-eyes. **A33:** Redaction = policy keys + defaults.
- **A34:** `safe_extract` is core (security-critical). **A35:** Resources injected as native clients. **A36:** fastembed in-process from vendored `cache_dir`.
- **A37:** All tools are `BaseTool`; governance is a proxy. **A38:** `langchain.mcp` (beta) isolated and pinned. **A39:** MCP servers live in `resources.yaml`. **A40:** Network policy is authoritative; `NETWORK_ALLOWLIST` is defense in depth.
- **A41:** Supported range is `langchain` 1.4+ and `langgraph` 1.2+ within 1.x; a 2.0 release requires a vdreamer major version.
- **A42:** `durability="sync"` by default, because crash recovery (A6) needs every step checkpointed. Agents may opt into `async`.
- **A43:** The stable v2 stream is the default driver. v3 event streaming (beta) is opt-in until it is stable.
- **A44:** Interrupts from `require_approval`, HITL middleware, MCP elicitation, and raw `interrupt()` all share one protocol.
- **A45:** Users are AI developers: `vdreamer run` replaces a UI for testing; the conformance kit replaces a client SDK.
