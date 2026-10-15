# `vdreamer`: LangGraph Agent Serving Toolkit Milestones

This file is derived from `docs/PRD.md` (Draft v4, protocol_version 1.0).

- Milestones are sequential, and each one lists its exit criteria.
- P1 capabilities are explicitly deferred (§7.1).
- **Two scopes, kept separate:**
  - **Development (uv).** uv is the package manager that developers and coding agents use to build vdreamer in this repo: environments, dependencies, lockfile, tests, CI, release builds. `AGENTS.md` instructs coding agents to use uv only.
  - **Product (uv-agnostic).** vdreamer itself never imports, shells out to, or requires uv. It ships as standard wheels, installs with plain `pip`, and its `bundle` command produces pip-installable wheelhouses for operators.

---

## Milestone 0: Repo Bootstrap & CI Foundation

**Goal:** a buildable, testable Python package skeleton. The repo is developed with uv, the package is installable without it, and the version pins (A41) are enforced from day one.

**Project definition**
- [x] `uv init --lib vdreamer` produces a standard PEP 621 `pyproject.toml` with `requires-python = ">=3.11"` and a standard build backend (hatchling), so `pip install vdreamer` works without uv.
- [x] `.python-version` pins the CI/dev interpreter (3.11).
- [x] Optional extras under `[project.optional-dependencies]`: `postgres`, `minio`, `fastembed`, `qdrant`, `neo4j`, `mcp` (`langchain[mcp]`), `sandbox`.
- [x] Runtime pins, added with `uv add`: `langchain>=1.4,<2` (needed for `langchain.mcp`), `langgraph>=1.2,<2`, `langgraph-checkpoint-postgres`.
- [x] Dev tooling in a `[dependency-groups] dev` group, added with `uv add --dev`: pytest, pytest-asyncio, ruff, mypy. Golden-event harness deps deferred to Milestone 2, when the harness exists (noted in `pyproject.toml`).

**Lockfile and commands**
- [x] `uv.lock` is committed and is the single source of truth. CI fails on drift via `uv lock --check` (plus a clean `uv sync --locked --extra postgres` step).
- [x] Standard dev commands, recorded in `AGENTS.md` for humans and coding agents:
  - `uv sync --locked --all-extras` (dev env)
  - `uv run pytest` (`-m smoke` is the floor/latest subset)
  - `uv run ruff check .` (`.agents/` excluded — installed skills are reference material)
  - `uv run mypy` (strict)
  - `uv run vdreamer …`

**CI dependency matrix (A41)**, as three jobs:
- [x] **locked:** `uv lock --check` → `uv sync --locked --all-extras` → ruff + mypy + tests.
- [x] **floor:** `uv lock --resolution lowest-direct` (not committed) → `uv sync --frozen --all-extras` → smoke tests against the minimum `langchain`/`langgraph` versions (verified locally: langchain 1.4.0 / langgraph 1.2.11).
- [x] **latest:** `uv lock --upgrade` (not committed) → `uv sync --frozen --all-extras` → smoke tests against the latest 1.x minor.
- Note: floor/latest sync with `--frozen` — a plain `uv sync` after `uv lock --resolution lowest-direct` silently re-resolves to highest.

**uv-independence check (product scope)**
- [x] CI job builds the wheel with `uv build`, then installs it into a clean `python -m venv` using plain `pip` (no uv on PATH), and runs `vdreamer --version` and `vdreamer validate` on a sample agent (`examples/smoke/pinger.py` until UC1–UC3 land in Milestone 12). Verified locally end-to-end.

**Zero-egress CI and housekeeping**
- [x] CI job runs `uv run --frozen --offline vdreamer doctor --offline`, asserting zero egress via an in-process socket guard (§5.2; the VM network policy stays the authoritative control per A40). Verified locally.
- [x] CI sets `UV_PYTHON_DOWNLOADS=never` so the dev toolchain never fetches interpreters in blocked-egress jobs.
- [x] `.gitignore`: `.venv/`, `docs/.PRD.md.un~` (vim undo artifact).
- [x] Record the build/test commands in `AGENTS.md` once tooling lands. (Done; also updated the repo-state section.)

**Exit:** ✅ `uv sync --locked --extra postgres` works on a clean checkout (verified); the plain-`pip` install job passes (verified locally; CI job is `.github/workflows/ci.yml:uv-independence`); locked/floor/latest all verified locally, CI runs on push.

---

## Milestone 1: Developer Contract Core (SDK primitives)

**Goal:** the `BaseAgent` developer contract (D1, D2, §2.2), the surface every agent author codes against.

**`BaseAgent` and build lifecycle**
- [x] `BaseAgent` attributes: `name`, `mode` (CHAT|HEADLESS|BOTH), `models`, `resources`, `input_schema`, `expose_thinking`, `access_roles`.
- [x] `build(ctx: BuildContext)` runs once per worker; the graph is cached. (`vdreamer.runtime.AgentRunner.build()`.)
- [x] `BuildContext`: `models`, `resources`, `tools`, `checkpointer`, `store`, `settings`, `agent_kwargs()`.
- [x] `agent_kwargs()` supplies `{checkpointer, store, context_schema=VDreamerContext, middleware=[VDreamerToolMiddleware()]}` for `create_agent` / `create_deep_agent`.
- [x] Optional hooks: `on_run_start`, `on_run_end`. (Fired by `runtime.AgentRunner` with `status="running"` / `"completed"|"failed"`; the worker maps interrupts onto `awaiting_approval` in M3.)
- [x] Accepted graph returns: `StateGraph`, compiled graphs, functional-API `@entrypoint`, `create_agent`, `create_deep_agent`. vdreamer compiles where needed (§2.4). (`validate.ensure_compiled`; the pass-through list is its docstring.)

**Runtime context and helpers**
- [x] `VDreamerContext` run context, with context-variable resolution so the implicit helpers (`emit_artifact`, `emit_citation`, `emit_progress`, `emit_log`, `require_approval`) work in any node or tool without parameters. (Helpers buffer into `ctx.pending`; the worker event pipeline that drains it lands in M2. Per-kind artifact data validation and MinIO writes land in M6.)

**Tools and policies**
- [x] `vdreamer.tool` decorator: `@tool` plus policy metadata (`redact`, `timeout_s`, `side_effects`, `approval`) stored in `tool.metadata["vdreamer"]` (A37). (`VDreamerToolMiddleware` is a pass-through; the 7-step governance pipeline and `ctx.tools.govern` land in M5, the approval gate in M8.)
- [x] `tool_policies` / `interrupt_policies` on `BaseAgent`, keyed by name or glob. (Enforcement of interrupt normalization for non-vdreamer sources lands in M8.)

**Discovery and persistence**
- [x] Agent discovery: `pkg.module:agent`, `--agents-dir`, or a `vdreamer.yaml` registry (D2). (`validate --registry` / `--agents-dir` landed; `serve`/`worker` flags land in M3.)
- [x] Persistence: `AsyncPostgresSaver` with `durability="sync"` by default and a per-agent `async` opt-in (A42, A6). (`persistence.open_checkpointer` factory + `durability_for`; worker queue/lease/resume wiring lands in M3.)

**Exit:** ✅ verified — `uv run pytest tests/contract` green (85 passed): a trivial in-repo agent (`tests/contract/test_milestone1_exit.py`) builds, compiles, and runs a single node through the checkpointer, fires the hooks, resolves helpers via the runner-bound contextvar, carries the §2.4 `agent_kwargs()` wiring, and passes every `validate_agent` check.

---

## Milestone 2: Protocol v1.0 & Event Pipeline

**Goal:** the normalized event envelope, its storage, and both stream drivers, all behind one adapter module (A31/A43).

**Storage and envelope**
- [x] `run_events(run_id, seq, type, ns, payload jsonb, ts)`, PK `(run_id, seq)` (A3); SSE `id: {run_id}:{seq}`. (`eventlog.RUN_EVENTS_DDL` + `PostgresEventLog.write` allocate `seq` from the table and stamp `envelope.sse_id`; the SSE framing itself is M4.)
- [x] Envelope: `{"v":"1.0", type, run_id, thread_id, agent, seq, ts, ns, payload}`. (`events.EventEnvelope` — field order is the contract.)

**Stream drivers.** One adapter module (A31), selected by `VDREAMER_STREAM_DRIVER`:
- [x] **v2 driver (default):** `astream(stream_mode=["messages","updates","custom"], subgraphs=True, version="v2")`. Pending interrupts are read from `aget_state(config).interrupts` after the stream ends. (`streaming.V2StreamDriver`, verified on langgraph 1.2.15.)
- [x] **v3 driver (opt-in, beta):** `astream_events(version="v3")`. Maps the `messages` projections (text, reasoning, tool calls, usage), lifecycle, subgraphs, and custom events. (`streaming.V3StreamDriver`; 1.2.15 fact: `data-delta`/`block-delta` message events carry no text, so they are skipped — recorded beta gap, gated by the golden suite.)

**Event mapping**
- [x] Event types: `run.status`, `token.delta`, `thinking.delta`, `message.completed`, `tool.call`, `tool.result`, `artifact.created`, `citation`, `interrupt.requested`, `interrupt.resolved`, `progress`, `log`. (`events.EventType`, all twelve.)
- [x] Thinking extraction (D18, A20) — `streaming.chunk_text_and_thinking` + `ThinkTagSplitter`; the splitter is a public, unit-tested primitive (per-alias wiring waits for `models.yaml`, M7), and `expose_thinking=false` suppresses deltas *and* the `message.completed.thinking` field:
  - standard `reasoning` blocks first
  - per-alias fallbacks: the `reasoning_content` field, then a stateful `<think>` splitter that handles split chunks
  - suppressed when `expose_thinking=false`
- [x] Custom data: vdreamer-enveloped custom data → artifact/citation/progress/log events; other custom data → `log`. `ns` is preserved for subgraphs. (`streaming._RunMapper.custom`; `ns` is `[]` at the root.)

**Delivery and testing**
- [ ] Delta batching (50 ms / 32 tokens), post-run compaction (A4), 15 s heartbeat, `X-Accel-Buffering: no` (A22). (Delta batching landed — `streaming.DeltaBatcher`, unit-tested and driver-wired — with the tuning constants `DELTA_BATCH_*`, `HEARTBEAT_INTERVAL_S`, `SSE_BUFFERING_HEADER`/`_VALUE`; the heartbeat + buffering header now fire on the M4 SSE transport (`gateway.sse`). Still open: post-run compaction (A4).)
- [x] Golden-event test suite: both drivers emit **identical** protocol events for the same graphs. This suite is the gate for promoting v3. (`tests/golden/`: all six scenarios at full v2↔v3 parity — no xfail — matched against golden type-sequence data recorded from the drivers plus the §3.3 payload-shape table applied to every event.)

**Exit:** ✅ verified — the golden-event suite passes for the v2 driver with v3 at full parity on every scenario (no xfail needed); v3 stays opt-in beta behind `VDREAMER_STREAM_DRIVER=v3` (langgraph 1.2.15 `LangChainBetaWarning`). `uv run pytest tests/golden` green (7 tests).

---

## Milestone 3: Runs, Queue, and Runtime Topology

**Goal:** the production run lifecycle, with Postgres as the only gateway↔worker transport (A7, D12).

**Entrypoints and queue**
- [x] Standard console-script entrypoints (`[project.scripts]`); in the repo, developers run them via `uv run` (`vdreamer = "vdreamer.cli:main"` wired since M0; `migrate`/`worker`/`dev` added in WP3):
  - [x] `vdreamer serve` (stateless gateway) — landed with Milestone 4.
  - [x] `vdreamer worker` — one process, one agent (`--agent` with `--registry`/`--agents-dir`, D2; `--claimant`, default `worker-{hostname}-{pid}`); DSN from `VDREAMER_PG_DSN` (A23), Postgres queue + `run_events` sink, startup recovery.
  - [x] `vdreamer dev` (single process on SQLite + local filesystem store, D11) — in-memory queue and sinks, `--input`/`--thread` smoke of the whole lifecycle, `--db` for the SQLite checkpoint file; hot reload is future work. The checkpointer is `AsyncSqliteSaver` (`langgraph-checkpoint-sqlite` is a base runtime dependency — no fallback, no warning), and the run's checkpoint persists across restarts: a fresh saver re-opened on the same `--db` file reads the final state back.
- [x] `vdreamer migrate` — idempotent schema application: `RUNS_DDL`, `RUN_EVENTS_DDL`, and the checkpointer's own `setup()`; DSN from `VDREAMER_PG_DSN` only (A23), friendly error without psycopg.
- [x] Queue tables: claim via `FOR UPDATE SKIP LOCKED`, with a 10 s lease heartbeat. (WP1: `RUNS_DDL`, `PostgresRunQueue` + `InMemoryRunQueue`, heartbeat at half the lease window.)
- [x] Run states: `queued → running → (awaiting_approval → running)* → completed | failed | rejected | expired | cancelled`. (WP1: `RunState` + the transition map, enforced per-UPDATE.)

**Cancel, shutdown, recovery**
- [x] Cancel: `POST /runs/{id}/cancel` calls `RunControl.request_drain()`, which leaves a resumable checkpoint; the run moves to `cancelled`. (WP2's worker mechanism + the M4 route: the gateway marks the row; the owning worker's heartbeat notices within half a lease window, drains with reason `cancelled`, and authors the final event. A `queued`/`awaiting_approval` run has no worker, so the gateway writes it.)
- [x] Worker SIGTERM: drain and requeue without consuming an attempt. (WP2: `release` gives back the claim's attempt increment; POSIX handler install, portable `drain_all`.)
- [x] Crash recovery: an expired lease requeues the run with `attempt+1` (max 3) and resumes from the checkpoint. The in-flight node re-executes, so side effects use `ctx.idempotency_key` (A6). (WP1's `requeue_expired` sweep + WP2's `recover_on_start`/lease-lost drain. The re-claim re-drives the row's input on the same checkpointed thread — the checkpointed state carries, and the ledger keyed on `ctx.idempotency_key` — the run id itself, stable across attempts (A6) — keeps the side effect exactly-once; planting a literal resume input on requeued rows is the M4 mechanism, same as approval resumes.)
- [x] Node timeouts and errors (`NodeTimeoutError`, `NodeError`) map to `run.status.error`. (WP2.)

**Fan-out and concurrency**
- [x] `LISTEN/NOTIFY {run_id, seq}` fan-out. Any gateway replica serves any SSE reconnect: replay `seq > Last-Event-ID`, then tail live. (M4: `PostgresEventLog.write` NOTIFYs in the insert's transaction; `PostgresEventListener` feeds an `EventHub`; the SSE loop re-reads the log on every wake-up and every heartbeat, so a lost notification costs latency, never an event.)
- [x] One active run per thread: `409 {active_run_id, reason}` (D12, A11). `?wait_for_idle=true` queues behind `running` only. (The queue guard + `wait_for_idle` landed in WP1, including the partial index behind it; the 409 render is the M4 gateway.)

**Exit:** ✅ verified — kill a worker mid-run; the run resumes from its checkpoint and the side effect runs exactly once (`tests/worker/test_recovery_idempotency.py`: worker 1 killed mid-run leaves the row claimed with a lapsing lease; the recovery sweep requeues; worker 2 — a fresh runner over the same checkpoint store — re-executes the side-effecting node, whose ledger keeps the effect exactly-once while the node itself runs twice; the run completes on attempt 2 with the pre-kill checkpointed state intact).

---

## Milestone 4: REST Gateway (`/api/v1`)

**Goal:** the full REST surface, with uniform errors and ops endpoints.

**Routes** (`vdreamer.gateway`: `routes_runs`, `routes_interrupts`, `routes_artifacts`, `routes_ops`)
- [x] `GET /agents`, `GET /agents/{name}` (mode, input JSON Schema). Role-gated agents are hidden from the list and `403 forbidden_role` on direct access.
- [x] Threads: create (`201`), list (owner-scoped), get (history from checkpoint), soft delete. (`threads.ThreadStore` + `THREADS_DDL`; someone else's thread is a `404`, never a `403`; history is the latest checkpoint's state channels; deleting a thread with an active run is `409 run_conflict`.)
- [x] `POST /threads/{id}/runs` → `202 {run_id}`; `GET /runs/{id}`; `POST /runs/{id}/cancel`. (Input validated against `input_schema` → `422 invalid_input`; reserved `config.configurable` keys refused; the initiator's `user_id`/`roles` and the config are stored on the row and bound into the worker's `VDreamerContext`; headless agents → `400 mode_not_supported`, D15.)
- [x] SSE with `Last-Event-ID` replay: `GET /runs/{id}/events`, `GET /threads/{id}/events` (joins the active or latest run). (`gateway.sse`: `id: {run_id}:{seq}`, 15 s `: heartbeat` frames, `X-Accel-Buffering: no`; closes on a terminal `run.status`, stays open across `awaiting_approval`.)
- [x] Interrupts: `GET /interrupts`, `GET /interrupts/{id}`, `POST /interrupts/{id}/resolve` (§4). (`interrupts.InterruptStore` + `INTERRUPTS_DDL`; the worker writes the §4.2 record on `interrupt.requested`; resolve checks visibility `404` → authority `403` → expiry `410` → version `409` (A12) → actions `422`, CAS-resolves, emits `interrupt.resolved`, plants the resume map and requeues; the worker turns it into `Command(resume=…)`. Milestone 8 adds `edit_schema` validation, four-eyes, the expiry sweeper, headless `reject → rejected`, and `interrupt_policies`.)
- [x] Artifacts: `/artifacts/{id}`, `/preview`, `/raw`, `/download` (§3.4). (Routes, access rules — thread owner or `artifact:read:all` — `410 artifact_expired`, the HTML CSP, and the `artifacts.ArtifactStore` contract. `serve` wires `UnavailableArtifactStore` until the M6 MinIO writer; paged/sorted table previews, CSV/XLSX conversion and `artifact://` rewriting are M6.)
- [x] Ops: `GET /protocol/schema.json`, `/openapi.json`, `/healthz`, `/readyz`, `/metrics`. (`protocol_schema.event_json_schema` is generated from the payload models; `/readyz` runs every backend probe — Postgres and the `LISTEN` connection under `serve` — with a 2 s budget.)
- [x] `GET /api/v1/docs` — Swagger UI served from vendored assets at `/static/vendor/swagger-ui/` (no CDN; §5.2/A13), pointing at `/api/v1/openapi.json`. (Verified rendering in a browser: zero console errors, every request same-origin.)
- [x] Vendored swagger-ui assets ship in the repo and are covered by the zero-egress checks. (`swagger-ui-dist` 5.33.1, sha256-pinned in `vdreamer.vendored`; `doctor --offline` fails on a missing/modified file or an external URL in the docs shell; `.gitattributes` keeps line endings from drifting the hashes.)
- [x] The gateway is built on FastAPI (D20), pinned via `uv add fastapi` and subject to the floor/latest CI pins; the framework stays internal — vdreamer's contract is the REST/SSE protocol and OpenAPI 3.1 artifacts. (`import vdreamer` never imports FastAPI — asserted by a test. The `smoke` set now includes four API tests; verified locally on the floor pins, langchain 1.4.0 / langgraph 1.2.11.)
- [x] FastAPI's default docs CDN is overridden: Swagger UI is served from vendored assets only (zero egress, §5.2), consistent with the vendored-swagger-ui items above. (`docs_url`/`redoc_url` disabled; the docs page runs under `script-src 'self'` and `connect-src 'self'`, with `validatorUrl: null`.)

**Errors and auth**
- [x] Uniform error body `{error: {code, message, details?}}` with stable codes (`run_conflict`, `forbidden_role`, `artifact_expired`, …). (`gateway.errors.ErrorCode`; FastAPI validation, routing 404/405 and unhandled 500s all use it; 500s never echo internals.)
- [x] Internal auth (D13, A14) — `gateway.auth.InternalAuthMiddleware`, pure ASGI so it runs before body parsing:
  - `Bearer <internal_token>` + `X-User-Id` / `X-User-Roles`
  - two tokens valid at once, for rotation (`VDREAMER_INTERNAL_TOKEN` + `VDREAMER_INTERNAL_TOKEN_PREVIOUS`, constant-time compare)
  - role gating via `access_roles`
  - the gateway binds to the internal interface only (`--host` refuses `0.0.0.0`/`::`)
  - (Upstream service tokens for `/trigger`, A16: landed with Milestone 9, `VDREAMER_SERVICE_TOKENS`.)

**Metrics**
- [x] Prometheus: queue depth, run latency, SSE connections, model latency/tokens per alias, tool latency/errors per source, MCP health, sandbox timeouts, outbox lag. (`vdreamer.metrics`: every family is defined and exposed; queue depth, run latency (worker side, `vdreamer worker --metrics-port`) and SSE connections are recorded now, the rest by their subsystems in M5/M6/M7/M9.)

**Earlier-milestone defects fixed here** (found while wiring the gateway)
- `PostgresEventLog.write` passed a raw `dict`/`list` to psycopg, which cannot adapt a `dict` (and sends a `list` as an array, not `jsonb`), so every real-Postgres event write would have failed. Now JSON text with `::jsonb` casts, plus `NOTIFY` and a retry on a raced `seq`.
- `PostgresRunQueue.enqueue` read the thread id as the status, so `wait_for_idle` could never succeed on Postgres; its `SKIP LOCKED` check could also skip a concurrently locked active row. A per-thread advisory lock now serializes enqueues.
- `claim()` had no agent filter: with several agents on one database, any worker claimed any agent's run.
- Recovered/released runs re-drove their input from the start. The worker now resumes from the run's own checkpoint (`RUN_ID_CONFIGURABLE_KEY` in the checkpoint metadata), so completed nodes no longer re-run; the M3 exit test now pins that.

**SQLite testing phase** (decided after M4: test on SQLite, migrate to Postgres after heavy developer testing)
- [x] `vdreamer.sqlite_state`: SQLite run queue, event log, thread and interrupt stores. Runs/threads/interrupts write through their in-memory twins (one copy of the semantics); the event log is read straight from SQLite. `vdreamer dev` keeps everything in its `--db` file, next to the checkpoints, and a run left `running` by a dead process is recovered by the boot sweep.
- [x] `tests/api` runs every stack-based contract test against both the in-memory and the SQLite stores; `tests/test_sqlite_state.py` covers restart survival. Verified in a real process: a completed run, its thread values, and its SSE replay survive a `vdreamer dev` restart.
- [ ] Postgres migration, after developer testing: run the live-PG tests (`VDREAMER_TEST_PG_DSN`) and switch deployments to `serve` + `worker`.

**Exit:** ✅ verified — `uv run pytest tests/api` green (98 passed; 5 live-Postgres tests skip without `VDREAMER_TEST_PG_DSN`) and covers every route, including auth failures and every error code; `/api/v1/openapi.json` is valid OpenAPI 3.1 (`openapi-spec-validator`) and lists exactly the §2.5 route table; `/api/v1/docs` renders from vendored assets, and its served HTML is asserted to carry the CSP and no external URL. Full suite: 377 passed, 10 skipped; ruff + mypy (strict) clean; `doctor --offline` green. Still owed: a live-Postgres run of the new stores — deferred to the Postgres migration (see the SQLite testing phase above).

---

## Milestone 5: Tool Governance & MCP

**Goal:** the governed tool pipeline and the MCP lifecycle (A37–A40).

**Governed tool pipeline** (`vdreamer.governance`)
- [x] `ctx.tools.govern(tools)` proxy, plus `VDreamerToolMiddleware` (`@wrap_tool_call`) for `create_agent`. Tool name and schema are preserved (A37). (`GovernedTool` delegates name, description, schema, and `ToolNode` injection facts to the wrapped tool; the middleware carries the agent's `tool_policies` via `ctx.agent_kwargs()`; an already-governed tool is never governed twice. `BuildContext.tools` is the `ToolKit`.)
- [x] Governance steps, in order (one `_Pipeline`, async + sync paths):
  1. Authorize (scope/roles). (`requires_roles`: any one; `scope`: granted by a role equal to it or an `fnmatch` pattern over it, e.g. `db:*`.)
  2. Emit `tool.call`, redacting policy keys plus the defaults password/token/secret/api_key/authorization (A33). (Injected args — `runtime`, state, store — never reach the event.)
  3. Optional approval gate. (`Policy.approval` → `require_approval` with `approve`/`reject`; a rejection is contained as `denied`. A resumed attempt re-emits its `tool.call` with the same `tool_call_id`. Four-eyes/expiry stay M8.)
  4. Execute under `timeout_s` (default 60), with `ctx.idempotency_key`. (`side_effects=True` → `{run key}:{tool_call_id}`, in both `current_context()` and `runtime.context`; stable across recovery because the call id is checkpointed.)
  5. Normalize: a result over `max_result_chars` (8,000) becomes a `json`/`file` artifact; the model gets a truncated body plus the `artifact_id`. (MCP structured content and base64 media blocks map the same way. Artifacts go through the new write side — `artifacts.write_artifact`, D7 order — into the store the worker binds per run.)
  6. Contain errors (`tool_error|timeout|denied|transport|invalid_args`). The run continues unless `fail_run=true`. (`invalid_args` messages never echo values; `fail_run` raises `GovernedToolError`.)
  7. Emit `tool.result` with `source` and `duration_ms`. (`source` rides the matching `tool.call`: `custom|langchain|mcp:{server}`.)
- [x] `ToolRuntime[VDreamerContext]` parameter access: run, thread, user, roles, idempotency_key, state, stream_writer, tool_call_id.
- [x] Governed tools are the single emitter of `tool.call`/`tool.result` (§2.4): the stream mapper turns their records into events and drops its own mapping for those call ids. An ungoverned call still maps with default redaction, but its `tool.call` is now held until the call runs (just before `tool.result`, or at run end) — the golden `tool_roundtrip` order changed accordingly.

**MCP.** Isolated in one module (`vdreamer.mcp`); `langchain.mcp` is version-pinned (A38/A39).
- [x] `resources.yaml` entries: `type: mcp`, `url` → Streamable HTTP or `command` → stdio, `bearer_env`, `startup: required|optional`, `refresh_s`, `include` globs, `identity`. (`vdreamer.resources` loads the file — `--resources` or `$VDREAMER_RESOURCES` on `dev`/`worker`/`validate` — and refuses inline secrets (A23); `identity: forward` is refused as P1; other resource types are M7's.)
- [x] One `MCPAdapter` per server at worker boot. `list_tools()` is cached and refreshed every `refresh_s`. Tools are named `{server}_{tool}` and selected with `ctx.tools.mcp(server, include=[...])`.
- [x] Each run records its tool-snapshot hash. (`vdreamer_tool_snapshot` in `configurable`, so it lands in the run's checkpoint metadata; the runner rebuilds the graph when the snapshot changes — "changes apply to new runs".)
- [x] Failures: a down `required` server fails `/readyz`; an `optional` one is excluded and logged. (Both retry on `refresh_s`; `vdreamer_mcp_server_up` per server. `/readyz` sees MCP under `vdreamer dev`; a split `worker` process has no readiness endpoint yet.)
- [x] Elicitation arrives as a LangGraph interrupt and is normalized (§4.5). (`langchain.mcp`'s interrupt → kind `mcp_elicitation`, actions `edit`/`reject`; the gateway builds `{"responses": {key: {action, content?}}}` — `edit` answers every request, `reject` declines them. Verified through the REST surface with the interrupt value `langchain.mcp` raises; a live elicitation round against a real server still needs a modern-protocol server fixture.)
- [x] Network rules (A40):
  - `NETWORK_ALLOWLIST` is enforced at config load **and** at connect. (Host globs and IPs/CIDRs; at connect a name off the host list must resolve only into allowed networks.)
  - stdio `command` must be an absolute path to a preinstalled binary.
  - Launchers that fetch packages at runtime (`npx`, `uvx`, `pip`) are rejected. (Also `uv`, `pipx`, `npm`, `pnpm`, `yarn`, `bun(x)`, and `python -m pip|uv|pipx`.)
  - stdio servers run under rlimits with only their declared env. (POSIX: `_mcp_launch.py` sets the rlimits and `execve`s with only the declared keys, since the MCP SDK merges a default env into every child. Windows dev hosts get neither — exercised on Linux CI, D19.)
- [x] `validate` checks: no ungoverned tools; approval routes wired. (`tools.governed`: every `ToolNode`, subgraphs included, holds only proxies or runs under the middleware; `approval.routes`: a headless agent's tool approval needs an `approver_role`, A16. `ApprovalGate` route wiring is M8.)

**Exit:** ✅ verified — a governed tool run shows a redacted `tool.call` → `tool.result` with an overflow artifact (`tests/tools/test_governance.py`, both stream drivers; `tests/api/test_tool_governance_api.py` serves the artifact bytes through the gateway on both store backends). MCP stdio and HTTP (bearer) servers are both discoverable against a real FastMCP fixture server, and a failed optional server is excluded while a failed required one fails readiness (`tests/tools/test_mcp.py`). Verified in a real `vdreamer dev --resources` process: `/readyz` probes the required server, the optional one is excluded, and the overflow artifacts are served from `artifacts/{thread}/{run}/` next to the `.db`. Full suite: 522 passed, 11 skipped; ruff + mypy (strict) clean; `doctor --offline` green (it now imports the MCP stack under the egress guard and forces `FASTMCP_CHECK_FOR_UPDATES=off`). The smoke set (now incl. the exit and MCP runs) passes on the floor pins (langchain 1.4.0, langgraph 1.2.11).

---

## Milestone 6: Artifacts, Files & Sandbox

**Goal:** the artifact system (D6–D8), file inputs, safe extraction, and the sandbox (§3.4–3.5).

**Artifact core** (`vdreamer.artifacts`, `vdreamer.artifact_data`, `vdreamer.objects`)
- [x] `emit_artifact(name, kind, data, metadata?, renderer?)` and the `Artifact` return type. (Plus `aemit_artifact`, the awaitable twin, and keyword-only `mime_type`/`filename` for `file`. `emit_artifact` works in any node or tool: on a sync node's executor thread it hands the write to the run's loop; called from `async def` code it blocks that loop on a helper thread, so async code should prefer `aemit_artifact`. `return Artifact(...)` (or a list, or one as `ToolMessage.artifact` with `response_format="content_and_artifact"`) from a governed tool is stored and the model gets a reference naming the `artifact_id`; it is a LangChain `ToolOutputMixin`, and on the middleware path an `ArtifactReturnBridge` carries it past `ToolNode`'s return-type check.)
- [x] 5 kinds: `table`, `image`, `html`, `json`, `file`. **No auto-detection**: unsupported data → `TypeError`. (Images must be PNG/JPEG/SVG by magic bytes; `json` takes a Pydantic model or a dict, over 1 MB is a `ValueError`; a `str` is a path for `image`/`file` and content for `html`.)
- [x] Write order (D7): bytes → MinIO `artifacts/{thread_id}/{run_id}/{artifact_id}.{ext}` → metadata row → `artifact.created` event. (`ArtifactStore` pairs a metadata table with an `ObjectStore`: in-memory for tests, SQLite rows + local filesystem for `vdreamer dev` (D11), Postgres `artifacts` table + MinIO for `serve`/`worker`. The worker binds its store per run; with no store bound the event still flows, without URLs.)
- [x] Previews per kind: schema + 10 rows + row_count / dimensions / title + summary / ≤64 KB body / filename + mime. (`json` over 64 KB previews as `{"truncated": true}`; the HTML summary is its `<meta name="description">`.)
- [x] `artifact.created` payload carries the full field set, including `question_id`, `session_id`, `view_url`, `download_url`.

**Tables** (`vdreamer.tables`; pyarrow and xlsxwriter come with the `[sandbox]` extra)
- [x] Incremental Parquet row groups. (65,536 rows per group; a row iterator is never held in memory whole.)
- [x] `/preview?offset&limit&sort` (limit ≤1,000). (`sort=col,-col`, nulls last; unsorted pages read only the row groups they span; an unknown column → `422 invalid_request`.)
- [x] `/download?format=csv|xlsx|parquet` with streaming conversion. XLSX beyond 1,048,575 rows → `422`. (CSV streams a row group at a time; XLSX is built in constant-memory mode in a temp file, then streamed; a format on a non-table → `422 unsupported_format`.)
- [x] `TABLE_MAX_ROWS` 5M, with `metadata.truncated`. (`VDREAMER_TABLE_MAX_ROWS` overrides; every table carries `truncated: true|false`.)

**HTML, access, retention**
- [x] HTML: `artifact://` links are rewritten to `/raw`. CSP allows no external origins, with `connect-src 'none'`. Chart JS is vendored at `/static/vendor/` (A13). (Rewritten at serve time. plotly.js 4.1.1 from the `plotly` 7.1.0 wheel, at `/static/vendor/plotly/plotly.min.js`, sha256-pinned and checked by `doctor --offline`. SVG images get the same CSP; every `/raw` sends `X-Content-Type-Options: nosniff`.)
- [x] Access: thread owner, granted approver, or `artifact:read:all` (A8). Retention is 30 days; a missing object returns `410` (A9). ("Granted approver": a caller holding the `approver_role` of one of the artifact's run interrupts, pending or resolved. Retention: `migrate` installs a MinIO lifecycle rule on `artifacts/` and `extracted/`; `vdreamer dev` sweeps its filesystem store at boot.)

**Files** (`vdreamer.files`, `vdreamer._extract`)
- [x] `FileRef` + `ctx.files.open(ref)`; the gateway validates the `inputs/` prefix (A17). (`FileRef` refuses any other prefix and any key that could climb out of it, so input validation answers `422 invalid_input`. `ctx.files` — on the run context, next to `ctx.sandbox` — reads `inputs/` and the run's own `extracted/` keys only: `open`, `read_bytes`, and their async twins.)
- [x] `ctx.files.safe_extract(ref, allow=[...]) -> Manifest` (A34, security-critical):
  - caps on size, entry count, compression ratio; nesting ≤2 (counted while inflating, never from headers; defaults 256 MB per file, 1 GB total, 10,000 entries, 100:1 above 1 MB; `ExtractLimits` overrides)
  - rejects traversal, absolute paths, symlinks, encryption (the whole archive, `UnsafeArchiveError` with a stable `code`; also control characters, corrupt or unsupported members)
  - name decoding UTF-8 → CP437 → CP1256 (CP437 wins only when every high byte is one of its accented letters)
  - manifest with sha256, plus a listing of skipped types (`allow` takes filename globs and mime patterns; kept files land at `extracted/{thread_id}/{run_id}/…`; archive paths never touch the disk, outputs are numbered files)
  - runs in the sandbox: `python -I -m vdreamer._extract` under the sandbox launcher (rlimits, uid, no network); Windows dev hosts run the same child without the launcher, since the caps are enforced in its own code.

**Sandbox** (`vdreamer.sandbox`, `vdreamer._sandbox_launch`)
- [x] `ctx.sandbox.run(code, inputs, timeout_s=60)` (D19/A19, Unix-only): dedicated uid, rlimits (CPU, 2 GB memory, file size, nproc), wiped scratch dir, empty env, network off where possible. Files written to `./out/` become artifacts. (A fresh single-threaded launcher applies the limits and `execve`s — no `preexec_fn` in the threaded worker. `VDREAMER_SANDBOX_UID` names the uid (the worker must be root); without it the worker's own uid is used and `RLIMIT_NPROC` is skipped (it counts the whole uid). Network: a new network namespace as root, else an unprivileged user+network namespace. The process group, and with a dedicated uid every process of that uid, is killed at the timeout. Outputs map by extension (PNG/JPEG/SVG → `image`, HTML → `html`, JSON objects → `json`, Parquet/CSV → `table`, else `file`); symlinks, hard links and special files are skipped. `arun` is the async twin; Windows raises `SandboxUnavailable`.)
- [x] The sandbox interpreter is the worker environment's Python. pandas, pyarrow, matplotlib, plotly, and numpy come from the `sandbox` extra. (`sys.executable -I -B`; the env holds only sandbox-local `HOME`/`TMPDIR`/`MPLCONFIGDIR`, `MPLBACKEND=Agg` and single-threaded BLAS.)

**Exit:** ✅ verified.
- The sandbox runs untrusted code with its rlimits verified (`tests/artifacts/test_sandbox.py`, run in a `python:3.11-slim` Linux container as root with dedicated uid 4242, and again as an unprivileged worker). Verified:
  - the CPU, address-space, file-size, open-file, core and nproc limits are applied;
  - a 3 GB allocation dies with `MemoryError`, and an oversized write is stopped;
  - a spinning child is killed at the timeout;
  - the env is empty, and the network namespace leaves no route out;
  - the sandbox uid cannot read the worker's files;
  - `out/` symlinks and hard links are skipped.
- `safe_extract` defeats the traversal/zip-bomb corpus (`tests/artifacts/test_safe_extract_corpus.py`, 25 cases: traversal, absolute paths, symlinks, encryption, a 1000:1 bomb, lying size headers, an overlapping-entry bomb, too many entries, a quine-style nest, corrupt members), plus `FileRef` → `safe_extract` through the REST surface (`tests/api/test_files_api.py`).
- Artifacts render end-to-end through the gateway proxy (`tests/api/test_artifacts_e2e.py`, both store backends: every kind's preview and field set, paged/sorted previews, CSV/XLSX/Parquet, the XLSX `422`, HTML rewriting under the CSP with the vendored chart JS, granted-approver access, `410` after the object is gone).
- MinIO: `MinioObjectStore` round-trip and the 30-day lifecycle rule ran live against an S3 server (`tests/artifacts/test_object_stores.py`; moto's S3 server in the container, since MinIO no longer publishes a public image) — not yet against MinIO itself.
- Checked in a real `vdreamer dev` process: a bad `FileRef` → `422`; extract → manifest; a sorted page of a 5,000-row table; CSV export; HTML rewritten; the bytes under `artifacts/`, `extracted/`, `inputs/` next to the `.db`.
- Full suite: 615 passed, 20 skipped; ruff and mypy (strict) clean; `doctor --offline` green (it now imports the artifact stack under the egress guard, pins the chart JS, and probes the sandbox). The smoke set (now including the D7-order and artifact-return runs) and the M6 suites pass on the floor pins.

---

## Milestone 7: Models & Resources

**Goal:** `models.yaml` and `resources.yaml` resolution (§3.6–§3.7, A35, A36, A21).

- [x] vLLM/Ollama aliases become LangChain chat models with `output_version="v1"`, each with a per-alias thinking mode (`reasoning_content`, `think_tags`). (`vdreamer.models`: `VllmChatModel` is a `ChatOpenAI` on Chat Completions that keeps vLLM's `reasoning_content`/`reasoning` field, which `ChatOpenAI` drops; `OllamaChatModel` is a `ChatOllama` (`reasoning_content` uses Ollama's own `think`). Both parse a leading `<think>…</think>` under `think_tags`, even split across chunks. The thinking mode is applied in the model layer, so streamed chunks, the final message and both stream drivers all see standard `reasoning` blocks, and reasoning is never sent back in the history. Mode `none` (the default) extracts nothing. `modalities`, `max_tokens` and `context_window` fill the model's `profile`. Every chat alias records `vdreamer_model_latency_seconds` and `vdreamer_model_tokens` per alias (§5.4). Building a model makes no request.)
- [x] fastembed aliases (dense/sparse/rerank) load once per worker from `cache_dir` as `TextEmbedding` / `SparseTextEmbedding` / `TextCrossEncoder`, with `local_files_only` and CPUExecutionProvider (A36). (`local_files_only: false` is refused, `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are forced before fastembed loads, and the objects are the native ones, unwrapped. `validate`/`doctor` locate the files with `lazy_load=True` and don't load the ONNX session.)
- [x] `resources.yaml`: qdrant/neo4j/sql adapters (`vdreamer.adapters`).
  - Pooled async clients are opened at worker start and injected, native and unwrapped, via `ctx.resources[alias]` (A35): `AsyncQdrantClient` (also qdrant's embedded `path`/`location: ":memory:"` for dev), neo4j `AsyncDriver`, SQLAlchemy `AsyncEngine` (the new `[sql]` extra). `ctx.models` / `ctx.resources` hold the agent's declared aliases only; any other alias raises a `KeyError` that says so.
  - `read_only` is enforced:
    - **sql:** every pooled connection goes read-only as it opens (PostgreSQL, MySQL/MariaDB, SQLite), committed so the pool's rollback doesn't undo it. Other dialects refuse `read_only`, so read-only credentials stay the primary control.
    - **neo4j:** a guard opens `READ_ACCESS` sessions and routes `execute_query` to readers, and the server rejects writes. `execute_write` raises.
    - **qdrant:** a guard lets the read methods through and raises on the rest.
  - Adapter class shape: `connect` / `probe` / `close`. A new type is one `ResourceAdapter` subclass plus `register_adapter`; its `__init__` validates the entry at load. An unknown type fails at load.
- [x] Secrets are referenced only through `*_env` keys and never written into YAML (A23). (`api_key_env`, `user_env`/`password_env`, `url_env`; inline secret keys and credentials inside a URL are refused at load. A missing variable fails at boot and in `validate`. Every declared endpoint must match `NETWORK_ALLOWLIST` at load and again at connect (A40): `NetworkAllowlist` moved to `vdreamer.network`, shared with MCP.)
- [x] `validate` / `doctor` fail on unknown aliases, unreachable endpoints, or missing model files.
  - `validate --models/--resources`: one check per declared alias (`model:<alias>`, `resource:<alias>`). A chat endpoint must answer and serve the configured model, fastembed files must be in the cache, and a resource must connect and pass its probe. `--no-probe` stays off the network (files and `*_env` variables are still checked). `build()` gets the real chat models, the located fastembed objects, and an unconnected placeholder per resource.
  - `doctor --offline --models/--resources` probes every alias, resource and MCP server after the zero-egress assertion.
  - `worker`/`dev` take `--models` (or `$VDREAMER_MODELS`). At boot they build the models, load fastembed once, probe the endpoints, and open the resources (`vdreamer.backends`). Any failure stops the process before it claims a run.

**Exit:** ✅ verified (`tests/models/test_m7_exit.py`). An agent declaring `models = ["reasoner", "local", "dense"]` and `resources = ["warehouse", "vectors"]`:
- passes `vdreamer validate` (in process, and the real CLI in a subprocess together with `doctor --offline`);
- runs against stub backends: an OpenAI-compatible vLLM stub and an Ollama stub over real HTTP, read-only SQLite, embedded qdrant, and a recorder in place of the fastembed class. The run goes through `open_agent_backends` + `AgentRunner` and through `vdreamer dev --input`; thinking reaches `thinking.delta` from both chat aliases;
- fails `validate` on an unknown alias, a stopped endpoint and an empty fastembed cache.

Also verified:
- **Stream drivers:** the thinking modes give identical `thinking.delta`/`token.delta`/`message.completed` events on v2 and v3 (`test_thinking_events.py`).
- **Live services:** the adapters ran against qdrant, Neo4j 5.26 (writes rejected by the server in READ sessions) and PostgreSQL 17 (read-only on every pooled connection), from a Linux container (155 passed).
- **Floor pins:** the M7 suites pass (langchain-openai 1.1.0, langchain-ollama 1.0.0, SQLAlchemy 2.0.40).
- **Not live yet:** real fastembed model loading runs only with `VDREAMER_TEST_FASTEMBED_CACHE` (no model files ship with the repo).
- **Fixes on the way:** M2's `message.completed` crashed on real `usage_metadata`, a plain dict that the M2 code called `.model_dump()` on; it now serializes as a dict. An M5 MCP test used a Windows-only absolute path.

---

## Milestone 8: HITL Protocol

**Goal:** the full interrupt/approval pipeline, with all four interrupt sources normalized (§4, A44).

**Approval primitives**
- [x] `require_approval(kind, payload, actions, approver_role?, scope?, edit_schema?, ttl_hours?)` wraps `interrupt()` (A15). The policy is serialized and enforced by the gateway. (Refuses unknown/empty action sets and a non-positive TTL; a headless run without `approver_role` raises `ValueError` (A16, `VDreamerContext.headless`); an `edit` payload is re-validated against `edit_schema` in the node and returned as its JSON dump.)
- [x] `ApprovalGate` subgraph: `prepare → gate → {approve|edit: execute, reject: on_reject, feedback: replan}`. `validate` flags unwired routes. (`vdreamer.approval`, exported as `vdreamer.ApprovalGate`: the gate routes with a `Command`, so state needs no routing key; `edit` writes the payload's keys into state (or `apply_edit`) before `execute`; `replan` reads the text from `state[decision_key]` and loops back through `prepare`. `build()` returns the `StateGraph`, `compile()` nests it. The gate node carries JSON-safe metadata that `validate` finds in subgraphs too: an offered action with no node is unwired, except a `HEADLESS` agent's `reject`.)
- [x] `interrupts` table record (status, version, expires_at, resolved_by, resolution). Resume restarts the node from the top. (M4's record; `expire_overdue` added to every store — in-memory, SQLite, Postgres with `FOR UPDATE SKIP LOCKED`.)

**Resolve endpoint**
- [x] Body `{action, payload?, text?, expected_version}`. Multi-action interrupts use `{decisions: [...], expected_version}`, in order. (For `tool_review` one decision per action request, each checked against its own `allowed_decisions`; a single-call review also takes the `action` form.)
- [x] Responses: `200` | `403 forbidden_role` | `409` stale version | `410` expired | `422` invalid edit. (`edit` payloads are validated against the record's `edit_schema` with `jsonschema` (now a base dependency; Draft 2020-12, as pydantic emits) → `422 invalid_edit` with `details.errors[{loc, msg}]`; an unusable schema refuses the edit rather than skipping it.)
- [x] Action semantics:
  - `edit`: validated against `edit_schema` and written to state before execution (gateway + node; `ApprovalGate` writes it)
  - `feedback`: routes to replan
  - `reject` in headless runs: run → `rejected` (the gateway ends the run straight from `awaiting_approval` — a new §3.2 transition — writes the final `run.status`, cancels sibling interrupts; nothing resumes. Chat `reject` resumes and routes to `on_reject`. Tool-review rejects and elicitation declines always resume: those are the middleware's/server's own decisions.)
  - `respond`: returns text to the tool

**Authority and expiry**
- [x] Authority: the initiator by default; `approver_role` overrides. `HITL_STRICT_SCOPES` enforces four-eyes (approver ≠ initiator, A32). (Comma-separated `fnmatch` patterns from the env (A23), on `GatewaySettings.strict_scopes`; the initiator gets `403 forbidden_role` with `details {reason: "four_eyes", scope}` even when holding the role.)
- [x] Expiry: a leader-elected sweeper marks the interrupt `expired`, the run `expired`, and unlocks the thread. (`interrupts.InterruptSweeper`, every 30 s in the gateway process: `vdreamer serve` elects with a session-level `pg_try_advisory_lock` (`PostgresAdvisoryLeader`; a dropped connection hands leadership over), `vdreamer dev` is always the leader. The run's final `run.status` is `expired` with `error.code = "interrupt_expired"`. The resolve route already answered `410` lazily.)
- [x] `awaiting_approval` frees the worker; resolution enqueues a resume. (Since M4.)

**Interrupt-source normalization (A44).** One record, event, and endpoint for all four:
- [x] `require_approval`
- [x] HITL middleware `interrupt_on` → `tool_review` (was normalized as `generic` before M8: `normalize_interrupt` now recognizes the `HITLRequest` shape. The payload keeps `action_requests[]` (args redacted with the A33 defaults) and `review_configs[]`; actions = the union of allowed decisions. The gateway resumes with the middleware's own `{"decisions": [{"type": "approve"} | {"type": "edit", "edited_action": {name, args}} | {"type": "reject", "message"?} | {"type": "respond", "message"}]}`, edits validated against the tool's `args_schema`. Verified on both stream drivers.)
- [x] MCP elicitation → `mcp_elicitation` (since M5)
- [x] raw `interrupt()` → `generic` (the resume is now the resolver payload verbatim — `payload` if given, else `text` — per §4.5, instead of `{action, text}`)
- [x] Policies for non-vdreamer sources attach via `interrupt_policies`. (`interrupts.apply_interrupt_policy`, applied by the worker's sink before the record and the event exist: tool-name keys (exact, then `fnmatch` globs) beat the kind key per field; `ttl_hours` restarts the clock. `validate` now requires a headless agent's `HumanInTheLoopMiddleware` reviews to get an `approver_role` from `interrupt_policies["tool_review"]`, and headless `ApprovalGate`s to name one (A16).)

**Exit:** ✅ verified — all four sources resolve through one endpoint, end to end through the REST surface with live workers on both store backends (`tests/api/test_hitl_m8.py::test_all_four_sources_resolve_through_one_endpoint`, smoke): an `ApprovalGate` (`require_approval`), `create_agent` + `HumanInTheLoopMiddleware` (`tool_review`), an MCP elicitation, and a raw `interrupt()`. The same suite covers four-eyes (initiator refused, another `dba` approves), the version race (the second resolver gets `409`; three concurrent CAS attempts, one winner), `edit` → `422` then a valid edit written to state, `feedback → replan` re-asking, chat `reject → on_reject`, headless `reject → rejected`, `interrupt_policies` on a tool review, and the sweeper expiring an interrupt and its run (thread unlocked, `410` after). Unit tests in `tests/hitl/`. Full suite: 736 passed, 24 skipped; ruff + mypy (strict) clean; `doctor --offline` green; the M8 suites pass on the floor pins (langchain 1.4.0). Checked in a real `vdreamer dev` process: an `ApprovalGate` run → `422` on a bad edit → valid edit → `completed` with the edited state. Not run live: `PostgresAdvisoryLeader` and the Postgres `expire_overdue`/`finish` SQL (the SQLite-first testing phase — they wait for the Postgres migration with the other live-PG tests).

---

## Milestone 9: Headless Trigger, Outbox & Callbacks

**Goal:** the `/trigger` path and notification plumbing (D15, A26, D10).

- [x] `POST /agents/{name}/trigger`: (`gateway/routes_trigger.py`; `200`/`202`/`400`/`401`/`403`/`404`/`409`/`422`. A `CHAT` agent is `400 mode_not_supported`. The run row carries `headless=true` — every `/trigger` run is headless, even a `BOTH` agent's — so approvals need an `approver_role` and a `reject` ends it `rejected` (`agents.runs_headless(agent, run.headless)`).)
  - `Idempotency-Key` required (`400 missing_idempotency_key`; over 255 chars → `422`)
  - body validated against `input_schema` (`{input, config?, thread_id?, callback?}`, `extra=forbid`; `422 invalid_input`, reserved `configurable` keys `422`)
  - `202 {run_id}`; a replayed key returns `200` with the original run (A18) (the run id is derived from `(agent, caller, key)`, so a replay — even with a different body — answers `200` + `Idempotent-Replayed: true` and creates nothing; concurrent first requests race on the id (`ON CONFLICT (id) DO NOTHING` on Postgres, `ValueError` in memory/SQLite) and the loser returns the winner's run, soft-deleting the thread it made)
  - creates a new thread unless `thread_id` is passed (the thread must be the caller's and the agent's, else `404`; a busy thread is `409 run_conflict`)
  - Service tokens (A16): `VDREAMER_SERVICE_TOKENS="name:token,…"` (env-only, ≥16 chars, unique names, never equal to an internal token). A service token opens only `POST /agents/{name}/trigger` and `GET /runs/{id}[/events]`; the caller is `service:<name>`, needs no `X-User-Id`, passes `access_roles`, and owns the threads/runs it starts.
- [x] Outbox table and delivery for headless or role-gated interrupts: HMAC-signed webhook and/or SMTP relay, deep link `{WEB_BASE_URL}/review/{interrupt_id}`. At-least-once, with dedup on `event_id`. (`vdreamer.outbox`: `OUTBOX_DDL` (`vdreamer migrate` applies it) + `PostgresOutboxStore`, its in-memory twin, and `SqliteOutboxStore` for `vdreamer dev` (D11). Write side: `OutboxEventSink` wraps every event writer — each worker's per-run sink and the gateway's (cancel, headless reject, expiry) — and adds the entries *before* the event write, so a crash repeats a notification, never drops it. `event_id = sha256(run, kind, interrupt_id | status)` and entry id `{event_id}:{channel}`, so a recovered attempt that re-emits an event adds nothing. Read side: `OutboxDispatcher` in the gateway process (`serve` and `dev`) claims due entries with a lease (`FOR UPDATE SKIP LOCKED`, any number of replicas, no leader), retries with exponential backoff (5 s doubling, capped at 1 h) and parks an entry `dead` after 10 tries or at once on a permanent failure (a target off the allowlist). Bodies are compact key-sorted JSON signed `X-Vdreamer-Signature: sha256=HMAC(secret, "{timestamp}.{body}")` with `X-Vdreamer-Timestamp`/`-Event-Id`/`-Event`; no redirects, no proxy env. Env: `VDREAMER_OUTBOX_WEBHOOK_URL` (needs `VDREAMER_OUTBOX_SECRET`), `WEB_BASE_URL`, `VDREAMER_SMTP_HOST`/`_PORT`/`_FROM`/`_TO`/`_USER`/`_PASSWORD`/`_STARTTLS` (the relay is checked against `NETWORK_ALLOWLIST`). Notifications carry the link and the interrupt's routing fields, never its payload. A chat run's role-gated interrupt notifies too; an initiator-only one does not.)
- [x] Run callbacks (`completed|failed|interrupt`) delivered via the outbox to `CALLBACK_ALLOWLIST` URLs (A26). (`CALLBACK_ALLOWLIST` uses the `NETWORK_ALLOWLIST` syntax; unset, every callback is `422 callback_not_allowed`. The URL is checked at `/trigger` — http(s), no credentials, a host glob match or a name that resolves only into the allowed networks — and again at delivery. Every terminal status but `completed` is the `failed` event, with the real status in the body; `interrupt` fires on each `interrupt.requested`.)
- [x] Metric: outbox lag. (`vdreamer_outbox_lag_seconds`: age of the oldest pending entry, set on every dispatch pass.)

**Exit:** ✅ verified — a replayed idempotency key returns the original run, and a callback is delivered at least once, with dedup. `tests/api/test_trigger_m9.py` (both store backends): the trigger contract, key scoping per caller and agent, `thread_id`, the allowlist refusals, service tokens, and end to end with live workers: `test_a_completed_run_calls_back_once_signed` (smoke) — the first delivery gets a `503`, the retry delivers, the signature verifies, a key replay adds nothing — and a headless `sitecheck` run whose role-gated approval goes to the callback and the webhook under one `event_id`, then a `reject` calls back `failed` (`status: rejected`). `tests/outbox/test_outbox.py`: config validation, entry derivation, sink dedup and write order, signing, HTTP/SMTP delivery, backoff, dead-lettering, a lapsed lease redelivering the same `event_id`, SQLite persistence across a restart, the Postgres SQL shapes. Full suite: 827 passed, 25 skipped; ruff + mypy (strict) clean; `doctor --offline` green. Checked in a real `vdreamer dev` process: a service-token `/trigger` → `202`, the same key → `200` with the same run, no key → `400`, an off-list callback → `422`, and the run's signed `completed` callback received by a local endpoint. Also fixed with M9: `vdreamer worker` refuses a wildcard `--metrics-host` before opening any store. Not run live: `PostgresOutboxStore` (`tests/outbox/test_outbox.py::test_postgres_outbox_live` skips without `VDREAMER_TEST_PG_DSN`, per the SQLite-first testing phase) and a real SMTP relay.

---

## Milestone 10: CLI, Zero-Egress & Packaging

**Goal:** the complete CLI surface and the air-gap deliverables (§2.3, §5.2–§5.3). Operators use standard pip; uv appears only in the dev-side release pipeline.

**CLI commands**
- [x] `dev` (hot reload), `serve`, `worker`, `run` (Milestone 11), `validate`, `migrate`, `export-schema`, `doctor --offline`, `bundle`. (`dev --reload` is a supervisor (`vdreamer.reload`): it runs `python -m vdreamer dev …` as a child and restarts it when a `*.py`/`*.yaml`/`*.json` file changes under the working directory, `PYTHONPATH`, `--agents-dir` or the config files' directories. It polls, has no new dependency and debounces a save burst; state survives in `--db` (D11). `run` lands with Milestone 11.)
- [x] `validate`: aliases resolve; headless agents declare `input_schema` and interrupt `approver_role` (A16); approval routes wired; vdreamer checkpointer used; no ungoverned tools. (New check `checkpointer`: `build()` runs with stand-in `ctx.checkpointer`/`ctx.store`; a returned `StateGraph` is compiled by vdreamer, and anything `build()` compiled itself must carry exactly `ctx.checkpointer` (and no store of its own) — none or its own saver fails with the `**ctx.agent_kwargs()` hint. The rest landed with M5/M7/M8.)
- [x] `migrate`: vdreamer schema plus checkpointer/store setup. (`persistence.open_store` → LangGraph's `AsyncPostgresStore.setup()`. Workers now pass the store to `build()` as `ctx.store` and compile `StateGraph`s with it; `vdreamer dev` uses `AsyncSqliteStore` on the `--db` file.)
- [x] `export-schema`: OpenAPI 3.1, event JSON Schema, input schemas, TS types. (`vdreamer.export_schema`: `openapi.json` (the document `/api/v1/openapi.json` serves), `events.schema.json`, `agents/<name>.input.schema.json`, `vdreamer.d.ts` and `manifest.json` (versions + sha256 per file). The TS comes from `vdreamer.tsgen`, a JSON Schema → TypeScript translator with no npm involved: the `Envelope<T, P>`/`KnownEvent`/`VdreamerEvent` unions plus `Events`, `Api` and `Agents.<name>` namespaces; anything it does not understand becomes `unknown`, never `any`. Deterministic: two exports are byte-identical. Agents optional. The conformance kit joins it in M11.)

**Zero-egress**
- [x] Runtime settings: LangSmith off; `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; fastembed `local_files_only`; bundled `TIKTOKEN_CACHE_DIR`; vendored chart JS, no CDNs; `NETWORK_ALLOWLIST` enforced; no telemetry or update checks. (`vdreamer.offline.apply_offline_env()` runs before every CLI command except `bundle` and *forces* `LANGSMITH_TRACING=false`, `LANGCHAIN_TRACING_V2=false`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `FASTMCP_CHECK_FOR_UPDATES=off`, `DO_NOT_TRACK=1`. `bundle` fetches the tiktoken encodings, the image bakes them in, and `TIKTOKEN_CACHE_DIR` points at them. fastembed `local_files_only`, the vendored chart JS and `NETWORK_ALLOWLIST` landed with M6/M7.)
- [x] `doctor --offline`: probes backends, model files, MCP servers, sandbox; prints LangChain/LangGraph versions; asserts zero egress (CI with blocked network). (Since M0–M7. M10 adds the tiktoken cache check: set but missing or empty fails; unset with tiktoken installed warns. Plain `doctor` now runs the same probes, since an air-gapped product has one mode.)
- [x] Vendored swagger-ui assets pass `doctor --offline` (alongside the vendored chart JS). (Since M4/M6.)

**Release pipeline (dev side, uv)**
- [x] `uv build` produces the vdreamer sdist and wheel. (CI `airgap` and `release` jobs.)
- [x] `uv export --frozen --no-dev --all-extras --format requirements.txt` produces a hash-pinned `constraints.txt` that ships with each release, so operators reproduce the tested dependency set. (With `--no-emit-project`: a constraints file cannot name the project itself. The `release` job, on `v*` tags, attaches the wheel, sdist, `constraints.txt` and the `export-schema` tarball to the GitHub release.)

**`vdreamer bundle` (product feature, operator-run on a connected host; no uv)**
- [x] Runs `python -m pip download` against the release `constraints.txt` plus the operator's agent requirements into `wheelhouse/`, emitting a hash-pinned `requirements.lock`. (`vdreamer.bundle`. pip treats hashes in a `-c` file as `--require-hashes` for every requirement, which would reject unpinned agent requirements. So pip resolves against a hash-free copy of the pins, and every downloaded file of a pinned project must then match the release's version and one of its sha256s, or the bundle fails. The lock pins and hashes every file in the wheelhouse; two versions of one project is an error. `--extra`, `-r`, `--vdreamer <wheel>`, and `--platform`/`--python-version`/`--implementation`/`--abi` for cross-target bundles (binary only). Also writes `tiktoken/`, `bundle.json` and `INSTALL.txt`. It is the one command that runs without the offline environment.)
- [x] `--models` pre-fetches the fastembed aliases into `cache_dir`. (Into `OUT/fastembed/`, then located offline with `check_fastembed_files` to prove the workers will find it; ship it as `fastembed.cache_dir`.)

**Install on the air-gapped side (no uv)**
- [x] `python -m venv`, then `pip install --no-index --find-links ./wheelhouse --require-hashes -r requirements.lock` (matches PRD §5.3). (`docs/operations.md`, `INSTALL.txt`, and the CI `airgap` job, which runs it in a `--network none` container.)

**OCI image**
- [x] `vdreamer:<ver>` (gateway + worker, two entrypoints), pushed to the internal registry. (`ENTRYPOINT ["vdreamer"]`: `docker run vdreamer:<ver> serve|worker|migrate …`. Pushing to the internal registry is the operator's step.)
- [x] Built in CI with a multi-stage Dockerfile. The builder stage may use uv (`uv sync --locked --no-dev --no-editable --all-extras`). The runtime stage copies only the venv: no uv, no network. (Pinned `uv:0.10.10` in the builder only, which also bakes the tiktoken encodings. The runtime image runs as uid 10001, has a `vdreamer-sandbox` user (uid 10002, D19), sets the offline `ENV`, and asserts `! command -v uv` at build. CI `image` job: build, then `doctor --offline` and a smoke run under `--network none`.)

**Docs**
- [x] vLLM prerequisites: `--enable-auto-tool-choice`, a matching `--tool-call-parser`, `--reasoning-parser` (A21). (`docs/operations.md` §7, with the air-gapped install, image, topology and environment reference.)

**Exit:** ✅ verified — locally (Windows host): `uv build` → wheel; `uv export` → hash-pinned `constraints.txt`; a plain-pip venv with no uv on `PATH` ran `vdreamer bundle --vdreamer <wheel> --extra postgres -c constraints.txt`, and all 67 distributions matched the release hashes; a fresh `python -m venv` installed them with `pip install --no-index --find-links wheelhouse --require-hashes -r requirements.lock` (no uv on `PATH`); `doctor --offline` passed (zero egress, tiktoken cache found, vendored assets); and `validate` plus a `dev --input` run of the reference agent completed. `docs/operations.md` describes the same flow on a Linux VM. In CI, the `airgap` job repeats it inside a `--network none` `python:3.11-slim` container (egress truly blocked, no uv) and the `image` job runs the image's `doctor --offline` and a smoke run with the network off. Not run locally: the Docker image build (the Docker engine on this host did not start), which the `image` job covers. Tests: `tests/packaging/` (export-schema determinism, OpenAPI 3.1 validity, the TS generator, bundle lock/constraints/CLI, offline env, `--reload`, the checkpointer check, the store). Full suite: 889 passed, 25 skipped; ruff + mypy (strict) clean.

---

## Milestone 11: Protocol Contract & Developer Client

**Goal:** everything client teams and AI developers need without vdreamer shipping any frontend code (PRD §6, A45). No npm packages and no Angular.

**Protocol contract**
- [x] `export-schema` emits OpenAPI 3.1, the event JSON Schema, per-agent input schemas, and generated TS types. (Since M10; now also `conformance/`, the kit below.)
- [x] Each release attaches these artifacts. (The CI `release` job tars the whole `export-schema` output, kit included, onto every `v*` release.)
- [x] Document the normative client behavior from §6.1 (`docs/protocol.md` §1–§4, with the per-event fold rules in the kit's `README.md`):
  - protocol-version check
  - fetch-based SSE with `Last-Event-ID` reconnect
  - dedup by `seq`
  - ignore unknown event types; treat unknown kinds as downloads
  - the reference fold into run state
  - Two rules the recordings surfaced and the docs now state as normative:
    - `message.completed` is authoritative and may arrive before the message's last `token.delta`s (the batcher flushes at stream end), so later deltas are ignored.
    - A resolved interrupt re-runs the paused node from the top, re-emitting its `message.completed`/`tool.call` with the same ids, so clients upsert by id.
- [x] Document the interrupt handling obligations from §6.4. (`docs/protocol.md` §5: `actions[]` only, `edit_schema` forms, the resolve body per source, `expected_version`, `409`/`410`/`403 four_eyes`.)
- [x] Document the Web Backend proxy requirements from §6.2. (`docs/protocol.md` §1 and §7: headers injected server-side, no buffering, idle timeout > 2× heartbeat, `Last-Event-ID` forwarded, same-origin artifacts.)

**Conformance kit**
- [x] Recorded SSE streams, each paired with the expected client state, covering:
  - chat
  - mid-run reconnect
  - tool overflow
  - all four interrupt sources
  - artifacts
  - citations
  - (`vdreamer.conformance`, package data in `conformance/kit/`, 10 recordings: `chat`, `reconnect` (a drop after 4 events, a lagging `Last-Event-ID` that replays duplicates, then a page-refresh rebuild from `/threads/{id}/events`), `tool_overflow`, `interrupt_require_approval`/`_tool_review`/`_mcp_elicitation`/`_generic` (stream → drop at `awaiting_approval` → the resolve request → `Last-Event-ID` resume), `artifacts` (all five kinds), `citations`, and the hand-written `forward_compat` (an unknown type from a newer minor, unknown kind/renderer, heartbeats, a duplicate seq). Recorded from the real in-process gateway by `tests/conformance/record_kit.py`, with ids and timestamps normalized; marked `-text` in `.gitattributes` so it is byte-identical on every OS.)
- [x] A Python reference fold (test-only, not a shipped SDK) proves every recording produces its expected state. (`tests/conformance/reference_fold.py`.)

**`vdreamer run <agent>` terminal client**
- [x] Starts a run and pretty-prints events (tokens, thinking, tool calls, artifacts, citations). (`vdreamer.client`. Each message's text prints once, even when `message.completed` precedes the deltas. ANSI color only on a TTY.)
- [x] Prompts inline to resolve interrupts. (It fetches the record and offers its `actions[]`, then builds the §4.5 body per source: `decisions[]` per request for `tool_review`, answers keyed by request for `mcp_elicitation`, an `edit` payload shown with `edit_schema`, text for `feedback`/`respond`. It sends `expected_version`; a `422` is shown and asked again, a `409` refreshes the record, a `403` or a missing inbox (a service token) detaches with a reattach hint.)
- [x] Reattaches to a thread after interruption. (`--thread ID` follows the thread's active run from its first event; Ctrl-C and detaches print `vdreamer run <agent> --thread <id>`.)
- [x] Talks to the REST API only, so it exercises the same contract clients use. (httpx with `X-Protocol-Version: 1`, a major mismatch refused, reconnect with exponential backoff plus jitter and `Last-Event-ID`, dedup by `(run_id, seq)`, unknown types shown dimmed. Chat loops on `you>` (`--once` for one run; `--message`, `--field NAME`, `--input JSON`); headless agents, `--trigger` and `--service` go through `/trigger` with an `Idempotency-Key`.)

**Exit:** ✅ verified (for the kit) — the conformance kit passes against the Python reference fold:
- `tests/conformance/test_conformance_kit.py`, 34 tests:
  - every recording folds to its `expected_state`;
  - scenario facts are checked on the raw events independently of the fold, and every recorded event validates against `events.schema.json`;
  - a major mismatch is refused;
  - a fresh live re-recording folds to the committed states, so the kit is current.
- `tests/client/test_run_client.py`, 11 tests: `vdreamer run` against a real uvicorn gateway over TCP:
  - a chat turn with thinking, and the chat loop;
  - the generic, `require_approval` (a refused edit, then an accepted one) and `tool_review` interrupts answered inline;
  - a service trigger detaching at a role-gated approval;
  - reattach to a waiting run;
  - plus reconnect/dedup/backoff and the protocol check on a scripted transport.
- Checked by hand across processes: `vdreamer dev` serving the pinger, and `vdreamer run pinger --field message --message … --once` and `vdreamer run pinger --trigger --input … --idempotency-key k1` both reached `[run completed]`.
- Full suite: 934 passed, 25 skipped; ruff + mypy (strict) clean.
- Deferred to Milestone 12: the second half of the exit, running UC1–UC3 end to end with `vdreamer run`, because those agents land there.

---

## Milestone 12: Reference Agents & Acceptance (P0 Complete)

**Goal:** UC1–UC3 in `examples/`, which are the acceptance test (§7.3). Core is P0-complete only when these build **without touching vdreamer internals**.

**UC1 SQL & Data Analysis (Chat)** — `examples/uc1_sql_analyst`, `SqlAnalystAgent`
- [x] `schema_link` (Cypher via `ctx.resources["semantic"]`) → `generate_sql` → `dry_run` (EXPLAIN) → `ApprovalGate(kind="sql_review", lineage)` → execute on `ctx.resources["warehouse"]` → `table` artifact → sandboxed chart → composite `html`. (Two compiled gates, read and write; `prepare=dry_run` runs `EXPLAIN [QUERY PLAN]`; the chart is matplotlib in `ctx.sandbox.arun`, and a `SandboxUnavailable` host logs "chart skipped" and still emits the table + report.)
- [x] Non-SELECT statements add `approver_role` + `db:write` scope. Exercises the `edit` action. (The write gate carries `approver_role="dba"` + `scope="db:write:warehouse"`; `edit_schema=SqlEdit`, and `feedback` routes back through `generate_sql`.)

**UC2 Telecom Site Validation (Headless)** — `examples/uc2_site_validation`, `SiteValidationAgent`
- [x] ZIP upload → `/trigger` with `package: FileRef` → `safe_extract` → VLM inspection + MCP site-inventory tools → `json` verdict (`renderer="verdict.v1"`) → `ApprovalGate(approver_role="site_approver")` → callback. (`inventory_server.py` is a real FastMCP stdio server — `get_site` read, `record_validation` a `side_effects=True` write.)

**UC3 Multi-PDF Research (Chat + Headless, EN + AR)** — `examples/uc3_pdf_research`
- [x] Ingest agent: sandbox parse → page-offset chunks → dense + sparse → Qdrant upsert. (`pypdf` runs *inside* `ctx.sandbox.arun`; because sandbox stdout is capped at 64 KB, the parser returns pages in zlib+base64 batches and the node loops until it has every page. Point ids are `uuid5`, so re-ingesting a document overwrites rather than duplicates.)
- [x] Chat agent (`create_agent`): hybrid `query_points` RRF → rerank → answer → `emit_citation` per chunk → comparison `table`. (`CitationMiddleware.after_agent` emits one `citation` per `[n]` the answer actually used, bound to the answer's `message_id`; sources are numbered per run.)
- [x] Exercises the fastembed aliases, `ctx.resources["vectors"]`, and `thinking.delta`.

**Packaging and acceptance gates**
- [x] In the repo, each example is a uv workspace member (`examples/*`) depending on `vdreamer` via a workspace source, so examples consume only the public API. Each example's dependencies are also declared in standard `pyproject.toml` form, so operators can install it with pip. (`[tool.uv.workspace]` members + a dev-group dependency on each example; `[tool.uv.sources] vdreamer = { workspace = true }`.)
- [x] CI runs UC1–UC3 against the floor (`lowest-direct`) **and** latest 1.x jobs (A41). (`tests/acceptance` added to both jobs in `.github/workflows/ci.yml`.)
- [x] Gap review: any core change a reference agent needs must pass the design rule (≥2 reference agents need it / can't be built on primitives) before merging into core.

**Gap review — what the reference agents needed from core.** Two changes, both justified under the design rule; everything else was built on existing primitives.
1. **`resolved_by` in the `require_approval` resume value** (`gateway/routes_interrupts.py`). UC1's write path must record *which* dba approved, and UC2's MCP receipt must carry the approver, not the role — two agents, and the identity is only known to the gateway, so it can't be built in agent code. `_graph_value` adds `resolved_by` (read from the interrupt record's own field) when the resume map is built, so what the graph receives carries who decided while the **stored resolution and the REST response stay byte-for-byte what the client sent** — no protocol-surface change, and the conformance kit needs no re-recording. The other three sources keep their shapes verbatim. `context.require_approval`'s docstring documents it, and `tests/api/test_interrupts.py` now asserts both halves: the record's `resolution` unchanged, the graph's value enriched.
2. **`vdreamer run --inbox`** (`client.py`, `cli.py`). Both UC1's four-eyes write and UC2's headless verdict need an approver who does *not* own the thread to find and resolve work; threads are owner-scoped (§4), so the inbox is the approver's view. This is client surface, not protocol — no new endpoint.

Also fixed while driving the examples: `vdreamer run --thread` with a `--message` replayed the thread's last finished run instead of starting the new turn; it now replays only when no new input was given, and still reattaches to a genuinely active run first.

**Exit:** ✅ `uv run pytest tests/acceptance` — 19 passed, 3 skipped (the live tier). All three agents run their full scenarios through a real uvicorn gateway, real workers, real stores and the outbox, driven by `vdreamer run` (`run_terminal`) exactly as a developer types it, with **zero vdreamer internals touched** — enforced by an AST scan over every `examples/` source (`test_reference_agents.py`: only top-level `vdreamer` names, all in `__all__`; the scan is itself tested against a planted violation), plus `validate_agent` for all four agents and an alias check over every shipped `models*.yaml`/`resources*.yaml`.
- `test_uc1_sql_analyst.py`, 4 tests: approve → table + report + chart-or-skip; a refused edit (422) then a valid one; feedback → replan → reject; and the write path — alice (dba) is refused by four-eyes and detaches (code 3), a non-dba's inbox is empty, bob approves from `--inbox`, alice reattaches to her thread and sees "Done: 1 row(s) changed in orders", with the SQLite warehouse actually updated.
- `test_uc2_site_validation.py`, 4 tests: approve → the MCP receipt carries `approved_by: "sam"` (not the role) and the outbox holds `interrupt` + `completed` callbacks; a mismatch + reject → run `rejected`, a `failed` callback, nothing recorded; an edited verdict; and an unsafe ZIP (`../../etc/passwd`) failing the run.
- `test_uc3_pdf_research.py`, 2 tests: EN + AR text documents and a PDF ingested into a real `AsyncQdrantClient(":memory:")` shared by both agents' runners (5 points on POSIX, 4 plus a "skipped" row where the sandbox is unavailable); then `search_documents` → a cited answer with the reasoning shown as thinking and `[1]` resolved to `doc://policy_en.txt` page 1; a second turn on the same thread running `compare_documents` → a `comparison` table with both languages, per-run source numbering, and citations only for the markers the answer used. Plus: no sources → no citations.
- `test_reference_agents.py`, 9 tests: the public-API scan, the scan's own negative test, `validate_agent` per agent, and the config alias checks.
- `test_uc_live_ollama.py`, 3 tests, **opt-in** (`VDREAMER_TEST_OLLAMA=1`): the same three stacks with each example's chat/VLM alias built from its own `config/models.ollama.yaml` — Ollama `gemma4:31b-cloud`. The `-cloud` tag runs the model on ollama.com, so this tier causes egress and never runs in CI; it asserts the workflow, not the model's wording.
- What is real in the acceptance tier vs. scripted at the boundary is documented in `tests/acceptance/_uc_kit.py`: real gateway/workers/stores/outbox/client, real Qdrant, the real MCP stdio server, real `safe_extract` and the real sandbox on POSIX; scripted only at the model and driver edges (`ScriptedModel`, `FakeNeo4j` reading the example's own `demo/semantic.json`, and lexical stand-ins with fastembed's method shapes).
- Air-gap note: the examples' production `models.yaml` point at vLLM + the multilingual fastembed models, pre-fetched by `vdreamer bundle --models`; `validate` on UC3 fails only where that fastembed cache is absent, which is the intended signal.
- Full suite: **955 passed, 28 skipped**; ruff + mypy (strict) clean.

**P0 complete.**

---

## First-contact review (post-P0)

A new-developer pass over the README, `vdreamer dev`/`run`, the REST/SSE surface and Swagger, plus the LangChain/LangGraph docs examples served by vdreamer (`examples/playground`, driven offline by `tests/acceptance/test_playground.py` and live against Ollama `gemma4:31b-cloud`). What it found and what changed:

- [x] **A docs-style tool node showed every tool call twice.** The LangGraph quickstart calls `tool.invoke(tool_call["args"])`, so the governed proxy had no id and minted one; the model's own call then surfaced too when its `ToolMessage` landed. A bare-args call now leaves the id to `governed_call`, which flags it `id_assigned`; the stream mapper binds such a call to the oldest held model call in the same namespace with the same name and matching (or masked) args, so the client sees one `tool.call`/`tool.result` under the model's id. Both drivers, golden test `test_golden_bare_args_governed_call_is_one_call`.
- [x] **Deltas trailed their `message.completed`.** A fast model finishes inside the 50 ms batch window, so its whole text arrived after completion. The mapper now flushes a message's open batches right before its `message.completed` (golden sequences and the conformance kit re-recorded; `protocol.md` still tells clients to ignore later deltas).
- [x] **Secrets in tool error messages.** Redaction covered arguments only, so `RuntimeError("password=hunter2")` reached the model, the event log and the client. `tools.redact_text` (public) masks `key=value`/`key: value` pairs for the redaction keys and any masked argument value echoed verbatim; governance applies it to contained errors (A33).
- [x] **`vdreamer dev --input` said only `-> failed (9 events)`.** It now validates the input against `input_schema` first (the gateway's `422`), prints every event as `vdreamer run` would, and prints the final error.
- [x] **`dev`/`worker`/`serve` ran agents `validate` rejects** (e.g. a `HEADLESS` agent with no `input_schema`). `validate.static_checks` (the checks that need no `build()`) now gate every entrypoint and `export-schema`.
- [x] **`vdreamer run --message` against a typed agent** sent `{"messages": …}`, which the gateway's schema silently dropped. The client now sends the text to the schema's single string field (or asks for `--field`/`--input`).
- [x] **`emit_progress` scale.** Undocumented; `0.2` rendered as `0%`. It is a percentage (PRD §3.3 row, docstring), out-of-range values raise, and the client prints `:g`.
- [x] **The smoke agent showed nothing in `vdreamer run`** (its reply was a state key). `pinger` also returns an `AIMessage`; the README now says what reaches clients.
- [x] **Swagger UI in a dark browser was unreadable** (checked in the desktop app's browser pane): the shell left the canvas transparent, so Swagger's light-theme text (`#3b4151`) sat on a dark page under a white auth bar. `docs.html` declares `color-scheme: light dark` with explicit backgrounds, and `swagger-init.js` toggles Swagger's own vendored `.dark-mode` theme from `prefers-color-scheme` (live on change). `/static` responses now send `Cache-Control: no-cache` (ETag revalidation), since a heuristically cached `swagger-init.js` survived a restart; and `GET /` redirects to `/api/v1/docs` instead of a 404. Authorize → Try it out → `GET /agents` verified in the pane.
- [x] **`vdreamer dev`'s default `./.vdreamer/` state was not gitignored**, so the quickstart dirtied the tree; `.vdreamer/` is now ignored.
- [x] **Docs:** a README "Your first agent" section (`ctx.tools.govern`, what clients see, `--message` mapping, `msg.text` under `output_version="v1"`, the 16-character token), and a clearer short-token error.

Exit: `uv run pytest` — 978 passed, 28 skipped (including `tests/acceptance/test_playground.py`, 7 tests); ruff + mypy (strict) clean; the playground's model-backed agents also run live against Ollama `gemma4:31b-cloud`.

Not changed, by design: `output_version="v1"` (§2.4) — the playground reads `msg.text`; Ollama `-cloud` models ignoring `format` (an Ollama limitation — the router uses `method="function_calling"`).

---

## Post-P0 (P1: tracked, not scheduled)

Per §7.1:
- Container sandbox image + `vdreamer-sandbox:<ver>`
- `vdreamer new` scaffolding (emits a standard `pyproject.toml`, tool-agnostic)
- Redis adapter
- OTLP export
- Quotas
- Time-travel API
- MCP `forward` identity
- Legacy SSE MCP transport

---

## Cross-cutting rules (apply to every milestone)

1. **Core = generic primitives only.** A capability enters core only if ≥2 reference agents need it and it can't be built on existing primitives (PRD design rule).
2. **Never reimplement LangGraph primitives.** Persistence, store, HITL, run control, and node timeouts come from upstream (§2.4).
3. **All streaming specifics stay in the one adapter module** (A31). v2 and v3 must keep passing the golden-event suite.
4. **Protocol version stays `1.0`.** Envelope `"v"` + `X-Protocol-Version`; clients ignore unknown event types (§6.1). Any protocol change updates the conformance kit in the same PR.
5. **Zero egress is mandatory.** Every new dependency, endpoint, or vendored asset must pass `doctor --offline` in blocked-egress CI (§5.2).
6. **A 2.0 of `langchain`/`langgraph` means a vdreamer major** (A41). Minors are tracked, never silently absorbed.
7. **Sandbox features are Unix-only** (D19). Develop on Windows, exercise on Linux CI (§5.1 reference topology).
8. **uv is the dev toolchain, not a product dependency.**
   - In the repo, developers and coding agents make dependency changes only through `uv add`/`uv remove`, keep `uv.lock` committed, and run everything via `uv run`. No `pip install` into the dev env, hand-edited `requirements*.txt`, or `setup.py`.
   - vdreamer's runtime code, CLI, `bundle` output, and install instructions must never import, invoke, or require uv. The plain-pip CI job enforces this.
9. **No frontend code.** No Angular, UI components, or client SDKs in this repo. Client teams build against the protocol contract (§6).
