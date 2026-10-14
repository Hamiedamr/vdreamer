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
  - (Upstream service tokens for `/trigger`, A16, land with Milestone 9.)

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

**Governed tool pipeline**
- [ ] `ctx.tools.govern(tools)` proxy, plus `VDreamerToolMiddleware` (`@wrap_tool_call`) for `create_agent`. Tool name and schema are preserved (A37).
- [ ] Governance steps, in order:
  1. Authorize (scope/roles).
  2. Emit `tool.call`, redacting policy keys plus the defaults password/token/secret/api_key/authorization (A33).
  3. Optional approval gate.
  4. Execute under `timeout_s` (default 60), with `ctx.idempotency_key`.
  5. Normalize: a result over `max_result_chars` (8,000) becomes a `json`/`file` artifact; the model gets a truncated body plus the `artifact_id`.
  6. Contain errors (`tool_error|timeout|denied|transport|invalid_args`). The run continues unless `fail_run=true`.
  7. Emit `tool.result` with `source` and `duration_ms`.
- [ ] `ToolRuntime[VDreamerContext]` parameter access: run, thread, user, roles, idempotency_key, state, stream_writer, tool_call_id.

**MCP.** Isolated in one module; `langchain.mcp` is version-pinned (A38/A39).
- [ ] `resources.yaml` entries: `type: mcp`, `url` → Streamable HTTP or `command` → stdio, `bearer_env`, `startup: required|optional`, `refresh_s`, `include` globs, `identity`.
- [ ] One `MCPAdapter` per server at worker boot. `list_tools()` is cached and refreshed every `refresh_s`. Tools are named `{server}_{tool}` and selected with `ctx.tools.mcp(server, include=[...])`.
- [ ] Each run records its tool-snapshot hash.
- [ ] Failures: a down `required` server fails `/readyz`; an `optional` one is excluded and logged.
- [ ] Elicitation arrives as a LangGraph interrupt and is normalized (§4.5).
- [ ] Network rules (A40):
  - `NETWORK_ALLOWLIST` is enforced at config load **and** at connect.
  - stdio `command` must be an absolute path to a preinstalled binary.
  - Launchers that fetch packages at runtime (`npx`, `uvx`, `pip`) are rejected.
  - stdio servers run under rlimits with only their declared env.
- [ ] `validate` checks: no ungoverned tools; approval routes wired.

**Exit:** a governed tool run shows a redacted `tool.call` → `tool.result` with an overflow artifact. MCP stdio and HTTP servers are both discoverable, and a failed optional server is excluded.

---

## Milestone 6: Artifacts, Files & Sandbox

**Goal:** the artifact system (D6–D8), file inputs, safe extraction, and the sandbox (§3.4–3.5).

**Artifact core**
- [ ] `emit_artifact(name, kind, data, metadata?, renderer?)` and the `Artifact` return type.
- [ ] 5 kinds: `table`, `image`, `html`, `json`, `file`. **No auto-detection**: unsupported data → `TypeError`.
- [ ] Write order (D7): bytes → MinIO `artifacts/{thread_id}/{run_id}/{artifact_id}.{ext}` → metadata row → `artifact.created` event.
- [ ] Previews per kind: schema + 10 rows + row_count / dimensions / title + summary / ≤64 KB body / filename + mime.
- [ ] `artifact.created` payload carries the full field set, including `question_id`, `session_id`, `view_url`, `download_url`.

**Tables**
- [ ] Incremental Parquet row groups.
- [ ] `/preview?offset&limit&sort` (limit ≤1,000).
- [ ] `/download?format=csv|xlsx|parquet` with streaming conversion. XLSX beyond 1,048,575 rows → `422`.
- [ ] `TABLE_MAX_ROWS` 5M, with `metadata.truncated`.

**HTML, access, retention**
- [ ] HTML: `artifact://` links are rewritten to `/raw`. CSP allows no external origins, with `connect-src 'none'`. Chart JS is vendored at `/static/vendor/` (A13).
- [ ] Access: thread owner, granted approver, or `artifact:read:all` (A8). Retention is 30 days; a missing object returns `410` (A9).

**Files**
- [ ] `FileRef` + `ctx.files.open(ref)`; the gateway validates the `inputs/` prefix (A17).
- [ ] `ctx.files.safe_extract(ref, allow=[...]) -> Manifest` (A34, security-critical):
  - caps on size, entry count, compression ratio; nesting ≤2
  - rejects traversal, absolute paths, symlinks, encryption
  - name decoding UTF-8 → CP437 → CP1256
  - manifest with sha256, plus a listing of skipped types

**Sandbox**
- [ ] `ctx.sandbox.run(code, inputs, timeout_s=60)` (D19/A19, Unix-only): dedicated uid, rlimits (CPU, 2 GB memory, file size, nproc), wiped scratch dir, empty env, network off where possible. Files written to `./out/` become artifacts.
- [ ] The sandbox interpreter is the worker environment's Python. pandas, pyarrow, matplotlib, plotly, and numpy come from the `sandbox` extra.

**Exit:** the sandbox executes untrusted code with rlimits verified; `safe_extract` defeats the traversal/zip-bomb test corpus; artifacts render end-to-end through the gateway proxy.

---

## Milestone 7: Models & Resources

**Goal:** `models.yaml` and `resources.yaml` resolution (§3.6–§3.7, A35, A36, A21).

- [ ] vLLM/Ollama aliases become LangChain chat models with `output_version="v1"`, each with a per-alias thinking mode (`reasoning_content`, `think_tags`).
- [ ] fastembed aliases (dense/sparse/rerank) load once per worker from `cache_dir` as `TextEmbedding` / `SparseTextEmbedding` / `TextCrossEncoder`, with `local_files_only` and CPUExecutionProvider (A36).
- [ ] `resources.yaml`: qdrant/neo4j/sql adapters.
  - Pooled async clients are opened at worker start and injected, native and unwrapped, via `ctx.resources[alias]` (A35).
  - `read_only` is enforced.
  - Adapter class shape: `connect` / `probe` / `close`.
- [ ] Secrets are referenced only through `*_env` keys and never written into YAML (A23).
- [ ] `validate` / `doctor` fail on unknown aliases, unreachable endpoints, or missing model files.

**Exit:** an agent declaring models and resources passes `uv run vdreamer validate` and runs against stub backends.

---

## Milestone 8: HITL Protocol

**Goal:** the full interrupt/approval pipeline, with all four interrupt sources normalized (§4, A44).

**Approval primitives**
- [ ] `require_approval(kind, payload, actions, approver_role?, scope?, edit_schema?, ttl_hours?)` wraps `interrupt()` (A15). The policy is serialized and enforced by the gateway.
- [ ] `ApprovalGate` subgraph: `prepare → gate → {approve|edit: execute, reject: on_reject, feedback: replan}`. `validate` flags unwired routes.
- [ ] `interrupts` table record (status, version, expires_at, resolved_by, resolution). Resume restarts the node from the top.

**Resolve endpoint**
- [ ] Body `{action, payload?, text?, expected_version}`. Multi-action interrupts use `{decisions: [...], expected_version}`, in order.
- [ ] Responses: `200` | `403 forbidden_role` | `409` stale version | `410` expired | `422` invalid edit.
- [ ] Action semantics:
  - `edit`: validated against `edit_schema` and written to state before execution
  - `feedback`: routes to replan
  - `reject` in headless runs: run → `rejected`
  - `respond`: returns text to the tool

**Authority and expiry**
- [ ] Authority: the initiator by default; `approver_role` overrides. `HITL_STRICT_SCOPES` enforces four-eyes (approver ≠ initiator, A32).
- [ ] Expiry: a leader-elected sweeper marks the interrupt `expired`, the run `expired`, and unlocks the thread.
- [ ] `awaiting_approval` frees the worker; resolution enqueues a resume.

**Interrupt-source normalization (A44).** One record, event, and endpoint for all four:
- [ ] `require_approval`
- [ ] HITL middleware `interrupt_on` → `tool_review`
- [ ] MCP elicitation → `mcp_elicitation`
- [ ] raw `interrupt()` → `generic`
- [ ] Policies for non-vdreamer sources attach via `interrupt_policies`.

**Exit:** all four sources resolve through one endpoint; four-eyes and version-conflict tests pass.

---

## Milestone 9: Headless Trigger, Outbox & Callbacks

**Goal:** the `/trigger` path and notification plumbing (D15, A26, D10).

- [ ] `POST /agents/{name}/trigger`:
  - `Idempotency-Key` required
  - body validated against `input_schema`
  - `202 {run_id}`; a replayed key returns `200` with the original run (A18)
  - creates a new thread unless `thread_id` is passed
- [ ] Outbox table and delivery for headless or role-gated interrupts: HMAC-signed webhook and/or SMTP relay, deep link `{WEB_BASE_URL}/review/{interrupt_id}`. At-least-once, with dedup on `event_id`.
- [ ] Run callbacks (`completed|failed|interrupt`) delivered via the outbox to `CALLBACK_ALLOWLIST` URLs (A26).
- [ ] Metric: outbox lag.

**Exit:** a replayed idempotency key returns the original run; a callback is delivered at least once, with dedup.

---

## Milestone 10: CLI, Zero-Egress & Packaging

**Goal:** the complete CLI surface and the air-gap deliverables (§2.3, §5.2–§5.3). Operators use standard pip; uv appears only in the dev-side release pipeline.

**CLI commands**
- [ ] `dev` (hot reload), `serve`, `worker`, `run` (Milestone 11), `validate`, `migrate`, `export-schema`, `doctor --offline`, `bundle`.
- [ ] `validate`: aliases resolve; headless agents declare `input_schema` and interrupt `approver_role` (A16); approval routes wired; vdreamer checkpointer used; no ungoverned tools.
- [ ] `migrate`: vdreamer schema plus checkpointer/store setup.
- [ ] `export-schema`: OpenAPI 3.1, event JSON Schema, input schemas, TS types.

**Zero-egress**
- [ ] Runtime settings: LangSmith off; `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; fastembed `local_files_only`; bundled `TIKTOKEN_CACHE_DIR`; vendored chart JS, no CDNs; `NETWORK_ALLOWLIST` enforced; no telemetry or update checks.
- [ ] `doctor --offline`: probes backends, model files, MCP servers, sandbox; prints LangChain/LangGraph versions; asserts zero egress (CI with blocked network).
- [ ] Vendored swagger-ui assets pass `doctor --offline` (alongside the vendored chart JS).

**Release pipeline (dev side, uv)**
- [ ] `uv build` produces the vdreamer sdist and wheel.
- [ ] `uv export --frozen --no-dev --all-extras --format requirements.txt` produces a hash-pinned `constraints.txt` that ships with each release, so operators reproduce the tested dependency set.

**`vdreamer bundle` (product feature, operator-run on a connected host; no uv)**
- [ ] Runs `python -m pip download` against the release `constraints.txt` plus the operator's agent requirements into `wheelhouse/`, emitting a hash-pinned `requirements.lock`.
- [ ] `--models` pre-fetches the fastembed aliases into `cache_dir`.

**Install on the air-gapped side (no uv)**
- [ ] `python -m venv`, then `pip install --no-index --find-links ./wheelhouse --require-hashes -r requirements.lock` (matches PRD §5.3).

**OCI image**
- [ ] `vdreamer:<ver>` (gateway + worker, two entrypoints), pushed to the internal registry.
- [ ] Built in CI with a multi-stage Dockerfile. The builder stage may use uv (`uv sync --locked --no-dev --no-editable --all-extras`). The runtime stage copies only the venv: no uv, no network.

**Docs**
- [ ] vLLM prerequisites: `--enable-auto-tool-choice`, a matching `--tool-call-parser`, `--reasoning-parser` (A21).

**Exit:** fresh VM + wheelhouse (installed with plain pip) + image → `doctor --offline` green → reference agent runs with egress blocked, and no uv present on the target.

---

## Milestone 11: Protocol Contract & Developer Client

**Goal:** everything client teams and AI developers need without vdreamer shipping any frontend code (PRD §6, A45). No npm packages and no Angular.

**Protocol contract**
- [ ] `export-schema` emits OpenAPI 3.1, the event JSON Schema, per-agent input schemas, and generated TS types.
- [ ] Each release attaches these artifacts.
- [ ] Document the normative client behavior from §6.1:
  - protocol-version check
  - fetch-based SSE with `Last-Event-ID` reconnect
  - dedup by `seq`
  - ignore unknown event types; treat unknown kinds as downloads
  - the reference fold into run state
- [ ] Document the interrupt handling obligations from §6.4.
- [ ] Document the Web Backend proxy requirements from §6.2.

**Conformance kit**
- [ ] Recorded SSE streams, each paired with the expected client state, covering:
  - chat
  - mid-run reconnect
  - tool overflow
  - all four interrupt sources
  - artifacts
  - citations
- [ ] A Python reference fold (test-only, not a shipped SDK) proves every recording produces its expected state.

**`vdreamer run <agent>` terminal client**
- [ ] Starts a run and pretty-prints events (tokens, thinking, tool calls, artifacts, citations).
- [ ] Prompts inline to resolve interrupts.
- [ ] Reattaches to a thread after interruption.
- [ ] Talks to the REST API only, so it exercises the same contract clients use.

**Exit:** the conformance kit passes against the Python reference fold. An AI developer can run UC1–UC3 end-to-end, including approvals, using only `vdreamer run`.

---

## Milestone 12: Reference Agents & Acceptance (P0 Complete)

**Goal:** UC1–UC3 in `examples/`, which are the acceptance test (§7.3). Core is P0-complete only when these build **without touching vdreamer internals**.

**UC1 SQL & Data Analysis (Chat)**
- [ ] `schema_link` (Cypher via `ctx.resources["semantic"]`) → `generate_sql` → `dry_run` (EXPLAIN) → `ApprovalGate(kind="sql_review", lineage)` → execute on `ctx.resources["warehouse"]` → `table` artifact → sandboxed chart → composite `html`.
- [ ] Non-SELECT statements add `approver_role` + `db:write` scope. Exercises the `edit` action.

**UC2 Telecom Site Validation (Headless)**
- [ ] ZIP upload → `/trigger` with `package: FileRef` → `safe_extract` → VLM inspection + MCP site-inventory tools → `json` verdict (`renderer="verdict.v1"`) → `ApprovalGate(approver_role="site_approver")` → callback.

**UC3 Multi-PDF Research (Chat + Headless, EN + AR)**
- [ ] Ingest agent: sandbox parse → page-offset chunks → dense + sparse → Qdrant upsert.
- [ ] Chat agent (`create_agent`): hybrid `query_points` RRF → rerank → answer → `emit_citation` per chunk → comparison `table`.
- [ ] Exercises the fastembed aliases, `ctx.resources["vectors"]`, and `thinking.delta`.

**Packaging and acceptance gates**
- [ ] In the repo, each example is a uv workspace member (`examples/*`) depending on `vdreamer` via a workspace source, so examples consume only the public API. Each example's dependencies are also declared in standard `pyproject.toml` form, so operators can install it with pip.
- [ ] CI runs UC1–UC3 against the floor (`lowest-direct`) **and** latest 1.x jobs (A41).
- [ ] Gap review: any core change a reference agent needs must pass the design rule (≥2 reference agents need it / can't be built on primitives) before merging into core.

**Exit:** all three agents run full scenarios in the Milestone-10 air-gap environment, with zero vdreamer internals touched. **P0 complete.**

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
