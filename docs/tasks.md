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
- [ ] `run_events(run_id, seq, type, ns, payload jsonb, ts)`, PK `(run_id, seq)` (A3); SSE `id: {run_id}:{seq}`.
- [ ] Envelope: `{"v":"1.0", type, run_id, thread_id, agent, seq, ts, ns, payload}`.

**Stream drivers.** One adapter module (A31), selected by `VDREAMER_STREAM_DRIVER`:
- [ ] **v2 driver (default):** `astream(stream_mode=["messages","updates","custom"], subgraphs=True, version="v2")`. Pending interrupts are read from `aget_state(config).interrupts` after the stream ends.
- [ ] **v3 driver (opt-in, beta):** `astream_events(version="v3")`. Maps the `messages` projections (text, reasoning, tool calls, usage), lifecycle, subgraphs, and custom events.

**Event mapping**
- [ ] Event types: `run.status`, `token.delta`, `thinking.delta`, `message.completed`, `tool.call`, `tool.result`, `artifact.created`, `citation`, `interrupt.requested`, `interrupt.resolved`, `progress`, `log`.
- [ ] Thinking extraction (D18, A20):
  - standard `reasoning` blocks first
  - per-alias fallbacks: the `reasoning_content` field, then a stateful `<think>` splitter that handles split chunks
  - suppressed when `expose_thinking=false`
- [ ] Custom data: vdreamer-enveloped custom data → artifact/citation/progress/log events; other custom data → `log`. `ns` is preserved for subgraphs.

**Delivery and testing**
- [ ] Delta batching (50 ms / 32 tokens), post-run compaction (A4), 15 s heartbeat, `X-Accel-Buffering: no` (A22).
- [ ] Golden-event test suite: both drivers emit **identical** protocol events for the same graphs. This suite is the gate for promoting v3.

**Exit:** the golden-event suite passes for the v2 driver; v3 runs as opt-in beta.

---

## Milestone 3: Runs, Queue, and Runtime Topology

**Goal:** the production run lifecycle, with Postgres as the only gateway↔worker transport (A7, D12).

**Entrypoints and queue**
- [ ] Standard console-script entrypoints (`[project.scripts]`); in the repo, developers run them via `uv run`:
  - `vdreamer serve` (stateless gateway)
  - `vdreamer worker`
  - `vdreamer dev` (single process on SQLite + local filesystem store, D11)
- [ ] Queue tables: claim via `FOR UPDATE SKIP LOCKED`, with a 10 s lease heartbeat.
- [ ] Run states: `queued → running → (awaiting_approval → running)* → completed | failed | rejected | expired | cancelled`.

**Cancel, shutdown, recovery**
- [ ] Cancel: `POST /runs/{id}/cancel` calls `RunControl.request_drain()`, which leaves a resumable checkpoint; the run moves to `cancelled`.
- [ ] Worker SIGTERM: drain and requeue without consuming an attempt.
- [ ] Crash recovery: an expired lease requeues the run with `attempt+1` (max 3) and resumes from the checkpoint. The in-flight node re-executes, so side effects use `ctx.idempotency_key` (A6).
- [ ] Node timeouts and errors (`NodeTimeoutError`, `NodeError`) map to `run.status.error`.

**Fan-out and concurrency**
- [ ] `LISTEN/NOTIFY {run_id, seq}` fan-out. Any gateway replica serves any SSE reconnect: replay `seq > Last-Event-ID`, then tail live.
- [ ] One active run per thread: `409 {active_run_id, reason}` (D12, A11). `?wait_for_idle=true` queues behind `running` only.

**Exit:** kill a worker mid-run; the run resumes from its checkpoint and the side effect runs exactly once (idempotency test with a side-effecting tool).

---

## Milestone 4: REST Gateway (`/api/v1`)

**Goal:** the full REST surface, with uniform errors and ops endpoints.

**Routes**
- [ ] `GET /agents`, `GET /agents/{name}` (mode, input JSON Schema).
- [ ] Threads: create (`201`), list (owner-scoped), get (history from checkpoint), soft delete.
- [ ] `POST /threads/{id}/runs` → `202 {run_id}`; `GET /runs/{id}`.
- [ ] SSE with `Last-Event-ID` replay: `GET /runs/{id}/events`, `GET /threads/{id}/events` (joins the active or latest run).
- [ ] Interrupts: `GET /interrupts`, `GET /interrupts/{id}`, `POST /interrupts/{id}/resolve` (§4).
- [ ] Artifacts: `/artifacts/{id}`, `/preview`, `/raw`, `/download` (§3.4).
- [ ] Ops: `GET /protocol/schema.json`, `/openapi.json`, `/healthz`, `/readyz`, `/metrics`.

**Errors and auth**
- [ ] Uniform error body `{error: {code, message, details?}}` with stable codes (`run_conflict`, `forbidden_role`, `artifact_expired`, …).
- [ ] Internal auth (D13, A14):
  - `Bearer <internal_token>` + `X-User-Id` / `X-User-Roles`
  - two tokens valid at once, for rotation
  - role gating via `access_roles`
  - the gateway binds to the internal interface only

**Metrics**
- [ ] Prometheus: queue depth, run latency, SSE connections, model latency/tokens per alias, tool latency/errors per source, MCP health, sandbox timeouts, outbox lag.

**Exit:** REST contract tests (`uv run pytest tests/api`) cover every route, including auth failures and error codes.

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
