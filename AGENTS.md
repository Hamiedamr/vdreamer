# AGENTS.md

## Repository state

- Milestones 0–9 landed (`docs/tasks.md` tracks each item and what it defers); `docs/PRD.md` is the entire product spec and the contract for everything built here. `src/vdreamer` holds:
  - **M1 developer contract:** `agents.py` (`BaseAgent`/`BuildContext`, `agent_kwargs()`, `tool_policies`, `interrupt_policies`, `durability`), `context.py` (run context + implicit helpers), `runtime.py` (`AgentRunner`), `tools.py` (policies, `@vdreamer.tool`, pass-through `VDreamerToolMiddleware`), `discovery.py`, `persistence.py`, `validate.py`, `doctor.py`, `cli.py`.
  - **M2 protocol + event pipeline:** `events.py` (envelope + payload models), `streaming.py` (the one adapter module, v2/v3 drivers, A31), `eventlog.py` (sinks, `run_events`, and since M4 the replay reader, `EventHub`, `NOTIFY`/`LISTEN`).
  - **M3 runs + topology:** `queue.py` (`runs` table, §3.2 lifecycle, leases), `worker.py`, `entrypoints.py` (`worker`, `dev`, and since M4 `serve`), `migrate.py`.
  - **M4 REST gateway:** `gateway/` (FastAPI app, D20; never imported by `import vdreamer`), `threads.py`, `interrupts.py` (§4.2 record), `artifacts.py` (read side; the write side is M6), `metrics.py`, `protocol_schema.py`, `vendored.py` (sha256-pinned Swagger UI under `gateway/static/vendor/`).
  - **M5 tool governance + MCP:** `governance.py` (the §2.7 seven-step pipeline, `GovernedTool` proxy, `ToolKit` = `ctx.tools`; `tools.VDreamerToolMiddleware` runs the same pipeline), `mcp.py` (the one `langchain.mcp`/FastMCP module: config + A40 rules, `MCPRegistry`), `_mcp_launch.py` (POSIX rlimit/env launcher for stdio servers), `resources.py` (`resources.yaml` loading).
  - **M6 artifacts, files, sandbox:** `artifacts.py` (`emit_artifact`/`aemit_artifact`/`Artifact`, the D7 write order, record stores incl. Postgres), `artifact_data.py` (per-kind encoding + previews, no auto-detection), `objects.py` (object stores: in-memory, dev filesystem, MinIO; `artifacts/` `inputs/` `extracted/` keys), `tables.py` (Parquet write/page/sort, CSV/XLSX export), `files.py` (`ctx.files`, `safe_extract` → `Manifest`), `_extract.py` (the extraction engine, run as a sandboxed child), `sandbox.py` + `_sandbox_launch.py` (`ctx.sandbox`, the POSIX launcher).
  - **M7 models + resources:** `models.py` (`models.yaml`: `VllmChatModel`/`OllamaChatModel` with the per-alias thinking mode, fastembed loading, endpoint probes, `AliasMap`), `adapters.py` (`ResourceAdapter` + `register_adapter`; qdrant, neo4j, sql; the read-only guards), `backends.py` (load both files, check an agent's aliases, open/close its models and resources), `network.py` (`NETWORK_ALLOWLIST`, shared with MCP).
  - **M9 headless trigger + outbox:** `gateway/routes_trigger.py` (`POST /agents/{name}/trigger`: `Idempotency-Key` → a derived run id, A18; `callback` checked against `CALLBACK_ALLOWLIST`, A26), service tokens in `gateway/auth.py`/`settings.py` (`VDREAMER_SERVICE_TOKENS`, A16), `outbox.py` (`OutboxEventSink` records callback/webhook/email entries as events are written; `OutboxDispatcher` delivers them with leases, backoff and dead-lettering; HMAC signing; stores: in-memory, SQLite, Postgres). `RunRecord.callback`/`.headless` are the `/trigger` run columns.
  - **M8 HITL:** `approval.py` (`ApprovalGate`, the prebuilt `prepare → gate → {execute, on_reject, replan}` subgraph), `interrupts.py` (plus `interrupt_policies` application, four-eyes matching, `end_waiting_run`, the leader-elected `InterruptSweeper`), `gateway/routes_interrupts.py` (per-source resume values, `edit_schema` validation via `jsonschema`, `HITL_STRICT_SCOPES`, headless `reject → rejected`), `streaming.normalize_interrupt` (`HumanInTheLoopMiddleware` → `tool_review`).
- Tests: `tests/` per area — `contract/` (M1 exit), `golden/` (M2 v2↔v3 parity), `worker/` (M3 exit), `api/` (M4 exit: `uv run pytest tests/api`), `tools/` (M5 exit: governance + MCP against a real FastMCP fixture server), `artifacts/` (M6 exit: kinds, D7 order, the `safe_extract` corpus, object stores, the POSIX sandbox; plus `api/test_artifacts_e2e.py` / `api/test_files_api.py`), `models/` (M7 exit: config, chat models against vLLM/Ollama stub servers in `_stub_llm.py`, thinking through both drivers, fastembed, adapters), `hitl/` + `api/test_hitl_m8.py` (M8 exit: all four interrupt sources through one endpoint; agents in `api/_hitl_agents.py`), `outbox/` + `api/test_trigger_m9.py` (M9 exit: idempotent replay, callbacks delivered at least once with dedup; the API kit wires the outbox, a service token and a callback allowlist like `vdreamer dev`). Sandbox tests skip on Windows; `VDREAMER_TEST_SANDBOX_UID` (as root) adds the dedicated-uid checks; `VDREAMER_TEST_MINIO_ENDPOINT` (+ `_ACCESS_KEY`/`_SECRET_KEY`) runs the live object-store test; `VDREAMER_TEST_QDRANT_URL`, `VDREAMER_TEST_NEO4J_URI` (+ `_USER`/`_PASSWORD`), `VDREAMER_TEST_FASTEMBED_CACHE` run the live adapter/model tests. Live-Postgres tests skip unless `VDREAMER_TEST_PG_DSN` is set. `tests/` has no `__init__.py`: shared kits are `_kit.py`/`_api.py`, and test module basenames must be unique across directories.
- Not built yet (M10+): `export-schema`/`bundle`, `vdreamer run`, the UC1–UC3 reference agents. Never assume an implementation exists — check `docs/tasks.md`.
- The PRD is the contract for everything built here. Read it before designing or writing code; when it conflicts with intuition, the PRD wins.
- PRD map: §2 architecture/LangGraph compat/tool governance, §3 events & artifacts, §4 HITL, §5 air-gapped packaging & zero-egress, §7 scope + non-goals. Cross-refs like `A41`/`D11` are assumption/decision IDs; §7.4 defines A24–A44.

## What this is

A Python SDK + CLI (`vdreamer`) that serves developer-written `BaseAgent` subclasses (LangGraph graphs) as a production REST/SSE API with zero generated server code, for air-gapped enterprise deployment. Python ≥3.11.

Users are AI developers only. vdreamer ships no frontend code. Client apps integrate against the protocol contract (PRD §6); its deliverables are schemas plus a conformance kit, not a client SDK.

## Hard constraints — violating these means rework

- **Version pins (A41):** `langchain>=1.4,<2` (needed for `langchain.mcp`), `langgraph>=1.2,<2`, `langgraph-checkpoint-postgres`. A 2.0 of either dependency requires a vdreamer major. CI must run the reference agents against both the pinned floor and the latest 1.x minor.
- **Design rule:** core ships generic primitives only; domain logic (retrieval, SQL semantics, validation rules) lives in agent code. A capability enters core only if ≥2 reference agents need it and it can't be built on existing primitives.
- **Streaming isolation (A31):** all LangGraph streaming specifics live in one adapter module — two drivers (`v2` default, `v3` opt-in via `VDREAMER_STREAM_DRIVER`) emitting identical protocol events, verified by golden-event tests. Don't spread stream handling elsewhere.
- **Persistence:** `durability="sync"` by default (A42) — crash recovery depends on every step being checkpointed. `async` is a per-agent opt-in.
- **Topology:** gateway ↔ worker communication is Postgres only (queue rows + `LISTEN/NOTIFY`, A7); only workers reach model servers, MinIO writes, and agent resources. The gateway is stateless (N replicas).
- **Tools are plain LangChain `BaseTool`s (A37):** no vdreamer base class. Governance is a proxy/middleware that preserves each tool's name and schema.
- **Protocol version `1.0`:** event envelope field `"v"` and the `X-Protocol-Version` header.
- **Secrets never appear in YAML:** config references env vars via `*_env` keys only (A23).
- **Zero egress is mandatory (§5.2):** no CDNs, telemetry, or update checks; `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`; fastembed `local_files_only`; vendored chart JS. `doctor --offline` runs in CI with egress blocked and must assert zero egress.

## Tooling

- **uv is the only package manager for this repo.** Use `uv add` / `uv remove` for dependencies, keep `uv.lock` committed, and run everything through `uv run` (tests, lint, the CLI). Never `pip install` into the dev environment, hand-edit `requirements*.txt`, or add `setup.py`.
- **vdreamer itself must never depend on uv.** Runtime code, the CLI, `bundle` output, and install docs must work with plain `pip`. A CI job installs the built wheel without uv to enforce this.
- **Standard commands (Milestone 0+).** Dependencies change only via `uv add` / `uv add --optional <extra>` / `uv remove`; `uv.lock` is committed and CI fails on drift (`uv lock --check`).
  - `uv sync --locked --all-extras` — dev environment from the lockfile
  - `uv run pytest` — tests (`-m smoke` is the subset CI runs against the floor/latest pins)
  - `uv run ruff check .` — lint (`.agents/` is excluded: installed skills are reference material)
  - `uv run mypy` — type-check (`src/`, strict)
  - `uv run vdreamer …` — the CLI (`--version`, `validate pkg.mod:agent`, `doctor --offline`)
  - CI is `.github/workflows/ci.yml`: locked / floor (`uv lock --resolution lowest-direct`) / latest (`uv lock --upgrade`) / plain-`pip` wheel install / zero-egress `doctor --offline`. Floor and latest regenerate the lock locally (never committed) and sync with `--frozen`.
- **Installed agent skills** (`.claude/skills/`, `.agents/skills/`) are reference material. When a skill conflicts with `docs/PRD.md` (for example, it suggests LangSmith, LangGraph Platform, `langgraph.json`, or `npx`/`uvx` launchers), the PRD wins.

## Layout (from the PRD; ✓ = exists today)

- One OCI image, two entrypoints: `vdreamer serve` (gateway ✓) and `vdreamer worker` (✓). `vdreamer dev` (✓) runs gateway + worker in one process with every store (runs, events, threads, interrupts, checkpoints) in one SQLite `--db` file (`sqlite_state.py`, D11); `serve`/`worker` use Postgres (✓) + MinIO (✓, `VDREAMER_MINIO_*` env; without it artifacts are unavailable). Dev objects live next to the `.db`: `artifacts/`, `inputs/`, `extracted/`.
- **Testing phase is SQLite-first (decided 2026-10-14):** developers test on `vdreamer dev` + SQLite; Postgres comes back after that testing. Keep the Postgres stores compiling and covered by their offline tests, but don't treat a live-Postgres run as a blocker. New stores need an in-memory twin (and SQLite persistence for dev) behind the same ABC.
- CLI surface: `dev` ✓, `serve` ✓, `worker` ✓, `run` (terminal client, M11), `validate` ✓, `migrate` ✓, `export-schema` (also emits the conformance kit, M10), `doctor --offline` ✓, `bundle` (M10).
- REST under `/api/v1` ✓; SSE with `Last-Event-ID` replay ✓; artifact bytes in MinIO (✓), state/metadata in Postgres.
- `resources.yaml` comes from `--resources` or `$VDREAMER_RESOURCES` (`type: mcp` → MCP servers, every other type → a resource adapter); `models.yaml` from `--models` or `$VDREAMER_MODELS`. Agents get only their declared aliases (`ctx.models` / `ctx.resources`), opened once at worker boot by `vdreamer.backends`; a failure there stops the worker.
- Thinking modes (`reasoning_content`, `think_tags`) are applied inside the chat model (`vdreamer.models`), not in the stream adapter: chunks reach the drivers as standard `reasoning` blocks. Keep it that way so v2/v3 parity holds.
- Artifacts go through `emit_artifact`/`aemit_artifact`/`return Artifact(...)` (or the internal `write_artifact`) only, so the D7 order holds: bytes → row → event. Object keys pass `objects.check_key`; `ctx.files` reads only `inputs/` and the run's own `extracted/` prefix.
- Sandbox and extraction children start through `_sandbox_launch.py` (a fresh process applies the limits, then `execve`s); never use `preexec_fn` in the threaded worker, and never fall back to running sandbox code unconfined.
- Interrupt resume values are built per source by the gateway (§4.5): `{action, payload?, text?}` for `require_approval`, the middleware's own `{"decisions": [{"type": …}]}` for `tool_review`, `{"responses": …}` for MCP elicitation, the payload verbatim for raw `interrupt()`. A headless run is `agents.runs_headless(agent, run.headless)`: every run of a `HEADLESS` agent and every `/trigger` run; its approval `reject` ends the run `rejected` in the gateway without resuming. Runs no worker holds (`awaiting_approval`) are ended through `interrupts.end_waiting_run`, which also writes the final `run.status`.
- Governed tools are the single emitter of `tool.call`/`tool.result`: their records ride the custom stream and the mapper drops its ungoverned mapping for those ids. Don't emit tool events anywhere else.
- Gateway config is env-only (A23): `VDREAMER_PG_DSN`, `VDREAMER_INTERNAL_TOKEN` (+ `VDREAMER_INTERNAL_TOKEN_PREVIOUS` for rotation, A14), `HITL_STRICT_SCOPES` (four-eyes `fnmatch` patterns, A32), `VDREAMER_SERVICE_TOKENS` (A16), `CALLBACK_ALLOWLIST` (A26), and the outbox's `VDREAMER_OUTBOX_WEBHOOK_URL`/`VDREAMER_OUTBOX_SECRET`/`WEB_BASE_URL`/`VDREAMER_SMTP_*`. `--host` refuses wildcard binds.
- Everything that leaves the platform (callbacks, interrupt webhooks/emails) goes through the outbox: wrap any new event writer in `OutboxEventSink`, and never POST to a callback directly. Outbox entries are added before the event is written (a crash duplicates, never drops); receivers dedup on `event_id`.
- `examples/` holds reference agents UC1–UC3. They are the acceptance test: core is P0-complete only when they build **without touching vdreamer internals** (§7.3).
- Python extras: `[postgres]`, `[minio]`, `[fastembed]`, `[qdrant]`, `[neo4j]`, `[sql]` (SQLAlchemy; bring your own async driver), `[mcp]`, `[sandbox]`. `langchain-openai` (vLLM) and `langchain-ollama` are base dependencies.

## Non-goals — do not build these (PRD §7.2)

LLM serving/fine-tuning; end-user auth/LDAP; mail fetching; cron; retrieval pipelines, chunking, ingestion, or vector-schema management in core; Neo4j semantic-layer sync; MCP prompts/resources/sampling or exposing vdreamer agents as MCP servers; generated server code; LangGraph Platform API compatibility; non-LangChain frameworks; cloud model providers; any frontend code (client apps, UI components, client SDKs, Angular or otherwise).

## Gotchas

- The reference topology is Linux VMs/containers (§5.1). Sandbox primitives (dedicated uid, rlimits, namespaces — D19/A19) are Unix-only and cannot be exercised on a Windows dev host; run them on Linux CI.
- `docs/.PRD.md.un~` is a vim undo artifact, not content — ignore it.
- psycopg 3 cannot adapt a bare `dict` (and sends a `list` as an array): pass `json.dumps(...)` with a `%s::jsonb` cast, as `queue.py`/`eventlog.py` do. Offline fakes don't catch this — add a live-PG test.
- Vendored assets are sha256-pinned and marked `-text` in `.gitattributes`; after changing one, update `vendored.VENDORED_ASSETS` or `doctor --offline` fails.
- Build/test commands are recorded in **Tooling** above (landed with Milestone 0).
