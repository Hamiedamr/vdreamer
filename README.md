# vdreamer

A Python SDK + CLI that serves developer-written `BaseAgent` subclasses — LangGraph graphs — as a production REST/SSE API with zero generated server code, for air-gapped enterprise deployment. Fully compatible with LangChain/LangGraph 1.x. Users are AI developers only: vdreamer ships no frontend code, and client apps integrate against the protocol contract (PRD §6). Core ships generic primitives only — domain logic (retrieval, SQL semantics, validation rules) lives in agent code.

## Status

Pre-1.0 (`0.1.0`). Milestones 0–6 are **complete** (`docs/tasks.md`): the `BaseAgent` developer contract, protocol v1.0 with the v2/v3 stream drivers, the Postgres run queue with workers and crash recovery, and the FastAPI REST/SSE gateway under `/api/v1` (threads, runs, resumable SSE, the interrupt inbox and resolve endpoint, the artifact routes, internal auth, Prometheus metrics, OpenAPI 3.1, and Swagger UI from vendored assets), the governed tool pipeline + MCP, and artifacts, files and the sandbox (five artifact kinds, Parquet tables with paged previews and CSV/XLSX export, `FileRef` + `safe_extract`, `ctx.sandbox`). Still to come, per later milestones: models/resources, the rest of HITL, `/trigger` + the outbox, packaging, and the UC1–UC3 reference agents.

## Development quickstart

Prerequisites: Python >= 3.11, and uv (the package manager for this repo; vdreamer itself never depends on it).

```
uv sync --locked --all-extras          # dev environment from the committed uv.lock
uv run pytest                          # full test suite
uv run pytest tests/contract          # the Milestone-1 exit criterion (contract suite)
uv run pytest tests/api               # the Milestone-4 exit criterion (REST contract suite)
uv run pytest tests/artifacts         # Milestone 6: artifacts, safe_extract corpus, sandbox (POSIX)
uv run pytest -m smoke                 # the subset CI runs against the floor and latest 1.x pins (A41)
uv run ruff check .                    # lint (.agents/ excluded: installed skills are reference material)
uv run mypy                            # type-check (src/, strict)
uv run vdreamer --version
uv run vdreamer validate pinger:agent # smoke agent; needs examples/smoke on PYTHONPATH
uv run vdreamer doctor --offline      # asserts zero egress
```

To try the gateway locally with no infrastructure, run `vdreamer dev` without `--input`: it serves `/api/v1` (docs at `/api/v1/docs`) and runs the worker in the same process, keeping runs, events, threads, approvals and checkpoints in one SQLite file (`--db`, default `./.vdreamer/dev.db`), so a restart picks up where it stopped. The current testing phase runs on this mode; the Postgres topology below comes later. It uses `VDREAMER_INTERNAL_TOKEN`, or prints a one-off token:

```
PYTHONPATH=examples/smoke uv run vdreamer dev --agent pinger:agent --port 8000
```

Agent tools are governed (redaction, timeouts, error containment, oversized results stored as artifacts). MCP servers are declared in a `resources.yaml` passed with `--resources` (or `$VDREAMER_RESOURCES`), for example `netinv: {type: mcp, url: https://mcp.netinv.internal/mcp, bearer_env: NETINV_TOKEN}`; agents select their tools with `ctx.tools.mcp("netinv")`. Agents emit artifacts with `emit_artifact(name, ArtifactKind.TABLE, rows)` (or `return Artifact(...)` from a tool); table artifacts need the `[sandbox]` extra (pyarrow). Under `vdreamer dev`, objects live next to the `.db` file: artifact bytes in `artifacts/`, inbound files (what a `FileRef` names) in `inputs/`, and `safe_extract` output in `extracted/`. `serve`/`worker` keep them in MinIO (`VDREAMER_MINIO_ENDPOINT`, `VDREAMER_MINIO_ACCESS_KEY`, `VDREAMER_MINIO_SECRET_KEY`, optional `VDREAMER_MINIO_BUCKET`/`VDREAMER_MINIO_SECURE`; `vdreamer migrate` creates the bucket and its 30-day lifecycle rule). `ctx.sandbox.run(code)` is POSIX-only; set `VDREAMER_SANDBOX_UID` (worker running as root) for a dedicated sandbox uid.

In production, `vdreamer serve` (the gateway) and one `vdreamer worker --agent NAME` per agent share Postgres (`VDREAMER_PG_DSN`); run `vdreamer migrate` once first.

For `validate pinger:agent`, make `examples/smoke/` importable first: bash `PYTHONPATH=examples/smoke uv run vdreamer validate pinger:agent`, or in PowerShell `$env:PYTHONPATH = "examples/smoke"`. CI does the same for the plain-pip job.

## CI matrix

`.github/workflows/ci.yml` runs five jobs on Python 3.11 with `UV_PYTHON_DOWNLOADS=never`:

- `locked` — `uv lock --check` fails on lockfile drift; `uv sync --locked --extra postgres` proves the Milestone-0 exit criterion; then ruff, mypy, and the full test suite against the committed lockfile.
- `floor` — `uv lock --resolution lowest-direct` (regenerated locally, never committed): smoke tests against the minimum supported pins (A41: langchain 1.4.0, langgraph 1.2.11).
- `latest` — `uv lock --upgrade` (regenerated locally, never committed): smoke tests against the latest 1.x minor, so new minors are tracked, never silently absorbed.
- `uv-independence` (plain-pip install) — `uv build` a wheel, install it into a clean `python -m venv` with plain `pip`, then run the CLI and validate a sample agent: proves vdreamer never requires uv at runtime.
- `zero-egress` — `uv run --frozen --offline vdreamer doctor --offline` asserts zero egress via an in-process socket guard (PRD §5.2).

## Repository layout

- `src/vdreamer/` — the package: `cli`, `agents` (`BaseAgent` contract), `context` (run context + helpers), `runtime` (`AgentRunner`), `tools` (policies + `@vdreamer.tool`), `discovery`, `persistence`, `validate`, `doctor`; `events`/`streaming`/`eventlog` (protocol + drivers + event log); `queue`/`worker`/`entrypoints`/`migrate` (runs and processes); `gateway/` (the FastAPI REST/SSE app and its vendored static assets: Swagger UI, plotly.js), `threads`, `interrupts`, `metrics`, `protocol_schema`, `vendored`; `governance`/`mcp`/`resources` (tools); `artifacts`/`artifact_data`/`objects`/`tables`/`files`/`sandbox` (artifacts, files, sandbox).
- `tests/` — pytest suite per area: `contract/` (M1), `golden/` + `streaming/` + `protocol/` (M2), `queue/` + `worker/` (M3), `api/` (M4), `tools/` (M5), `artifacts/` (M6), plus CLI, validate, doctor.
- `examples/smoke/` — `pinger.py`, the Milestone-0 smoke agent the CI matrix runs (the UC1–UC3 reference agents land with Milestone 12).
- `docs/` — `PRD.md` (the spec), `tasks.md` (the milestone plan).
- Root: `AGENTS.md`, `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`.