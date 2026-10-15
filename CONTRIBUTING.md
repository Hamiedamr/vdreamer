# Contributing to vdreamer

This file is for people changing vdreamer itself. To *use* vdreamer in your project, see the [README](README.md).

`docs/PRD.md` is the product spec and the contract for everything here; `docs/tasks.md` tracks the milestones and records design decisions and gap reviews. `AGENTS.md` lists the hard constraints (version pins, streaming isolation, zero egress, uv-only tooling) for humans and coding agents alike.

## Development setup

Prerequisites: Python 3.11+ and [uv](https://docs.astral.sh/uv/). uv is the repo's package manager only — vdreamer itself never depends on it, and a CI job installs the wheel with plain `pip` to prove it.

```bash
uv sync --locked --all-extras     # dev environment from the committed uv.lock
uv run pytest                     # full test suite
uv run pytest -m smoke            # the subset the floor/latest CI jobs run
uv run ruff check .               # lint
uv run mypy                       # type-check (src/, strict)
uv run vdreamer doctor --offline  # asserts zero egress
```

Dependencies change only through `uv add` / `uv add --optional <extra>` / `uv remove`; `uv.lock` is committed and CI fails on drift (`uv lock --check`).

Test areas, by directory: `contract/` (the agent contract), `golden/` + `streaming/` + `protocol/` (event protocol, v2/v3 driver parity), `queue/` + `worker/` (runs), `api/` (REST/SSE gateway), `tools/` (governance, MCP), `artifacts/` (artifacts, `safe_extract`, sandbox — POSIX), `models/` (models and resources against stub servers), `hitl/` (approvals), `outbox/` (callbacks), `packaging/` (export-schema, bundle, offline env), `conformance/` + `client/` (the client contract, `vdreamer run`), `acceptance/` (the example agents end to end). Live-service tests skip unless their env var is set (`VDREAMER_TEST_PG_DSN`, `VDREAMER_TEST_MINIO_ENDPOINT`, `VDREAMER_TEST_QDRANT_URL`, `VDREAMER_TEST_NEO4J_URI`, `VDREAMER_TEST_FASTEMBED_CACHE`, `VDREAMER_TEST_OLLAMA=1`).

Protocol changes re-record the conformance kit in the same change: `uv run python tests/conformance/record_kit.py`.

## Examples

`examples/` holds agents that exercise vdreamer from the outside, using the public API only (`tests/acceptance` enforces it with an AST scan). Each is a uv workspace member:

- `examples/playground/` — first-contact agents and LangChain/LangGraph docs examples (`tests/acceptance/test_playground.py`).
- `examples/uc1_sql_analyst/`, `examples/uc2_site_validation/`, `examples/uc3_pdf_research/` — the reference agents: SQL analysis with a reviewed query and a four-eyes write path; headless site validation (ZIP → vision model → MCP → approved verdict → callback); multi-document research with hybrid retrieval and citations in English and Arabic.
- `examples/smoke/pinger.py` — the smoke agent CI runs.

To run one, make it importable (the workspace members are installed by `uv sync`; for `pinger`, set `PYTHONPATH=examples/smoke`), e.g. `cd examples/playground && uv run vdreamer dev --registry vdreamer.yaml --agent hello`.

## Repository layout

- `src/vdreamer/` — the package: the agent contract (`agents`, `context`, `tools`, `runtime`, `discovery`, `validate`), the event protocol (`events`, `streaming` — the one module with LangGraph streaming specifics, `eventlog`), runs (`queue`, `worker`, `entrypoints`, `migrate`), the gateway (`gateway/`, `threads`, `interrupts`, `metrics`), tool governance and MCP (`governance`, `mcp`, `resources`), artifacts/files/sandbox (`artifacts`, `objects`, `tables`, `files`, `sandbox`), models and resources (`models`, `adapters`, `backends`, `network`), HITL (`approval`), the outbox (`outbox`), packaging (`offline`, `export_schema`, `bundle`), the terminal client (`client`) and the conformance kit (`conformance/`).
- `tests/`, `examples/`, `docs/`, `Dockerfile`, `.github/workflows/ci.yml`.

## CI

`.github/workflows/ci.yml` runs on Python 3.11:

- `locked` — lockfile check, ruff, mypy and the full suite against the committed lock.
- `floor` / `latest` — smoke and acceptance tests against the minimum supported pins (`uv lock --resolution lowest-direct`) and the latest 1.x minors (`uv lock --upgrade`); neither lock is committed.
- `uv-independence` — builds the wheel and installs it with plain `pip` into a clean venv.
- `zero-egress` — `doctor --offline` with its socket guard.
- `airgap` — `vdreamer bundle`, then a `--network none` container installs it with `pip --no-index --require-hashes`.
- `image` — builds the `Dockerfile` and runs `doctor` and a smoke run inside it, network off.
- `release` — on `v*` tags, attaches the wheel, sdist, `constraints.txt` and the `export-schema` output to the GitHub release.
