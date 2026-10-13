# vdreamer

A Python SDK + CLI that serves developer-written `BaseAgent` subclasses — LangGraph graphs — as a production REST/SSE API with zero generated server code, for air-gapped enterprise deployment. Fully compatible with LangChain/LangGraph 1.x. Users are AI developers only: vdreamer ships no frontend code, and client apps integrate against the protocol contract (PRD §6). Core ships generic primitives only — domain logic (retrieval, SQL semantics, validation rules) lives in agent code.

## Status

Pre-1.0 (`0.1.0`). Milestone 0 (Repo Bootstrap & CI Foundation) is **complete**; Milestone 1 (Developer Contract Core) is **in progress** (`docs/tasks.md`).

Today the repo holds the package skeleton: the CLI (`--version`, `validate`, `doctor --offline`), the first `BaseAgent` contract slice, agent discovery, tests, and CI. Anything past Milestone 0 — runs/queue, the REST/SSE gateway, HITL, artifacts, tool governance/MCP, models/resources — does not exist yet.

## Development quickstart

Prerequisites: Python >= 3.11, and uv (the package manager for this repo; vdreamer itself never depends on it).

```
uv sync --locked --all-extras          # dev environment from the committed uv.lock
uv run pytest                          # full test suite
uv run pytest -m smoke                 # the subset CI runs against the floor and latest 1.x pins (A41)
uv run ruff check .                    # lint (.agents/ excluded: installed skills are reference material)
uv run mypy                            # type-check (src/, strict)
uv run vdreamer --version
uv run vdreamer validate pinger:agent # smoke agent; needs examples/smoke on PYTHONPATH
uv run vdreamer doctor --offline      # asserts zero egress
```

For `validate pinger:agent`, make `examples/smoke/` importable first: bash `PYTHONPATH=examples/smoke uv run vdreamer validate pinger:agent`, or in PowerShell `$env:PYTHONPATH = "examples/smoke"`. CI does the same for the plain-pip job.

## CI matrix

`.github/workflows/ci.yml` runs five jobs on Python 3.11 with `UV_PYTHON_DOWNLOADS=never`:

- `locked` — `uv lock --check` fails on lockfile drift; `uv sync --locked --extra postgres` proves the Milestone-0 exit criterion; then ruff, mypy, and the full test suite against the committed lockfile.
- `floor` — `uv lock --resolution lowest-direct` (regenerated locally, never committed): smoke tests against the minimum supported pins (A41: langchain 1.4.0, langgraph 1.2.11).
- `latest` — `uv lock --upgrade` (regenerated locally, never committed): smoke tests against the latest 1.x minor, so new minors are tracked, never silently absorbed.
- `uv-independence` (plain-pip install) — `uv build` a wheel, install it into a clean `python -m venv` with plain `pip`, then run the CLI and validate a sample agent: proves vdreamer never requires uv at runtime.
- `zero-egress` — `uv run --frozen --offline vdreamer doctor --offline` asserts zero egress via an in-process socket guard (PRD §5.2).

## Repository layout

- `src/vdreamer/` — the package: `cli`, `agents` (`BaseAgent` contract slice), `discovery`, `validate`, `doctor`.
- `tests/` — pytest suite: CLI, contract, validate, doctor.
- `examples/smoke/` — `pinger.py`, the Milestone-0 smoke agent the CI matrix runs (the UC1–UC3 reference agents land with Milestone 12).
- `docs/` — `PRD.md` (the spec), `tasks.md` (the milestone plan).
- Root: `AGENTS.md`, `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`.