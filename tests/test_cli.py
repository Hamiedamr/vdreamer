"""CLI entry-point tests."""

import subprocess
import sys
from importlib import metadata as importlib_metadata

import pytest

import vdreamer
from vdreamer.cli import build_parser


def test_version_flag_prints_package_version() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "vdreamer", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == f"vdreamer {vdreamer.__version__}"


def test_console_script_is_registered() -> None:
    # The `vdreamer` console script is the product's single entry point;
    # the uv-independence CI job runs it from a plain `python -m venv` install.
    entry_points = importlib_metadata.entry_points(group="console_scripts")
    assert any(ep.name == "vdreamer" and ep.value.startswith("vdreamer") for ep in entry_points)


# ---------------------------------------------------------------------------
# Milestone 3 subcommands: migrate / worker / dev
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected_fragment"),
    [
        (["migrate", "--help"], "VDREAMER_PG_DSN"),
        (["worker", "--help"], "--claimant"),
        (["dev", "--help"], "--input"),
    ],
)
def test_m3_subcommand_help_renders(argv: list[str], expected_fragment: str, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(argv)
    assert excinfo.value.code == 0
    assert expected_fragment in capsys.readouterr().out


def test_worker_and_dev_require_agent() -> None:
    for command in ("worker", "dev"):
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args([command])
        assert excinfo.value.code == 2


def test_migrate_fails_friendly_without_a_dsn(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from vdreamer.migrate import run_migrate

    monkeypatch.delenv("VDREAMER_PG_DSN", raising=False)
    assert run_migrate() == 1
    captured = capsys.readouterr()
    assert "VDREAMER_PG_DSN is not set" in captured.err


def test_worker_fails_friendly_without_a_dsn(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from vdreamer.entrypoints import run_worker

    monkeypatch.delenv("VDREAMER_PG_DSN", raising=False)
    assert run_worker(agent="pinger:agent") == 1
    captured = capsys.readouterr()
    assert "VDREAMER_PG_DSN is not set" in captured.err


def test_migrate_fails_friendly_without_psycopg(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from vdreamer import MissingPostgresDependencyError
    from vdreamer import queue as queue_module
    from vdreamer.migrate import run_migrate

    def _raise() -> None:
        raise MissingPostgresDependencyError("psycopg is required (test seam)")

    monkeypatch.setenv("VDREAMER_PG_DSN", "postgresql://vd:vd@localhost:5432/vd")
    monkeypatch.setattr(queue_module, "_load_async_connection_class", _raise)
    assert run_migrate() == 1
    captured = capsys.readouterr()
    assert "psycopg" in captured.err


def test_dev_input_smoke_runs_the_whole_lifecycle(capsys, tmp_path) -> None:
    # The real D11 path: the agent's checkpoints land on the SQLite --db file
    # (langgraph-checkpoint-sqlite is a base dependency) — no fallback, no warning.
    from vdreamer.entrypoints import run_dev

    rc = run_dev(
        agent="pinger:agent",
        db=str(tmp_path / "dev.db"),
        input_json='{"message": "ping"}',
    )
    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out
    assert "queued run_" in out
    assert "-> completed (attempt 1," in out
    assert "checkpoints=sqlite" in out, "dev runs on SQLite (D11)"
    assert "WARNING" not in captured.err, "a fresh checkout lands on SQLite with zero warning"


def test_dev_input_prints_the_events(capsys, tmp_path) -> None:
    from vdreamer.entrypoints import run_dev

    assert run_dev(agent="pinger:agent", db=str(tmp_path / "dev.db"),
                   input_json='{"message": "ping"}') == 0
    out = capsys.readouterr().out
    assert "[run running]" in out and "pong: ping" in out and "[run completed]" in out


def test_dev_input_is_checked_against_the_input_schema(capsys, tmp_path) -> None:
    # The gateway answers 422 invalid_input; dev says why before running anything.
    from vdreamer.entrypoints import run_dev

    assert run_dev(agent="pinger:agent", db=str(tmp_path / "dev.db"),
                   input_json='{"message": 7}') == 1
    captured = capsys.readouterr()
    assert "does not match the input_schema of agent 'pinger'" in captured.err
    assert "message:" in captured.err
    assert "queued" not in captured.out


BOOMER = """
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from vdreamer import BaseAgent, Mode


class S(TypedDict, total=False):
    messages: list


def boom(state):
    raise ValueError("node blew up")


class Boomer(BaseAgent):
    name = "boomer"
    mode = Mode.CHAT

    def build(self, ctx):
        graph = StateGraph(S)
        graph.add_node("boom", boom)
        graph.add_edge(START, "boom")
        graph.add_edge("boom", END)
        return graph


agent = Boomer()
"""

SCHEMALESS = """
from vdreamer import BaseAgent, Mode


class Schemaless(BaseAgent):
    name = "schemaless"
    mode = Mode.HEADLESS

    def build(self, ctx):
        raise AssertionError("never built")


agent = Schemaless()
"""


def test_dev_input_reports_why_a_run_failed(capsys, tmp_path, monkeypatch) -> None:
    from vdreamer.entrypoints import run_dev

    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "boomer.py").write_text(BOOMER, encoding="utf-8")
    assert run_dev(agent="boomer:agent", db=str(tmp_path / "dev.db"),
                   input_json='{"messages": []}') == 1
    captured = capsys.readouterr()
    assert "-> failed" in captured.out
    assert "node_error: node blew up" in captured.err


def test_entrypoints_refuse_an_agent_validate_rejects(capsys, tmp_path, monkeypatch) -> None:
    from vdreamer.entrypoints import run_dev, run_worker

    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "schemaless.py").write_text(SCHEMALESS, encoding="utf-8")
    assert run_dev(agent="schemaless:agent", db=str(tmp_path / "dev.db"),
                   input_json="{}") == 1
    err = capsys.readouterr().err
    assert "fails the contract" in err and "input_schema: required for HEADLESS/BOTH" in err
    assert "vdreamer validate" in err
    assert run_worker(agent="schemaless:agent") == 1
    assert "fails the contract" in capsys.readouterr().err


def test_dev_persists_the_run_checkpoint_to_the_sqlite_file(tmp_path) -> None:
    # D11's point: run state survives a dev restart. `dev --input` checkpoints
    # the run to the SQLite file; a fresh saver re-opened on the same file
    # reads the graph's final writes back via `aget_state`. No sockets —
    # aiosqlite is in-process, so the suite stays socket-free for doctor
    # --offline's EgressGuard.
    import asyncio
    from typing import Any

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from pinger import agent as pinger_agent

    from vdreamer.entrypoints import run_dev


    db = str(tmp_path / "dev.db")
    assert run_dev(
        agent="pinger:agent",
        db=db,
        input_json='{"message": "ping"}',
        thread_id="t-dev",
    ) == 0

    async def reopen() -> Any:
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            compiled = pinger_agent.build(None).compile(checkpointer=saver)
            return await compiled.aget_state({"configurable": {"thread_id": "t-dev"}})

    snapshot = asyncio.run(reopen())
    assert snapshot.next == (), "the run checkpointed its completion on the thread"
    assert snapshot.values["reply"] == "pong: ping", (
        "the checkpointed state carries the graph's final writes"
    )
