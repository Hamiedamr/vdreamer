"""Milestone 10: zero-egress settings, ``dev --reload``, the checkpointer check, the store."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph

from vdreamer import BaseAgent, BuildContext, Mode
from vdreamer.offline import OFFLINE_ENV, apply_offline_env, tiktoken_cache_problem
from vdreamer.reload import changes, child_argv, snapshot, supervise, watch_roots
from vdreamer.validate import validate_agent

# -- zero-egress runtime settings (§5.2) -------------------------------------------------


def test_offline_env_is_forced_not_defaulted() -> None:
    env = {"LANGSMITH_TRACING": "true", "HF_HUB_OFFLINE": "0", "OTHER": "x"}
    apply_offline_env(env)
    assert env["LANGSMITH_TRACING"] == "false" and env["HF_HUB_OFFLINE"] == "1"
    assert env["LANGCHAIN_TRACING_V2"] == "false" and env["DO_NOT_TRACK"] == "1"
    assert env["OTHER"] == "x"


def test_every_command_runs_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vdreamer.cli import main

    for key in OFFLINE_ENV:
        monkeypatch.setenv(key, "tampered")
    assert main(["export-schema", "--out", str(tmp_path)]) == 0
    assert {key: os.environ[key] for key in OFFLINE_ENV} == OFFLINE_ENV


def test_the_tiktoken_cache_check(tmp_path: Path) -> None:
    assert "is not a directory" in str(
        tiktoken_cache_problem({"TIKTOKEN_CACHE_DIR": str(tmp_path / "missing")})
    )
    assert "is empty" in str(tiktoken_cache_problem({"TIKTOKEN_CACHE_DIR": str(tmp_path)}))
    (tmp_path / "9b5ad71b2ce5302211f9c61530b329a4922fc6a4").write_bytes(b"encoding")
    assert tiktoken_cache_problem({"TIKTOKEN_CACHE_DIR": str(tmp_path)}) is None
    unset = tiktoken_cache_problem({})
    assert unset is None or "is unset" in unset  # a problem only when tiktoken is installed


def test_doctor_fails_on_a_shipped_but_empty_tiktoken_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from vdreamer.doctor import run_offline

    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    assert run_offline() == 1
    assert "[FAIL] tiktoken cache:" in capsys.readouterr().out
    (tmp_path / "encoding").write_bytes(b"x")
    assert run_offline() == 0
    assert f"[ok] tiktoken cache: {tmp_path}" in capsys.readouterr().out


# -- dev --reload --------------------------------------------------------------------------


def test_snapshot_and_changes(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("a = 1\n")
    (tmp_path / "models.yaml").write_text("aliases: {}\n")
    (tmp_path / "data.db").write_bytes(b"not watched")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("ignored\n")
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "x.py").write_text("ignored by request\n")
    before = snapshot([tmp_path], ignored=[tmp_path / "state"])
    assert sorted(Path(path).name for path in before) == ["agent.py", "models.yaml"]

    agent = tmp_path / "agent.py"
    os.utime(agent, ns=(before[str(agent.resolve())] + 10**9,) * 2)
    (tmp_path / "new.py").write_text("b = 2\n")
    (tmp_path / "models.yaml").unlink()
    after = snapshot([tmp_path], ignored=[tmp_path / "state"])
    assert [Path(path).name for path in changes(before, after)] == [
        "agent.py", "models.yaml", "new.py"
    ]


def test_watch_roots_and_child_argv(tmp_path: Path) -> None:
    inner = tmp_path / "agents"
    inner.mkdir()
    other = tmp_path.parent / f"{tmp_path.name}-cfg"
    other.mkdir()
    roots = watch_roots(
        cwd=tmp_path,
        agents_dir=str(inner),  # inside cwd: not watched twice
        files=(str(other / "models.yaml"), None),
        pythonpath=str(inner),
    )
    assert roots == [tmp_path.resolve(), other.resolve()]
    argv = child_argv(["dev", "--agent", "a:b", "--reload", "--port", "9"])
    assert argv[1:] == ["-m", "vdreamer", "dev", "--agent", "a:b", "--port", "9"]


class _Proc:
    def __init__(self, exit_after: int | None = None) -> None:
        self.polls = 0
        self.exit_after = exit_after
        self.terminated = False

    def poll(self) -> int | None:
        self.polls += 1
        if self.terminated:
            return -15
        if self.exit_after is not None and self.polls > self.exit_after:
            return 3
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return -15

    def kill(self) -> None:
        self.terminated = True


def test_supervise_restarts_on_change_and_returns_the_child_code(tmp_path: Path) -> None:
    source = tmp_path / "agent.py"
    source.write_text("v = 1\n")
    spawned: list[_Proc] = []
    ticks = {"n": 0}

    def spawn(command: list[str]) -> _Proc:
        process = _Proc(exit_after=2 if len(spawned) == 1 else None)
        spawned.append(process)
        return process

    def sleep(seconds: float) -> None:
        ticks["n"] += 1
        if ticks["n"] == 2:  # the developer saves the file
            stamp = source.stat().st_mtime_ns + 10**9
            os.utime(source, ns=(stamp, stamp))

    code = supervise(["vdreamer", "dev"], [tmp_path], spawn=spawn, sleep=sleep, interval_s=0)
    assert len(spawned) == 2 and spawned[0].terminated
    assert code == 3  # the second child exited on its own


def test_dev_reload_refuses_input(capsys: pytest.CaptureFixture[str]) -> None:
    from vdreamer.cli import main

    with pytest.raises(SystemExit):
        main(["dev", "--agent", "a:b", "--reload", "--input", "{}"])
    assert "--reload" in capsys.readouterr().err


# -- validate: the vdreamer checkpointer is used ------------------------------------------


class S(TypedDict, total=False):
    n: int


def _graph() -> StateGraph[Any]:
    graph: StateGraph[Any] = StateGraph(S)
    graph.add_node("one", lambda state: {"n": 1})
    graph.add_edge(START, "one")
    graph.add_edge("one", END)
    return graph


class _Agent(BaseAgent):
    name = "cp"
    mode = Mode.CHAT
    how = "stategraph"

    def build(self, ctx: BuildContext) -> Any:
        if self.how == "stategraph":
            return _graph()
        if self.how == "ctx":
            return _graph().compile(**{k: v for k, v in ctx.agent_kwargs().items()
                                       if k in ("checkpointer", "store")})
        if self.how == "none":
            return _graph().compile()
        if self.how == "own":
            return _graph().compile(checkpointer=InMemorySaver())
        from langgraph.store.memory import InMemoryStore

        return _graph().compile(checkpointer=ctx.checkpointer, store=InMemoryStore())


@pytest.mark.parametrize(
    ("how", "ok", "detail"),
    [
        ("stategraph", True, "StateGraph compiled by vdreamer"),
        ("ctx", True, "ctx.checkpointer"),
        ("none", False, "without a checkpointer"),
        ("own", False, "its own InMemorySaver"),
        ("own_store", False, "its own InMemoryStore store"),
    ],
)
def test_validate_requires_the_vdreamer_checkpointer(how: str, ok: bool, detail: str) -> None:
    agent = _Agent()
    agent.how = how
    (check,) = [c for c in validate_agent(agent) if c.name == "checkpointer"]
    assert check.ok is ok and detail in check.detail


# -- the long-term memory store (§2.4) ----------------------------------------------------


async def test_the_runner_compiles_with_the_store() -> None:
    from langgraph.store.memory import InMemoryStore

    from vdreamer.runtime import AgentRunner

    seen: list[Any] = []

    class StoreAgent(BaseAgent):
        name = "store"
        mode = Mode.CHAT

        def build(self, ctx: BuildContext) -> Any:
            def remember(state: S) -> dict[str, Any]:
                store = get_store()
                store.put(("notes",), "k", {"v": 1})
                seen.append(store)
                return {"n": 1}

            graph: StateGraph[Any] = StateGraph(S)
            graph.add_node("remember", remember)
            graph.add_edge(START, "remember")
            graph.add_edge("remember", END)
            return graph

    store = InMemoryStore()
    runner = AgentRunner(StoreAgent(), checkpointer=InMemorySaver(), store=store)
    compiled = runner.build()
    await compiled.ainvoke({}, {"configurable": {"thread_id": "t"}})
    assert seen == [store] and store.get(("notes",), "k") is not None


async def test_the_dev_store_persists_on_the_dev_file(tmp_path: Path) -> None:
    from vdreamer.entrypoints import _open_dev_store

    db = str(tmp_path / "dev.db")
    async with _open_dev_store(db) as store:
        await store.aput(("users", "alice"), "prefs", {"lang": "ar"})
    async with _open_dev_store(db) as store:
        item = await store.aget(("users", "alice"), "prefs")
    assert item is not None and item.value == {"lang": "ar"}


async def test_open_store_runs_setup_before_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    from vdreamer.persistence import open_store

    events: list[str] = []

    class _FakeStore:
        @classmethod
        @asynccontextmanager
        async def from_conn_string(cls, dsn: str) -> AsyncGenerator[Any]:
            events.append(f"connect:{dsn}")
            yield cls()

        async def setup(self) -> None:
            events.append("setup")

    monkeypatch.setattr("vdreamer.persistence._load_store_class", lambda: _FakeStore)
    async with open_store("postgresql://x") as store:
        assert isinstance(store, _FakeStore)
        assert events == ["connect:postgresql://x", "setup"]


def test_the_store_class_is_langgraphs() -> None:
    from langgraph.store.postgres.aio import AsyncPostgresStore

    from vdreamer.persistence import _load_store_class

    assert _load_store_class() is AsyncPostgresStore
