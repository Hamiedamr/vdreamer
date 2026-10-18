"""Lesson 1: the echo agent — the serving contract, threads, and the event stream."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from _tutorial_kit import TUTORIALS, served, vdreamer_run
from langgraph.graph import StateGraph
from vdreamer_tutorials.t01_serving_an_agent.agent import EchoAgent, agent
from vdreamer_tutorials.testing import AgentSession

from vdreamer import BuildContext
from vdreamer.validate import validate_agent


async def test_the_agent_answers_with_progress_and_a_log() -> None:
    result = await AgentSession(agent).run("hello")

    assert result.status == "completed"
    assert result.answer == "You said: hello (message 1 in this thread)"
    assert result.types == [
        "run.status", "progress", "log", "token.delta", "message.completed", "run.status",
    ]


async def test_a_thread_keeps_its_history_and_threads_are_isolated() -> None:
    session = AgentSession(agent)
    await session.run("first", thread_id="a")
    second = await session.run("second", thread_id="a")
    other = await session.run("elsewhere", thread_id="b")

    assert second.answer == "You said: second (message 2 in this thread)"
    assert other.answer == "You said: elsewhere (message 1 in this thread)"
    saved = await session.state("a")
    assert [m.text for m in saved.values["messages"]] == [
        "first",
        "You said: first (message 1 in this thread)",
        "second",
        "You said: second (message 2 in this thread)",
    ]


def test_a_graph_compiled_without_the_checkpointer_is_refused() -> None:
    class SelfCompiled(EchoAgent):
        name = "t01-self-compiled"

        def build(self, ctx: BuildContext) -> Any:
            graph: StateGraph[Any] = super().build(ctx)
            return graph.compile()

    failed = {c.name: c.detail for c in validate_agent(SelfCompiled()) if not c.ok}
    assert "checkpointer" in failed, failed
    assert all(c.ok for c in validate_agent(agent))


def test_vdreamer_run_through_a_real_gateway(monkeypatch: Any) -> None:
    with served([agent]) as url:
        result = vdreamer_run(
            url, monkeypatch, agent="t01-echo", message="hello", once=True, color=False
        )
    assert result.code == 0, result.out
    assert "[progress] reading your message 50%" in result.out
    assert "You said: hello (message 1 in this thread)" in result.out
    assert "[run completed]" in result.out


def test_the_documented_one_shot_command(tmp_path: Path) -> None:
    command = [
        sys.executable, "-m", "vdreamer", "dev", "--registry", "vdreamer.yaml",
        "--agent", "t01-echo", "--db", str(tmp_path / "dev.db"),
        "--input", '{"messages": [{"role": "user", "content": "hello"}]}',
    ]
    env = {**os.environ, "VDREAMER_INTERNAL_TOKEN": "tutorial-token-0123456789"}
    done = subprocess.run(
        command, cwd=TUTORIALS, env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "You said: hello (message 1 in this thread)" in done.stdout
    assert "-> completed (attempt 1, 6 events)" in done.stdout
