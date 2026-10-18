"""Lesson 17: a memory test suite — persistence, retrieval, isolation, deletion, restarts."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from _tutorial_kit import TOKEN, TUTORIALS, served, vdreamer_run
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite.aio import AsyncSqliteStore
from vdreamer_tutorials.t17_testing_memory.agent import agent
from vdreamer_tutorials.testing import AgentSession

# -- retrieval: what was stored comes back, in order --------------------------------------------


async def test_retrieval_returns_what_was_stored_in_order() -> None:
    session = AgentSession(agent, store=InMemoryStore(), user_id="ada")
    for note in ("size 44", "camps in winter", "no down"):
        await session.run(f"remember forever: {note}")
    recalled = await session.run("recall")
    assert recalled.answer == (
        "This conversation: nothing. Always: ['size 44', 'camps in winter', 'no down']."
    )


# -- isolation: the matrix of who sees what --------------------------------------------------------


@pytest.mark.parametrize(
    ("user", "thread", "expected"),
    [
        ("ada", "t-1", "This conversation: ['blue tent']. Always: ['size 44']."),  # same both
        ("ada", "t-2", "This conversation: nothing. Always: ['size 44']."),  # same user
        ("lin", "t-3", "This conversation: nothing. Always: nothing."),  # another user
    ],
)
async def test_isolation_matrix(user: str, thread: str, expected: str) -> None:
    session = AgentSession(agent, store=InMemoryStore())
    await session.run("remember for now: blue tent", user_id="ada", thread_id="t-1")
    await session.run("remember forever: size 44", user_id="ada", thread_id="t-1")
    assert (await session.run("recall", user_id=user, thread_id=thread)).answer == expected


async def test_concurrent_users_do_not_mix() -> None:
    session = AgentSession(agent, store=InMemoryStore())
    users = [f"user{i}" for i in range(8)]
    await asyncio.gather(*(
        session.run(f"remember forever: note of {u}", user_id=u, thread_id=f"t-{u}")
        for u in users
    ))
    for u in users:
        answer = (await session.run("recall", user_id=u, thread_id=f"r-{u}")).answer
        assert answer == f"This conversation: nothing. Always: ['note of {u}']."


async def test_no_user_no_long_term_memory() -> None:
    session = AgentSession(agent, store=InMemoryStore(), user_id="")
    assert (await session.run("remember forever: x")).answer == (
        "Long-term memory needs a signed-in user."
    )


# -- deletion --------------------------------------------------------------------------------------


async def test_forgetting_is_complete_and_scoped() -> None:
    store = InMemoryStore()
    session = AgentSession(agent, store=store)
    await session.run("remember forever: a", user_id="ada", thread_id="1")
    await session.run("remember forever: b", user_id="ada", thread_id="1")
    await session.run("remember forever: c", user_id="lin", thread_id="2")
    assert (await session.run("forget forever", user_id="ada", thread_id="1")).answer == (
        "Forgot 2 notes."
    )
    assert await store.asearch(("memory-desk", "ada")) == []
    assert len(await store.asearch(("memory-desk", "lin"))) == 1


# -- persistence: durable backends, new processes --------------------------------------------------


async def test_both_memories_survive_reopening_the_database(tmp_path: Path) -> None:
    db = str(tmp_path / "memory.db")
    async with AsyncSqliteSaver.from_conn_string(db) as saver, \
            AsyncSqliteStore.from_conn_string(db) as store:
        await store.setup()
        first = AgentSession(agent, checkpointer=saver, store=store, user_id="ada")
        await first.run("remember for now: blue tent", thread_id="t-1")
        await first.run("remember forever: size 44", thread_id="t-1")

    async with AsyncSqliteSaver.from_conn_string(db) as saver, \
            AsyncSqliteStore.from_conn_string(db) as store:
        second = AgentSession(agent, checkpointer=saver, store=store, user_id="ada")
        answer = (await second.run("recall", thread_id="t-1")).answer
    assert answer == "This conversation: ['blue tent']. Always: ['size 44']."


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def dev_server(db: Path) -> Iterator[str]:
    """`vdreamer dev` exactly as documented, in its own process, on its own port."""
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "vdreamer", "dev", "--registry", "vdreamer.yaml",
         "--agent", "t17-memory-desk", "--db", str(db), "--port", str(port)],
        cwd=TUTORIALS, env={**os.environ, "VDREAMER_INTERNAL_TOKEN": TOKEN},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                if httpx.get(f"{url}/api/v1/healthz").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"vdreamer dev did not start:\n{process.stdout.read()}")
            time.sleep(0.2)
        yield url
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()


def test_memory_survives_a_vdreamer_dev_restart(tmp_path: Path, monkeypatch: Any) -> None:
    db = tmp_path / "dev.db"
    with dev_server(db) as url:
        first = vdreamer_run(url, monkeypatch, agent="t17-memory-desk", user="ada", once=True,
                             message="remember for now: blue tent", color=False)
        vdreamer_run(url, monkeypatch, agent="t17-memory-desk", user="ada", once=True,
                     thread=first.thread_id(), message="remember forever: size 44", color=False)
    # The process is gone; only ./dev.db remains.
    with dev_server(db) as url:
        same_thread = vdreamer_run(url, monkeypatch, agent="t17-memory-desk", user="ada",
                                   once=True, thread=first.thread_id(), message="recall",
                                   color=False)
        new_thread = vdreamer_run(url, monkeypatch, agent="t17-memory-desk", user="ada",
                                  once=True, message="recall", color=False)
        other_user = vdreamer_run(url, monkeypatch, agent="t17-memory-desk", user="lin",
                                  once=True, message="recall", color=False)
    assert "This conversation: ['blue tent']. Always: ['size 44']." in same_thread.out
    assert "This conversation: nothing. Always: ['size 44']." in new_thread.out
    assert "This conversation: nothing. Always: nothing." in other_user.out


def test_the_gateway_hides_one_users_thread_from_another(monkeypatch: Any) -> None:
    with served([agent], {agent.name: lambda: {"store": InMemoryStore()}}) as url:
        ada = vdreamer_run(url, monkeypatch, agent=agent.name, user="ada", once=True,
                           message="remember for now: surprise party", color=False)
        lin = {"Authorization": f"Bearer {TOKEN}", "X-User-Id": "lin"}
        thread = f"{url}/api/v1/threads/{ada.thread_id()}"
        read = httpx.get(thread, headers=lin)
        run = httpx.post(f"{thread}/runs", headers=lin,
                         json={"input": {"messages": [{"role": "user", "content": "recall"}]}})
    # 404, not 403: another user's thread is indistinguishable from no thread at all.
    assert read.status_code == run.status_code == 404
    assert read.json()["error"]["code"] == "thread_not_found"
