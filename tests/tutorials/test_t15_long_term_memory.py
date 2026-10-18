"""Lesson 15: long-term memory in ctx.store — across threads, per user, durable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _tutorial_kit import served, vdreamer_run
from langchain_core.tools import ToolException
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite.aio import AsyncSqliteStore
from vdreamer_tutorials.t15_long_term_memory.agent import BASE_PROMPT, agent, notes_namespace
from vdreamer_tutorials.testing import AgentSession, call, scripted


def test_no_user_means_no_memory() -> None:
    assert notes_namespace("ada") == ("customers", "ada", "notes")
    with pytest.raises(ToolException):
        notes_namespace(None)


async def test_a_note_survives_into_a_new_conversation() -> None:
    model = scripted(call("save_customer_note", note="camps in winter, below -5 °C"), "Noted.",
                     "Take the 0 °C bag.")
    session = AgentSession(agent, models={"chat": model}, store=InMemoryStore(), user_id="ada")
    await session.run("I always camp in winter", thread_id="monday")
    await session.run("Which sleeping bag?", thread_id="tuesday")  # a new thread

    assert model.prompts[2][0].text == (
        BASE_PROMPT + "\nWhat you know about this customer:\n- camps in winter, below -5 °C"
    )


async def test_customers_never_see_each_others_notes() -> None:
    model = scripted(call("save_customer_note", note="shoe size 44"), "Noted.", "Hello!")
    session = AgentSession(agent, models={"chat": model}, store=InMemoryStore())
    await session.run("I wear size 44", user_id="ada", thread_id="a")
    await session.run("Hi", user_id="lin", thread_id="b")

    assert model.prompts[2][0].text == BASE_PROMPT


async def test_a_run_without_a_user_cannot_write_memory() -> None:
    model = scripted(call("save_customer_note", note="x"), "Sorry.")
    result = await AgentSession(agent, models={"chat": model}, store=InMemoryStore(),
                                user_id="").run("Remember x")
    assert result.tool_results[0]["error"]["message"] == (
        "Long-term memory needs a signed-in customer."
    )


async def test_forgetting_deletes_only_the_callers_notes() -> None:
    store = InMemoryStore()
    await store.aput(("customers", "lin", "notes"), "n1", {"note": "lin's note"})
    model = scripted(call("save_customer_note", note="a"), "ok",
                     call("forget_customer_notes"), "Forgotten.")
    session = AgentSession(agent, models={"chat": model}, store=store, user_id="ada")
    await session.run("remember a")
    result = await session.run("forget everything about me")

    assert result.tool_results[0]["summary"] == "Deleted 1 notes."
    assert await store.asearch(("customers", "ada", "notes")) == []
    assert len(await store.asearch(("customers", "lin", "notes"))) == 1


async def test_the_dev_store_persists_across_restarts(tmp_path: Path) -> None:
    """`vdreamer dev` keeps the store in its SQLite file: a new process sees old notes."""
    db = str(tmp_path / "dev.db")
    async with AsyncSqliteStore.from_conn_string(db) as store:
        await store.setup()
        model = scripted(call("save_customer_note", note="prefers metric units"), "Noted.")
        await AgentSession(agent, models={"chat": model}, store=store, user_id="ada").run("metric")

    async with AsyncSqliteStore.from_conn_string(db) as reopened:  # "after a restart"
        model = scripted("Sure.")
        await AgentSession(agent, models={"chat": model}, store=reopened, user_id="ada").run("hi")
    assert "- prefers metric units" in model.prompts[0][0].text


def test_through_the_gateway_two_users(monkeypatch: Any) -> None:
    store = InMemoryStore()
    model = scripted(call("save_customer_note", note="camps in winter"), "Noted.",
                     "Winter bag for you.", "Hi Lin.")

    def backends() -> dict[str, Any]:
        return {"models": {"chat": model}, "store": store}

    with served([agent], {agent.name: backends}) as url:
        ada1 = vdreamer_run(url, monkeypatch, agent=agent.name, user="ada", once=True,
                            message="I camp in winter", color=False)
        ada2 = vdreamer_run(url, monkeypatch, agent=agent.name, user="ada", once=True,
                            message="Which bag?", color=False)
        vdreamer_run(url, monkeypatch, agent=agent.name, user="lin", once=True,
                     message="Hi", color=False)
    assert ada1.thread_id() != ada2.thread_id()
    assert [item.value["note"] for item in store.search(("customers", "ada", "notes"))] == [
        "camps in winter"
    ]
    assert store.search(("customers", "lin", "notes")) == []
    assert "- camps in winter" in model.prompts[2][0].text  # ada's second conversation
    assert model.prompts[3][0].text == BASE_PROMPT  # lin's
