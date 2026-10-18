"""Lesson 10: retries for transient failures, a fallback model, a guard on empty answers."""

from __future__ import annotations

from typing import Any

from _tutorial_kit import served, vdreamer_run
from vdreamer_tutorials.t10_failures_and_retries.agent import (
    FALLBACK_ANSWER,
    FlakyCarrier,
    ResilientTrackerAgent,
)
from vdreamer_tutorials.testing import AgentSession, RunResult, ScriptedChatModel, call, scripted


async def _run(carrier: FlakyCarrier, chat: ScriptedChatModel,
               backup: ScriptedChatModel | None = None) -> RunResult:
    agent = ResilientTrackerAgent(carrier, retry_delay_s=0)  # no waiting in tests
    models = {"chat": chat, "backup": backup or scripted()}
    return await AgentSession(agent, models=models).run("Where is TRK-1001?")


async def test_a_transient_tool_failure_is_retried_invisibly() -> None:
    carrier = FlakyCarrier(failures_before_success=2)
    result = await _run(carrier, scripted(call("track_parcel", tracking_id="TRK-1001"),
                                          "It is at the Frankfurt hub."))

    assert carrier.calls == 3  # two failures, then success
    assert len(result.tool_calls) == 1  # one call as far as the client and model know
    assert result.tool_results[0]["status"] == "ok"


async def test_a_persistent_tool_failure_reaches_the_model_after_the_retries() -> None:
    carrier = FlakyCarrier(failures_before_success=99)
    result = await _run(carrier, scripted(call("track_parcel", tracking_id="TRK-1001"),
                                          "Sorry, the carrier is not answering."))

    assert carrier.calls == 3  # 1 attempt + max_retries=2
    error = result.tool_results[0]["error"]
    assert error["code"] == "tool_error"
    assert "failed after 3 attempts with ConnectionError" in error["message"]
    assert result.status == "completed"


async def test_a_failing_model_is_retried() -> None:
    chat = scripted(ConnectionError("model server: connection reset"), "Recovered.")
    result = await _run(FlakyCarrier(), chat)
    assert result.answer == "Recovered." and chat.calls_made == 2


async def test_when_retries_run_out_the_backup_model_answers() -> None:
    down = ConnectionError("model server down")
    chat, backup = scripted(down, down, down), scripted("Answer from the backup model.")
    result = await _run(FlakyCarrier(), chat, backup)

    assert chat.calls_made == 3 and backup.calls_made == 1
    assert result.answer == "Answer from the backup model."
    assert result.status == "completed"


async def test_an_empty_answer_is_replaced_and_the_client_sees_the_replacement() -> None:
    result = await _run(FlakyCarrier(), scripted(""))

    assert result.answer == FALLBACK_ANSWER
    assert result.payloads("log")[0]["level"] == "warning"
    messages = result.values["messages"]
    assert [m.text for m in messages if m.type == "ai"] == [FALLBACK_ANSWER]  # empty one removed


async def test_when_everything_fails_the_run_fails() -> None:
    down = ConnectionError("model server down")
    result = await _run(FlakyCarrier(), scripted(down, down, down), scripted(down, down, down))
    assert result.status == "failed"
    assert "model server down" in result.payloads("run.status")[-1]["error"]["message"]


def test_through_the_gateway(monkeypatch: Any) -> None:
    agent = ResilientTrackerAgent(FlakyCarrier(failures_before_success=1), retry_delay_s=0)

    def backends() -> dict[str, Any]:
        chat = scripted(call("track_parcel", tracking_id="TRK-1001"), "At the Frankfurt hub.")
        return {"models": {"chat": chat, "backup": scripted()}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name,
                              message="Where is TRK-1001?", once=True, color=False)
    assert result.code == 0, result.out
    assert result.out.count("-> track_parcel(") == 1
    assert "<- ok TRK-1001: in transit" in result.out
