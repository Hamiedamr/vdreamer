"""Lesson 12: settings from the environment, run-aware logs, hooks, events and metrics."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from _tutorial_kit import TUTORIALS, served, vdreamer_run
from vdreamer_tutorials.t12_config_and_observability.agent import ObservableAgent
from vdreamer_tutorials.t12_config_and_observability.logs import LOGGER_NAME, RunContextFilter
from vdreamer_tutorials.t12_config_and_observability.settings import (
    ObservabilitySettings,
    SettingsError,
)
from vdreamer_tutorials.testing import AgentSession, call, scripted

# -- configuration ---------------------------------------------------------------------------------


def test_defaults_and_overrides_from_the_environment() -> None:
    assert ObservabilitySettings.from_env({}) == ObservabilitySettings()
    loaded = ObservabilitySettings.from_env({"T12_SHOP_NAME": "Example Gear Co",
                                             "T12_SLOW_TOOL_MS": "250", "UNRELATED": "x"})
    assert loaded.shop_name == "Example Gear Co" and loaded.slow_tool_ms == 250


def test_every_problem_is_reported_at_once() -> None:
    with pytest.raises(SettingsError) as raised:
        ObservabilitySettings.from_env({"T12_LOG_LEVEL": "LOUD", "T12_SLOW_TOOL_MS": "-1"})
    message = str(raised.value)
    assert "T12_LOG_LEVEL" in message and "T12_SLOW_TOOL_MS" in message


def test_secrets_are_referenced_by_name() -> None:
    settings = ObservabilitySettings.from_env({"T12_ORDERS_API_KEY_ENV": "ORDERS_KEY"})
    assert settings.orders_api_key({"ORDERS_KEY": "s3cr3t"}) == "s3cr3t"
    assert "s3cr3t" not in settings.model_dump_json()


def test_a_bad_environment_stops_vdreamer_at_startup(tmp_path: Path) -> None:
    env = {**os.environ, "T12_LOG_LEVEL": "LOUD",
           "VDREAMER_INTERNAL_TOKEN": "tutorial-token-0123456789"}
    done = subprocess.run(
        [sys.executable, "-m", "vdreamer", "dev", "--registry", "vdreamer.yaml",
         "--agent", "t12-observable", "--db", str(tmp_path / "dev.db"), "--input", "{}"],
        cwd=TUTORIALS, env=env, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 1
    assert "dev: cannot load 't12-observable': SettingsError: invalid configuration: " \
        "T12_LOG_LEVEL" in done.stderr


# -- logs, hooks and events ------------------------------------------------------------------


class Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []
        self.addFilter(RunContextFilter())

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def logs() -> Iterator[Collect]:
    collect = Collect()
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(collect)
    try:
        yield collect
    finally:
        logger.removeHandler(collect)


async def test_every_log_line_carries_the_run(logs: Collect) -> None:
    agent = ObservableAgent(ObservabilitySettings(log_level="DEBUG"))
    model = scripted(call("get_order_status", order_id="A1001"), "Shipped.")
    result = await AgentSession(agent, models={"chat": model}, user_id="ada").run("A1001?")

    messages = [r.getMessage() for r in logs.records]
    assert messages[0].startswith("run started agent=t12-observable")
    assert "order lookup order_id=A1001 status=shipped" in messages
    assert messages[-1].startswith("run ended status=completed duration_ms=")
    assert sum(m.startswith("model call messages=") for m in messages) == 2
    run_ids = {r.run_id for r in logs.records}  # type: ignore[attr-defined]
    assert len(run_ids) == 1 and run_ids != {"-"}
    assert {r.user_id for r in logs.records} == {"ada"}  # type: ignore[attr-defined]
    # The client sees progress, not the server's log lines.
    assert result.payloads("progress") == [{"label": "looking up the order", "pct": 50.0}]
    assert result.payloads("log") == []


async def test_log_level_comes_from_settings(logs: Collect) -> None:
    agent = ObservableAgent(ObservabilitySettings(log_level="WARNING"))
    model = scripted(call("get_order_status", order_id="A1001"), "Shipped.")
    await AgentSession(agent, models={"chat": model}).run("A1001?")
    assert logs.records == []


# -- metrics ---------------------------------------------------------------------------------------


def test_vdreamer_exports_prometheus_metrics(monkeypatch: Any) -> None:
    agent = ObservableAgent(ObservabilitySettings())

    def backends() -> dict[str, Any]:
        return {"models": {"chat": scripted(call("get_order_status", order_id="A1002"),
                                            "Processing.")}}

    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name, message="A1002?",
                              once=True, color=False)
        metrics = httpx.get(f"{url}/api/v1/metrics").text
    assert result.code == 0, result.out
    assert "[progress] looking up the order 50%" in result.out
    assert 'vdreamer_tool_latency_seconds_count{source="custom"}' in metrics
    assert "vdreamer_queue_depth" in metrics
