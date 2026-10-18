"""Live tier: tutorial agents against a real model (opt-in, never in CI).

The ``chat`` alias is built from ``tutorials/config/models.ollama.yaml`` by
vdreamer itself (``gemma4:31b-cloud`` by default — the ``-cloud`` tag runs on
ollama.com, so this tier causes egress). Opt in with ``VDREAMER_TEST_OLLAMA=1``
and a running Ollama on ``127.0.0.1:11434``; point ``VDREAMER_TUTORIALS_MODELS``
at another models file (e.g. ``config/models.vllm.yaml``) to use another server.

Assertions are about the workflow — which tool, which arguments, a completed run —
never about the model's wording.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from _tutorial_kit import TUTORIALS
from vdreamer_tutorials.t02_minimal_react_agent.agent import agent as order_status
from vdreamer_tutorials.t04_custom_tool.agent import agent as shipping
from vdreamer_tutorials.t06_testing_and_traces.agent import ReturnsAgent
from vdreamer_tutorials.testing import AgentSession

from vdreamer.models import build_models, load_models

pytestmark = pytest.mark.skipif(
    os.environ.get("VDREAMER_TEST_OLLAMA") != "1",
    reason="live model tier: set VDREAMER_TEST_OLLAMA=1 (gemma4:31b-cloud, egress)",
)


def _chat() -> Any:
    path = Path(os.environ.get("VDREAMER_TUTORIALS_MODELS",
                               TUTORIALS / "config" / "models.ollama.yaml"))
    return build_models(load_models(path), ["chat"])["chat"]


async def test_t02_looks_the_order_up() -> None:
    result = await AgentSession(order_status, models={"chat": _chat()}).run(
        "Where is my order A1001?"
    )
    print(result.trace())
    assert result.status == "completed"
    assert [(n, a["order_id"].upper()) for n, a in result.tool_calls] == [
        ("get_order_status", "A1001")
    ]
    assert result.answer


async def test_t04_prices_a_domestic_parcel() -> None:
    result = await AgentSession(shipping, models={"chat": _chat()}).run(
        "How much does it cost to ship a 2 kg parcel within the same country, standard service?"
    )
    print(result.trace())
    assert result.status == "completed"
    (name, args), *_ = result.tool_calls
    assert name == "estimate_shipping"
    assert args["destination"] == "domestic" and float(args["weight_kg"]) == 2.0
    assert "6.9" in result.answer  # 4.50 + 1.20 * 2 = 6.90


async def test_t06_checks_the_return_window() -> None:
    agent = ReturnsAgent(today=lambda: date(2026, 10, 1))
    result = await AgentSession(agent, models={"chat": _chat()}).run(
        "Can I still return order A1003?"
    )
    print(result.trace())
    assert result.status == "completed"
    assert ("check_return", {"order_id": "A1003"}) in result.tool_calls
    assert result.answer
