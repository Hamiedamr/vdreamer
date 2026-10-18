"""The documented commands against model *servers*: real HTTP, real LangChain clients.

The shipped ``config/models.ollama.yaml`` and ``config/models.vllm.yaml`` are
copied with ``base_url`` pointed at the repository's stub servers
(``tests/models/_stub_llm.py``: an Ollama API and an OpenAI-compatible API that
answer from a script). Then ``vdreamer dev --registry vdreamer.yaml --agent …
--models … --input …`` runs exactly as a reader types it, in a subprocess, from
``tutorials/``. What this covers that the scripted-model tests do not: vdreamer
building ``ChatOllama``/``ChatOpenAI`` from the YAML, the startup probe, tool
calls parsed from each wire format, and ``output_version="v1"`` content.

What it does not cover: a real model's judgement. That is the opt-in live tier.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from _tutorial_kit import TESTS, TUTORIALS

sys.path.insert(0, str(TESTS / "models"))
from _stub_llm import Reply, start_stub  # noqa: E402


def _config_for(stub_kind: str, base_url: str, tmp_path: Path) -> tuple[Path, str]:
    """The shipped config with base_url replaced; returns (path, served model name)."""
    shipped = TUTORIALS / "config" / f"models.{stub_kind}.yaml"
    data = yaml.safe_load(shipped.read_text(encoding="utf-8"))
    data["aliases"]["chat"]["base_url"] = base_url
    path = tmp_path / shipped.name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path, str(data["aliases"]["chat"]["model"])


@pytest.mark.parametrize("kind", ["ollama", "vllm"])
def test_the_lesson_2_command_against_a_model_server(kind: str, tmp_path: Path) -> None:
    script = [
        Reply(content="", tool_calls=[{"name": "get_order_status",
                                       "args": {"order_id": "A1001"}}]),
        Reply(content="Your order A1001 has shipped and arrives on 4 October."),
    ]
    stub = start_stub(kind, ["placeholder"], script)
    try:
        models, served_model = _config_for(kind, stub.base_url, tmp_path)
        stub.models[:] = [served_model]  # the stub serves exactly the configured model
        env = {
            **os.environ,
            "VDREAMER_INTERNAL_TOKEN": "tutorial-token-0123456789",
            "LLM_API_KEY": "stub-key",  # models.vllm.yaml names it via api_key_env
        }
        done = subprocess.run(
            [sys.executable, "-m", "vdreamer", "dev", "--registry", "vdreamer.yaml",
             "--agent", "t02-order-status", "--models", str(models),
             "--db", str(tmp_path / "dev.db"),
             "--input", '{"messages": [{"role": "user", "content": "Where is A1001?"}]}'],
            cwd=TUTORIALS, env=env, capture_output=True, text=True, timeout=120,
        )
    finally:
        stub.stop()

    out = done.stdout + done.stderr
    assert done.returncode == 0, out
    assert '-> get_order_status({"order_id": "A1001"})' in out
    assert "<- ok Order A1001: shipped on 2026-10-01, arriving 2026-10-04" in out
    assert "Your order A1001 has shipped and arrives on 4 October." in out
    # The model server received the tool's schema and, on the second call, its result.
    first, second = stub.requests[0], stub.requests[1]
    assert any("get_order_status" in str(tool) for tool in first["tools"])
    assert "Order A1001: shipped on 2026-10-01" in str(second["messages"])
