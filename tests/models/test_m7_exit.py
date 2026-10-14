"""Milestone 7 exit: an agent declaring models and resources passes
``vdreamer validate`` and runs against stub backends.

Stubs: an OpenAI-compatible (vLLM) and an Ollama server over real HTTP
(``_stub_llm``), SQLite for the ``sql`` resource, qdrant's embedded mode for
``qdrant``, and a recorder for the fastembed class (no model files ship with
the repo; ``test_fastembed_loading`` covers the real class).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import m7_agent
import pytest
from _stub_llm import Reply, StubServer, start_stub
from _tools_kit import drive
from langgraph.checkpoint.memory import InMemorySaver

from vdreamer import models as models_module
from vdreamer.backends import load_backend_config, open_agent_backends
from vdreamer.entrypoints import run_dev
from vdreamer.runtime import AgentRunner
from vdreamer.validate import run_validate

pytest.importorskip("qdrant_client")
pytest.importorskip("sqlalchemy")

HERE = Path(__file__).resolve().parent


@pytest.fixture
def stubs() -> Iterator[tuple[StubServer, StubServer]]:
    vllm = start_stub(
        "vllm", ["qwen3-32b"], [Reply(content="Cairo is the match.", reasoning="cairo hit id 1")]
    )
    ollama = start_stub(
        "ollama", ["llama3"],
        [Reply(content="Cairo.", reasoning="short summary", inline_think=True)],
    )
    yield vllm, ollama
    vllm.stop()
    ollama.stop()


@pytest.fixture
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models_module, "_fastembed_class", lambda kind: m7_agent.FakeTextEmbedding)


@pytest.fixture
def config_files(
    tmp_path: Path, stubs: tuple[StubServer, StubServer], monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    import sqlite3

    vllm, ollama = stubs
    db = tmp_path / "dwh.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sites (id INTEGER, name TEXT)")
        conn.executemany("INSERT INTO sites VALUES (?, ?)", [(1, "Cairo"), (2, "Giza")])
    monkeypatch.setenv("M7_VLLM_KEY", "stub-key")
    monkeypatch.setenv("M7_DWH_URL", f"sqlite+aiosqlite:///{db.as_posix()}")
    models = tmp_path / "models.yaml"
    models.write_text(
        textwrap.dedent(
            f"""\
            aliases:
              reasoner: {{provider: vllm, base_url: "{vllm.base_url}", model: qwen3-32b,
                         api_key_env: M7_VLLM_KEY, max_tokens: 512, thinking: reasoning_content}}
              local:    {{provider: ollama, base_url: "{ollama.base_url}", model: llama3,
                         thinking: think_tags}}
              dense:    {{provider: fastembed, kind: embedding, model: BAAI/bge-small-en-v1.5}}
            fastembed: {{cache_dir: "{(tmp_path / 'fastembed').as_posix()}"}}
            """
        ),
        encoding="utf-8",
    )
    resources = tmp_path / "resources.yaml"
    resources.write_text(
        textwrap.dedent(
            """\
            warehouse: {type: sql, url_env: M7_DWH_URL, read_only: true}
            vectors:   {type: qdrant, location: ":memory:"}
            """
        ),
        encoding="utf-8",
    )
    return models, resources


@pytest.mark.usefixtures("fake_fastembed")
def test_validate_passes(
    config_files: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    models, resources = config_files
    assert run_validate("m7_agent:agent", models=models, resources=resources) == 0
    out = capsys.readouterr().out
    assert "[ok] model:reasoner: vllm" in out and "serves qwen3-32b" in out
    assert "[ok] model:local: ollama" in out
    assert "[ok] model:dense: fastembed embedding BAAI/bge-small-en-v1.5" in out
    assert "[ok] resource:warehouse: sql reachable (read_only)" in out
    assert "[ok] resource:vectors: qdrant reachable" in out


@pytest.mark.smoke
@pytest.mark.usefixtures("fake_fastembed")
async def test_the_agent_runs_against_the_stubs(
    config_files: tuple[Path, Path], stubs: tuple[StubServer, StubServer]
) -> None:
    models, resources = config_files
    config = load_backend_config(models, resources)
    lines: list[str] = []
    async with open_agent_backends(m7_agent.agent, config, log=lines.append) as backends:
        assert sorted(backends.models) == ["dense", "local", "reasoner"]
        assert sorted(backends.resources) == ["vectors", "warehouse"]
        runner = AgentRunner(
            m7_agent.agent,
            checkpointer=InMemorySaver(),
            models=backends.models,
            resources=backends.resources,
        )
        run = await drive(m7_agent.agent, runner=runner, input={"question": "where is cairo?"})
    assert run.error is None, run.error
    assert run.result["sites"] == ["Cairo", "Giza"]
    assert run.result["hits"] == [1]
    assert run.result["answer"] == "Cairo is the match."
    assert run.result["summary"] == "Cairo."
    thinking = "".join(e["text"] for e in run.of("thinking.delta"))
    assert "cairo hit id 1" in thinking and "short summary" in thinking
    vllm, ollama = stubs
    assert "sites=['Cairo', 'Giza'] hits=[1]" in vllm.requests[-1]["messages"][0]["content"]
    assert any("model reasoner: vllm" in line for line in lines)
    assert any("resource warehouse: sql connected, read_only" in line for line in lines)


@pytest.mark.usefixtures("fake_fastembed")
def test_vdreamer_dev_boots_the_backends_and_completes_a_run(
    config_files: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models, resources = config_files
    code = run_dev(
        agent="m7_agent:agent",
        db=str(tmp_path / "dev.db"),
        input_json='{"question": "where is cairo?"}',
        models=str(models),
        resources=str(resources),
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "vdreamer dev: model reasoner: vllm" in out
    assert "vdreamer dev: resource vectors: qdrant connected" in out
    assert "-> completed" in out


@pytest.mark.usefixtures("fake_fastembed")
def test_validate_fails_on_unknown_aliases(
    config_files: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models, resources = config_files
    trimmed = tmp_path / "trimmed.yaml"
    trimmed.write_text(
        "\n".join(line for line in resources.read_text().splitlines() if "vectors" not in line),
        encoding="utf-8",
    )
    assert run_validate("m7_agent:agent", models=models, resources=trimmed) == 1
    out = capsys.readouterr().out
    assert "[FAIL] resource:vectors: unknown alias" in out
    assert run_validate("m7_agent:agent", resources=resources) == 1
    out = capsys.readouterr().out
    assert "[FAIL] model:reasoner: unknown alias (no models.yaml" in out


@pytest.mark.usefixtures("fake_fastembed")
def test_validate_fails_on_unreachable_endpoints(
    config_files: tuple[Path, Path],
    stubs: tuple[StubServer, StubServer],
    capsys: pytest.CaptureFixture[str],
) -> None:
    models, resources = config_files
    stubs[0].stop()
    assert run_validate("m7_agent:agent", models=models, resources=resources) == 1
    out = capsys.readouterr().out
    assert "[FAIL] model:reasoner:" in out and "unreachable" in out
    # --no-probe stays off the network: the dead endpoint is not contacted.
    assert run_validate("m7_agent:agent", models=models, resources=resources, probe=False) == 0
    out = capsys.readouterr().out
    assert "(not contacted)" in out


def test_validate_fails_on_missing_model_files(
    config_files: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("fastembed")
    models, resources = config_files  # the real fastembed class, an empty cache_dir
    assert run_validate("m7_agent:agent", models=models, resources=resources, probe=False) == 1
    out = capsys.readouterr().out
    assert "[FAIL] model:dense: model 'dense': cannot load embedding model" in out


def test_the_cli_validates_and_doctor_probes(config_files: tuple[Path, Path]) -> None:
    """The real CLI, in a subprocess, on the chat models + resources (no fastembed)."""
    models, resources = config_files
    chat_only = models.parent / "chat.yaml"
    chat_only.write_text(
        "\n".join(
            line for line in models.read_text().splitlines()
            if "dense" not in line and "fastembed:" not in line
        ),
        encoding="utf-8",
    )
    agent_file = models.parent / "m7_chat_agent.py"
    agent_file.write_text(
        "from m7_agent import M7Agent\n\n\nclass Chat(M7Agent):\n    name = 'm7-chat'\n"
        "    models = ['reasoner', 'local']\n\n\nagent = Chat()\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(HERE), str(models.parent)])}
    done = subprocess.run(
        [sys.executable, "-m", "vdreamer", "validate", "m7_chat_agent:agent",
         "--models", str(chat_only), "--resources", str(resources)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "validate Chat: PASS" in done.stdout
    doctor = subprocess.run(
        [sys.executable, "-m", "vdreamer", "doctor", "--offline",
         "--models", str(chat_only), "--resources", str(resources)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert "[ok] model reasoner: vllm" in doctor.stdout, doctor.stdout + doctor.stderr
    assert "[ok] resource warehouse: sql reachable (read_only)" in doctor.stdout
    assert "[ok] egress: zero outbound socket calls" in doctor.stdout
    assert doctor.returncode == 0, doctor.stdout + doctor.stderr
