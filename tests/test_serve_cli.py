"""Milestone-4 CLI surface: ``serve``, ``dev``'s gateway flags, ``worker --metrics-*``.

Every check here fails *before* any socket opens: missing config is refused
up front (A14, A23), so ``serve`` never starts half-configured.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vdreamer.cli import build_parser, main
from vdreamer.entrypoints import ServeAgentsError, resolve_serve_agents, run_serve

TOKEN = "current-internal-token-0123456789"


def test_serve_help_mentions_the_contract(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["serve", "--help"])
    out = capsys.readouterr().out
    assert "VDREAMER_INTERNAL_TOKEN" in out and "--agents-dir" in out and "--host" in out


def test_serve_parses_repeatable_agents_and_bind_flags() -> None:
    args = build_parser().parse_args(
        ["serve", "--agent", "a:x", "--agent", "b:y", "--host", "10.0.0.2", "--port", "9000"]
    )
    assert (args.agent, args.host, args.port) == (["a:x", "b:y"], "10.0.0.2", 9000)
    dev = build_parser().parse_args(["dev", "--agent", "pinger:agent", "--port", "9100"])
    assert (dev.host, dev.port) == ("127.0.0.1", 9100)
    worker = build_parser().parse_args(["worker", "--agent", "p:a", "--metrics-port", "9300"])
    assert (worker.metrics_host, worker.metrics_port) == ("127.0.0.1", 9300)


def test_serve_requires_the_internal_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("VDREAMER_INTERNAL_TOKEN", raising=False)
    assert run_serve(agents=["pinger:agent"]) == 1
    assert "VDREAMER_INTERNAL_TOKEN" in capsys.readouterr().err


def test_serve_refuses_wildcard_binds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VDREAMER_INTERNAL_TOKEN", TOKEN)
    assert run_serve(agents=["pinger:agent"], host="0.0.0.0") == 1
    assert "internal interface only" in capsys.readouterr().err


def test_serve_requires_agents(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VDREAMER_INTERNAL_TOKEN", TOKEN)
    assert run_serve() == 1
    assert "--agent, --registry, or --agents-dir" in capsys.readouterr().err
    assert run_serve(agents=["no.such.module:agent"]) == 1
    assert "cannot import" in capsys.readouterr().err


def test_serve_requires_a_dsn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VDREAMER_INTERNAL_TOKEN", TOKEN)
    monkeypatch.delenv("VDREAMER_PG_DSN", raising=False)
    assert run_serve(agents=["pinger:agent"]) == 1
    assert "VDREAMER_PG_DSN" in capsys.readouterr().err


def test_worker_refuses_a_wildcard_metrics_bind(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from vdreamer import entrypoints

    monkeypatch.setenv("VDREAMER_PG_DSN", "postgresql://vd:vd@localhost:5432/vd")

    class _Queue:
        def __init__(self, dsn: str) -> None:
            pass

        async def setup(self) -> None:
            return None

    monkeypatch.setattr(entrypoints, "PostgresRunQueue", _Queue)
    monkeypatch.setattr(entrypoints, "PostgresInterruptStore", _Queue)
    code = entrypoints.run_worker(agent="pinger:agent", metrics_host="0.0.0.0", metrics_port=1)
    assert code == 1
    assert "internal interface only" in capsys.readouterr().err


def test_resolve_serve_agents_from_a_registry_and_dir(tmp_path: Path) -> None:
    registry = tmp_path / "vdreamer.yaml"
    registry.write_text("agents:\n  pinger: pinger:agent\n", encoding="utf-8")
    assert list(resolve_serve_agents([], registry=str(registry))) == ["pinger"]
    assert list(resolve_serve_agents(["pinger"], registry=str(registry))) == ["pinger"]

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "solo.py").write_text(
        "from vdreamer import BaseAgent\n"
        "class Solo(BaseAgent):\n    name = 'solo'\n"
        "agent = Solo()\n",
        encoding="utf-8",
    )
    served = resolve_serve_agents([], registry=str(registry), agents_dir=str(agents_dir))
    assert sorted(served) == ["pinger", "solo"]


def test_resolve_serve_agents_rejects_duplicate_names(tmp_path: Path) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "twin.py").write_text(
        "from vdreamer import BaseAgent\n"
        "class Twin(BaseAgent):\n    name = 'pinger'\n"
        "agent = Twin()\n",
        encoding="utf-8",
    )
    with pytest.raises(ServeAgentsError, match="pinger"):
        resolve_serve_agents(["pinger:agent", "pinger"], agents_dir=str(agents_dir))
