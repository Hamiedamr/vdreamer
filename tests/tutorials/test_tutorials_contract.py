"""The tutorial series as a whole: public API only, every agent registered and valid.

- **Public API only.** Every module under ``tutorials/src`` imports vdreamer's
  top-level names only (no ``vdreamer.<submodule>``) and touches only names in
  ``vdreamer.__all__`` — the same bar the reference agents meet
  (``tests/acceptance/test_reference_agents.py``). So no lesson teaches an
  internal or an invented API.
- **Registered and valid.** Each lesson's agents are listed in
  ``tutorials/vdreamer.yaml``, resolve by name, and pass ``vdreamer validate``'s
  checks with their model aliases bound to scripted models.
- **Configs complete.** Every shipped ``config/models*.yaml`` loads and defines
  every alias a registered agent declares.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from _tutorial_kit import LESSONS, TUTORIALS
from vdreamer_tutorials.testing import scripted

import vdreamer
from vdreamer import load_registry, resolve_agent
from vdreamer.backends import load_backend_config
from vdreamer.validate import validate_agent

PUBLIC = set(vdreamer.__all__)
SOURCES = sorted(LESSONS.rglob("*.py"))
REGISTRY = TUTORIALS / "vdreamer.yaml"


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        where = f"{path.relative_to(LESSONS).as_posix()}:{getattr(node, 'lineno', 0)}"
        if isinstance(node, ast.Import):
            found += [f"{where} import {a.name}" for a in node.names
                      if a.name.startswith("vdreamer.")]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("vdreamer."):
                found.append(f"{where} from {node.module} import ...")
            elif node.module == "vdreamer":
                found += [f"{where} from vdreamer import {a.name} (not public)"
                          for a in node.names if a.name not in PUBLIC]
        elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
              and node.value.id == "vdreamer" and node.attr not in PUBLIC):
            found.append(f"{where} vdreamer.{node.attr} (not public)")
    return found


def test_the_tutorials_use_vdreamers_public_api_only() -> None:
    assert SOURCES, LESSONS
    assert [v for path in SOURCES for v in _violations(path)] == []


def test_the_scan_catches_internal_imports(tmp_path: Path) -> None:
    probe = LESSONS / "_probe_internal.py"
    probe.write_text(
        "import vdreamer\nfrom vdreamer.models import build_models\n"
        "from vdreamer import Worker, _private\nvdreamer.governance\n",
        encoding="utf-8",
    )
    try:
        found = _violations(probe)
    finally:
        probe.unlink()
    assert len(found) == 3, found


def _registry() -> dict[str, str]:
    return dict(load_registry(REGISTRY))


def _agents() -> dict[str, Any]:
    return {name: resolve_agent(name, registry=REGISTRY) for name in _registry()}


def test_every_lesson_registers_its_agents() -> None:
    registered = _registry()
    lessons = sorted(p.name for p in LESSONS.iterdir() if p.is_dir() and p.name[:1] == "t")
    assert lessons, LESSONS
    for lesson in lessons:
        assert any(f".{lesson}." in spec for spec in registered.values()), lesson
    for name, spec in registered.items():
        lesson = spec.split(".")[1]  # "t01_serving_an_agent" → names start with "t01-"
        assert name.startswith(lesson[:3] + "-"), (name, spec)


@pytest.mark.parametrize("name", sorted(_registry()))
def test_each_registered_agent_validates(name: str) -> None:
    agent = resolve_agent(name, registry=REGISTRY)
    assert agent.name == name
    models = {alias: scripted() for alias in agent.models}
    failed = [f"{c.name}: {c.detail}" for c in validate_agent(agent, models=models) if not c.ok]
    assert failed == []


@pytest.mark.parametrize("config", sorted((TUTORIALS / "config").glob("models*.yaml")),
                         ids=lambda p: p.name)
def test_the_shipped_model_configs_define_every_alias(config: Path) -> None:
    loaded = load_backend_config(config, None)
    assert loaded.models is not None
    declared = {alias for agent in _agents().values() for alias in agent.models}
    assert declared <= set(loaded.models.aliases)
