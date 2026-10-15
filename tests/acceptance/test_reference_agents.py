"""The §7.3 bar: UC1–UC3 build on vdreamer's public API only, and pass ``validate``.

Static: every example module imports ``vdreamer`` top-level names only (no
``vdreamer.<submodule>``), and every ``vdreamer.<name>`` it touches is in
``vdreamer.__all__``. Dynamic: each agent passes ``validate_agent`` with its
aliases bound to stand-ins, and each shipped config file loads and defines
every alias its agents declare.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest
from _uc_kit import EXAMPLES, FakeNeo4j, LexicalDense, LexicalRerank, LexicalSparse, scripted
from pdf_research import PdfIngestAgent, PdfResearchAgent
from site_validation import SiteValidationAgent
from sql_analyst import SqlAnalystAgent

import vdreamer
from vdreamer.backends import load_backend_config
from vdreamer.mcp import MCPRegistry, mcp_servers_from_resources
from vdreamer.validate import validate_agent

PUBLIC = set(vdreamer.__all__)
SOURCES = sorted(
    path for uc in ("uc1_sql_analyst", "uc2_site_validation", "uc3_pdf_research")
    for path in (EXAMPLES / uc).rglob("*.py")
)


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        where = f"{path.relative_to(EXAMPLES).as_posix()}:{getattr(node, 'lineno', 0)}"
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


def test_the_examples_use_the_public_api_only() -> None:
    assert len(SOURCES) >= 9, SOURCES
    violations = [v for path in SOURCES for v in _violations(path)]
    assert violations == []


def test_the_scan_catches_internal_imports(tmp_path: Path) -> None:
    probe = EXAMPLES / "uc1_sql_analyst" / "_probe_internal.py"
    probe.write_text(
        "import vdreamer\nfrom vdreamer.models import build_models\n"
        "from vdreamer import Worker, _private\nvdreamer.governance\n",
        encoding="utf-8",
    )
    try:
        found = _violations(probe)
    finally:
        probe.unlink()
    assert len(found) == 3, found  # the submodule import, `_private`, `vdreamer.governance`


def _mcp() -> MCPRegistry:
    """Declared, never contacted (as ``vdreamer validate`` does): the tool list is empty."""
    server = {"type": "mcp", "command": sys.executable,
              "args": ["-m", "site_validation.inventory_server"]}
    return MCPRegistry(mcp_servers_from_resources({"siteinv": server}))


AGENTS: dict[str, tuple[Any, dict[str, Any], dict[str, Any]]] = {
    "sql-analyst": (SqlAnalystAgent(), {"analyst": scripted("x")},
                    {"semantic": FakeNeo4j(), "warehouse": object()}),
    "site-validation": (SiteValidationAgent(), {"vlm": scripted("x")}, {}),
    "pdf-ingest": (PdfIngestAgent(), {"dense": LexicalDense(), "sparse": LexicalSparse()},
                   {"vectors": object()}),
    "pdf-research": (PdfResearchAgent(),
                     {"reasoner": scripted("x"), "dense": LexicalDense(),
                      "sparse": LexicalSparse(), "rerank": LexicalRerank()},
                     {"vectors": object()}),
}


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_each_reference_agent_validates(name: str) -> None:
    agent, models, resources = AGENTS[name]
    mcp = _mcp() if name == "site-validation" else None
    checks = validate_agent(agent, mcp=mcp, models=models, resources=resources)
    failed = [f"{c.name}: {c.detail}" for c in checks if not c.ok]
    assert failed == []
    assert agent.name == name


CONFIGS = {
    "uc1_sql_analyst": ["sql-analyst"],
    "uc2_site_validation": ["site-validation"],
    "uc3_pdf_research": ["pdf-ingest", "pdf-research"],
}


@pytest.mark.parametrize("uc", sorted(CONFIGS))
def test_the_shipped_configs_define_every_declared_alias(uc: str) -> None:
    config_dir = EXAMPLES / uc / "config"
    model_files = sorted(config_dir.glob("models*.yaml"))
    resource_files = sorted(config_dir.glob("resources*.yaml"))
    assert model_files and resource_files
    for models_path in model_files:
        for resources_path in resource_files:
            config = load_backend_config(models_path, resources_path)
            assert config.models is not None
            for name in CONFIGS[uc]:
                agent = AGENTS[name][0]
                assert set(agent.models) <= set(config.models.aliases), models_path.name
                assert set(agent.resources) <= set(config.resources), resources_path.name
