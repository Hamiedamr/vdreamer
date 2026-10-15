"""Agent discovery (D2) tests: specs, the ``vdreamer.yaml`` registry, ``--agents-dir``."""

import logging
from pathlib import Path

import pytest

from vdreamer.agents import BaseAgent
from vdreamer.discovery import (
    AgentLoadError,
    RegistryError,
    discover_agents_dir,
    load_agent,
    load_registry,
    resolve_agent,
)

_DISCOVERY_LOGGER = "vdreamer.discovery"

# The pattern of examples/smoke/pinger.py, generated into temp modules:
# minimal BaseAgent subclasses building a trivial StateGraph.
_MODULE_HEADER = """\
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from vdreamer import BaseAgent, BuildContext, Mode


class _State(TypedDict):
    message: str
    reply: str


def _reply(state: _State) -> dict[str, Any]:
    return {"reply": f"pong: {state['message']}"}

"""

_CLASS_TEMPLATE = """\


class @CLS@(BaseAgent):
    name = "@NAME@"
    mode = Mode.BOTH

    def build(self, ctx: BuildContext) -> StateGraph:
        graph = StateGraph(_State)
        graph.add_node("reply", _reply)
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)
        return graph


@VAR@ = @CLS@()
"""


def _module_source(*agents: tuple[str, str]) -> str:
    """Module text defining one agent class + module-level instance per pair."""
    source = _MODULE_HEADER
    for cls_name, agent_name in agents:
        source += (
            _CLASS_TEMPLATE.replace("@CLS@", cls_name)
            .replace("@NAME@", agent_name)
            .replace("@VAR@", cls_name.lower())
        )
    return source


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _registry_file(tmp_path: Path, text: str) -> Path:
    return _write(tmp_path / "vdreamer.yaml", text)


# --- load_agent (existing behavior; the base case specs resolve to) ---


def test_load_agent_success() -> None:
    agent = load_agent("pinger:agent")
    assert isinstance(agent, BaseAgent)
    assert agent.name == "pinger"


def test_load_agent_rejects_bad_spec_format() -> None:
    for bad in ("pkg.module", ":agent", "pinger:"):
        with pytest.raises(AgentLoadError, match="must look like"):
            load_agent(bad)


def test_load_agent_rejects_unknown_module() -> None:
    with pytest.raises(AgentLoadError, match="cannot import module"):
        load_agent("no_such_module_zz:agent")


def test_load_agent_rejects_missing_attribute() -> None:
    with pytest.raises(AgentLoadError, match="has no attribute"):
        load_agent("pinger:no_such_attr")


def test_load_agent_rejects_non_base_agent_attribute() -> None:
    with pytest.raises(AgentLoadError, match="is not a vdreamer BaseAgent instance"):
        load_agent("pinger:PingerInput")


# --- load_registry ---


def test_load_registry_happy_path(tmp_path: Path) -> None:
    registry = _registry_file(
        tmp_path,
        "agents:\n"
        "  site-validator: myapp.agents:site_validator\n"
        "  pinger: pinger:agent\n",
    )
    assert load_registry(registry) == {
        "site-validator": "myapp.agents:site_validator",
        "pinger": "pinger:agent",
    }


def test_load_registry_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="cannot read registry file"):
        load_registry(tmp_path / "missing.yaml")


def test_registry_error_is_an_agent_load_error() -> None:
    # Existing `except AgentLoadError` handlers must catch registry problems.
    assert issubclass(RegistryError, AgentLoadError)


def test_load_registry_rejects_non_mapping(tmp_path: Path) -> None:
    registry = _registry_file(tmp_path, "- one\n- two\n")
    with pytest.raises(RegistryError, match="must be a mapping"):
        load_registry(registry)


def test_load_registry_rejects_missing_agents_key(tmp_path: Path) -> None:
    registry = _registry_file(tmp_path, "{}\n")
    with pytest.raises(RegistryError, match="has no 'agents' key"):
        load_registry(registry)


def test_load_registry_rejects_non_mapping_agents(tmp_path: Path) -> None:
    registry = _registry_file(tmp_path, "agents:\n  - one\n  - two\n")
    with pytest.raises(RegistryError, match="'agents' must be a mapping"):
        load_registry(registry)


def test_load_registry_rejects_non_string_spec(tmp_path: Path) -> None:
    registry = _registry_file(tmp_path, "agents:\n  pinger:\n    module: pinger\n")
    with pytest.raises(RegistryError, match="spec string"):
        load_registry(registry)


def test_load_registry_rejects_empty_agents(tmp_path: Path) -> None:
    registry = _registry_file(tmp_path, "agents: {}\n")
    with pytest.raises(RegistryError, match="registry has no agents"):
        load_registry(registry)


def test_registry_never_holds_secrets(tmp_path: Path) -> None:
    # A23: registries carry references (import specs), never secrets — any
    # other top-level key is a shape violation.
    registry = _registry_file(tmp_path, "password: hunter2\nagents:\n  pinger: pinger:agent\n")
    with pytest.raises(RegistryError, match="never holds secrets"):
        load_registry(registry)


# --- discover_agents_dir ---


def test_discover_agents_dir_finds_named_instances(tmp_path: Path) -> None:
    _write(
        tmp_path / "site.py",
        _module_source(("AlphaAgent", "alpha"), ("BetaAgent", "beta")),
    )
    agents = discover_agents_dir(tmp_path)
    assert set(agents) == {"alpha", "beta"}
    assert isinstance(agents["alpha"], BaseAgent)
    assert agents["alpha"].name == "alpha"
    assert agents["beta"].name == "beta"


def test_discover_agents_dir_skips_underscore_modules(tmp_path: Path) -> None:
    _write(tmp_path / "_private.py", _module_source(("PrivateAgent", "private")))
    agents = discover_agents_dir(tmp_path)
    assert agents == {}


def test_discover_agents_dir_skips_broken_modules(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(tmp_path / "broken.py", 'raise RuntimeError("kaboom")\n')
    _write(tmp_path / "good.py", _module_source(("GoodAgent", "good")))
    with caplog.at_level(logging.WARNING, logger=_DISCOVERY_LOGGER):
        agents = discover_agents_dir(tmp_path)
    # The broken module is warned about and skipped; the healthy one survives.
    assert set(agents) == {"good"}
    assert "broken.py" in caplog.text
    assert "kaboom" in caplog.text


def test_discover_agents_dir_duplicate_name_first_sorted_module_wins(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(tmp_path / "aaa_dup.py", _module_source(("DupFirst", "dup")))
    _write(tmp_path / "zzz_dup.py", _module_source(("DupSecond", "dup")))
    with caplog.at_level(logging.WARNING, logger=_DISCOVERY_LOGGER):
        agents = discover_agents_dir(tmp_path)
    assert list(agents) == ["dup"]
    assert type(agents["dup"]).__name__ == "DupFirst"
    assert "duplicate agent name" in caplog.text


def test_discover_agents_dir_skips_nameless_agents(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(tmp_path / "nameless.py", _module_source(("NamelessAgent", "")))
    with caplog.at_level(logging.WARNING, logger=_DISCOVERY_LOGGER):
        agents = discover_agents_dir(tmp_path)
    assert agents == {}
    assert "nameless" in caplog.text


def test_discover_agents_dir_zero_agent_module_is_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(tmp_path / "noagents.py", "answer = 42\n")
    with caplog.at_level(logging.DEBUG, logger=_DISCOVERY_LOGGER):
        agents = discover_agents_dir(tmp_path)
    assert agents == {}
    # A module with no agents is at most a debug note, never a warning.
    assert "skipping" not in caplog.text


# --- resolve_agent ---


def test_resolve_colon_spec_passthrough() -> None:
    agent = resolve_agent("pinger:agent")
    assert agent.name == "pinger"


def test_resolve_colon_spec_ignores_registry(tmp_path: Path) -> None:
    # A colon spec is explicit; the registry is never consulted for it.
    registry = _registry_file(tmp_path, "agents:\n  other: pinger:agent\n")
    agent = resolve_agent("pinger:agent", registry=registry)
    assert agent.name == "pinger"


def test_resolve_bare_name_via_registry(tmp_path: Path) -> None:
    registry = _registry_file(tmp_path, "agents:\n  pinger: pinger:agent\n")
    agent = resolve_agent("pinger", registry=registry)
    assert isinstance(agent, BaseAgent)
    assert agent.name == "pinger"


def test_resolve_bare_name_via_agents_dir(tmp_path: Path) -> None:
    _write(tmp_path / "alpha.py", _module_source(("AlphaAgent", "alpha")))
    agent = resolve_agent("alpha", agents_dir=tmp_path)
    assert agent.name == "alpha"


def test_resolve_registry_wins_over_agents_dir(tmp_path: Path) -> None:
    registry = _registry_file(tmp_path, "agents:\n  dup: pinger:agent\n")
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    _write(agents_dir / "dupmod.py", _module_source(("DirAgent", "dup")))
    agent = resolve_agent("dup", registry=registry, agents_dir=agents_dir)
    assert type(agent).__name__ == "PingerAgent"


def test_resolve_falls_through_registry_to_agents_dir(tmp_path: Path) -> None:
    # Bare name missing from the registry is still found in the agents dir.
    registry = _registry_file(tmp_path, "agents:\n  other: pinger:agent\n")
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    _write(agents_dir / "alpha.py", _module_source(("AlphaAgent", "alpha")))
    agent = resolve_agent("alpha", registry=registry, agents_dir=agents_dir)
    assert agent.name == "alpha"


def test_resolve_unknown_bare_name() -> None:
    with pytest.raises(AgentLoadError, match="unknown agent name 'nope'"):
        resolve_agent("nope")


def test_resolve_missing_spec_with_registry(tmp_path: Path) -> None:
    registry = _registry_file(tmp_path, "agents:\n  pinger: pinger:agent\n")
    with pytest.raises(
        RegistryError, match="spec required: pass pkg.module:agent, --registry, or --agents-dir"
    ):
        resolve_agent(registry=registry)


def test_resolve_missing_registry_file(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="cannot read registry file"):
        resolve_agent("pinger", registry=tmp_path / "missing.yaml")


def test_resolve_missing_agents_dir(tmp_path: Path) -> None:
    # Missing registry *file* → RegistryError; missing agents *dir* →
    # AgentLoadError (the type this test pins).
    with pytest.raises(AgentLoadError, match="agents dir"):
        resolve_agent("alpha", agents_dir=tmp_path / "missing")


def test_load_agent_imports_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `vdreamer dev --agent my_agent:agent` from a project folder: the console
    # script's sys.path does not include the cwd, so discovery adds it.
    import sys

    (tmp_path / "cwd_agent_mod.py").write_text(
        "from vdreamer import BaseAgent\n"
        "class A(BaseAgent):\n"
        "    name = 'cwd-agent'\n"
        "agent = A()\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", str(tmp_path))])
    assert load_agent("cwd_agent_mod:agent").name == "cwd-agent"
