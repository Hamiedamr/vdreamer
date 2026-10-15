"""Agent discovery (D2): import specs, a ``vdreamer.yaml`` registry, or an agents dir.

``pkg.module:agent`` specs are the base case that ``vdreamer validate`` and CI
use directly (:func:`load_agent`). All three D2 forms resolve through
:func:`resolve_agent`: ``vdreamer validate --registry`` / ``--agents-dir``
landed with Milestone 1, while ``vdreamer serve`` / ``vdreamer worker`` gain
the same flags with Milestone 3.

``vdreamer.yaml`` registry shape (``yaml.safe_load``; no agent is imported at
registry time — resolution loads lazively through :func:`load_agent`):

    agents:
      site-validator: myapp.agents:site_validator
      pinger: pinger:agent

Registries carry references, never secrets: YAML may reference secrets only
via ``*_env`` keys (A23), the registry shape holds import specs only, and any
other top-level key is a shape violation (:class:`RegistryError`).

``--agents-dir`` is a flat directory of ``*.py`` modules exposing module-level
``BaseAgent`` instances (:func:`discover_agents_dir`): files starting with
``_`` are skipped, a module that fails to import is logged and skipped — one
broken module never kills discovery — and subdirectories are not scanned (M1
limitation; package agents are reached by spec or registry entry instead).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

import yaml

from vdreamer.agents import BaseAgent

_log = logging.getLogger(__name__)

# Agents-dir modules are registered in sys.modules under a synthetic package
# name so a file named like a stdlib or site-package module (say ``json.py``)
# can never shadow the real one for other imports.
_AGENTS_MODULE_PREFIX = "_vdreamer_agents_dir"


class AgentLoadError(Exception):
    """An agent spec failed to import, or does not name a vdreamer agent."""


class RegistryError(AgentLoadError):
    """The ``vdreamer.yaml`` registry is missing or has the wrong shape.

    Subclasses :class:`AgentLoadError` so existing ``except AgentLoadError``
    handlers catch registry problems too.
    """


def _ensure_cwd_importable() -> None:
    """Put the working directory on ``sys.path``, as ``python -m`` and uvicorn do.

    The ``vdreamer`` console script starts with its own bin directory as
    ``sys.path[0]``, so ``vdreamer dev --agent my_agent:agent`` run from the
    project folder would not find ``my_agent.py`` without this.
    """
    cwd = os.getcwd()
    if cwd not in sys.path and "" not in sys.path:
        sys.path.insert(0, cwd)


def load_agent(spec: str) -> BaseAgent:
    """Load the agent named by a ``pkg.module:attribute`` spec."""
    module_name, sep, attr = spec.partition(":")
    if not module_name or not sep or not attr:
        raise AgentLoadError(f"agent spec {spec!r} must look like 'pkg.module:agent'")
    _ensure_cwd_importable()
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AgentLoadError(f"cannot import module {module_name!r}: {exc}") from exc
    try:
        obj = getattr(module, attr)
    except AttributeError:
        raise AgentLoadError(f"module {module_name!r} has no attribute {attr!r}") from None
    if not isinstance(obj, BaseAgent):
        raise AgentLoadError(f"{spec!r} is not a vdreamer BaseAgent instance")
    return obj


def load_registry(path: str | Path) -> dict[str, str]:
    """Load a ``vdreamer.yaml`` registry into its name → spec mapping.

    Parses with ``yaml.safe_load`` and validates the documented shape (see
    the module docstring). No agent is imported here — resolution loads
    lazily via :func:`load_agent`.

    Raises:
        RegistryError: the file is missing or unreadable, is not valid YAML,
            is not a mapping, carries top-level keys other than ``agents`` (a
            registry holds references, never secrets — A23), lacks the
            ``agents`` key, or its ``agents`` is empty / not a mapping / has
            entries whose name or spec is not a string.
    """
    registry_path = Path(path)
    try:
        raw = registry_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryError(f"cannot read registry file {registry_path}: {exc}") from exc
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise RegistryError(f"registry {registry_path} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise RegistryError(
            f"registry {registry_path} must be a mapping with an 'agents' key, "
            f"got {type(loaded).__name__}"
        )
    unexpected = [key for key in loaded if key != "agents"]
    if unexpected:
        raise RegistryError(
            f"registry {registry_path} has unexpected top-level keys {unexpected!r}; "
            "a registry maps agent names to import specs and never holds secrets (A23)"
        )
    if "agents" not in loaded:
        raise RegistryError(f"registry {registry_path} has no 'agents' key")
    agents = loaded["agents"]
    if not isinstance(agents, dict):
        raise RegistryError(
            f"registry {registry_path}: 'agents' must be a mapping of name → spec, "
            f"got {type(agents).__name__}"
        )
    specs: dict[str, str] = {}
    for agent_name, agent_spec in agents.items():
        if not isinstance(agent_name, str) or not isinstance(agent_spec, str):
            raise RegistryError(
                f"registry {registry_path}: 'agents' entries must map a name to a "
                f"'pkg.module:agent' spec string, got {agent_name!r} → {agent_spec!r}"
            )
        specs[agent_name] = agent_spec
    if not specs:
        raise RegistryError("registry has no agents")
    return specs


def _import_agents_module(module_path: Path) -> ModuleType | None:
    """Import one agents-dir module, isolating failures to that module.

    The module is registered in ``sys.modules`` under a prefixed synthetic
    name — the standard ``importlib`` programmatically-import pattern, so
    ``sys.modules`` lookups (pickling, dataclasses) work — before
    ``exec_module`` runs, and dropped again if execution fails. Returns
    ``None`` (after logging) when the module cannot be imported, so one
    broken module never kills the whole directory scan.
    """
    spec = importlib.util.spec_from_file_location(
        f"{_AGENTS_MODULE_PREFIX}.{module_path.stem}", module_path
    )
    if spec is None or spec.loader is None:
        _log.warning("agents dir: skipping %s: not importable as a Python module", module_path)
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[spec.name]
        _log.warning("agents dir: skipping %s: %s", module_path, exc)
        return None
    return module


def discover_agents_dir(path: str | Path) -> dict[str, BaseAgent]:
    """Discover module-level ``BaseAgent`` instances in a flat agents directory.

    Every ``*.py`` file directly under ``path`` (sorted by file name) is
    imported and each module-level value that is a ``BaseAgent`` instance is
    collected, keyed by ``instance.name``. Files starting with ``_`` are
    skipped (``__init__.py``, ``__pycache__`` residue, private helpers), a
    module that fails to import is warned about and skipped — one broken
    module never kills discovery — and so are instances whose name is empty
    or already taken (the first module in sorted order wins). A module
    yielding zero agents is fine (logged at debug). Subdirectories are not
    scanned (M1 limitation); package agents are reached by spec or registry
    entry instead.

    Raises:
        AgentLoadError: ``path`` is not an existing directory.
    """
    agents_path = Path(path)
    if not agents_path.is_dir():
        raise AgentLoadError(f"agents dir {agents_path} does not exist or is not a directory")
    agents: dict[str, BaseAgent] = {}
    for module_path in sorted(agents_path.glob("*.py")):
        if module_path.name.startswith("_"):
            continue
        module = _import_agents_module(module_path)
        if module is None:
            continue
        found = 0
        for value in vars(module).values():
            if not isinstance(value, BaseAgent):
                continue
            agent_name = value.name
            if not agent_name:
                _log.warning("agents dir: skipping nameless agent in %s", module_path)
                continue
            if agent_name in agents:
                _log.warning(
                    "agents dir: skipping duplicate agent name %r in %s (first wins)",
                    agent_name,
                    module_path,
                )
                continue
            agents[agent_name] = value
            found += 1
        if not found:
            _log.debug("agents dir: %s exposed no agents", module_path)
    return agents


def resolve_agent(
    spec: str | None = None,
    *,
    registry: str | Path | None = None,
    agents_dir: str | Path | None = None,
) -> BaseAgent:
    """Resolve an agent by spec, registry name, or agents dir (D2).

    Resolution order:

    1. ``spec`` containing ``:`` is a ``pkg.module:agent`` spec —
       :func:`load_agent` handles it; ``registry`` and ``agents_dir`` are
       ignored because the spec is explicit.
    2. A bare name is looked up in the registry first (name → spec →
       :func:`load_agent`), then in the agents dir — the registry wins when
       both define the name.
    3. ``spec`` is ``None`` → :class:`RegistryError`: the caller must say
       which agent to serve — vdreamer never guesses between sources or
       between multiple agents.

    A missing registry *file* raises :class:`RegistryError` (via
    :func:`load_registry`); a missing agents *directory* raises
    :class:`AgentLoadError` (via :func:`discover_agents_dir`). A name found
    nowhere raises :class:`AgentLoadError`.
    """
    if spec is None:
        raise RegistryError("spec required: pass pkg.module:agent, --registry, or --agents-dir")
    if ":" in spec:
        return load_agent(spec)
    if registry is not None:
        specs = load_registry(registry)
        if spec in specs:
            return load_agent(specs[spec])
    if agents_dir is not None:
        discovered = discover_agents_dir(agents_dir)
        if spec in discovered:
            return discovered[spec]
    raise AgentLoadError(f"unknown agent name {spec!r}")
