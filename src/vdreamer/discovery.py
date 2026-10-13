"""Agent discovery (D2).

``pkg.module:agent`` specs are the base case that ``vdreamer validate`` and
CI need. ``--agents-dir`` and the ``vdreamer.yaml`` registry land with
Milestone 1.
"""

from __future__ import annotations

import importlib

from vdreamer.agents import BaseAgent


class AgentLoadError(Exception):
    """An agent spec failed to import, or does not name a vdreamer agent."""


def load_agent(spec: str) -> BaseAgent:
    """Load the agent named by a ``pkg.module:attribute`` spec."""
    module_name, sep, attr = spec.partition(":")
    if not module_name or not sep or not attr:
        raise AgentLoadError(f"agent spec {spec!r} must look like 'pkg.module:agent'")
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
