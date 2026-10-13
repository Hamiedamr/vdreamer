"""vdreamer — serve LangGraph agents as a production REST/SSE API.

The contract lives in ``docs/PRD.md`` (protocol version ``1.0``); the plan in
``docs/tasks.md``. Milestone 0 provides the package skeleton, the CLI entry
points CI depends on (``--version``, ``validate``, ``doctor --offline``),
and the first slice of the ``BaseAgent`` developer contract (PRD §2.2).
"""

from importlib import metadata as importlib_metadata

from vdreamer.agents import BaseAgent, BuildContext, Mode

__all__ = ["BaseAgent", "BuildContext", "Mode", "__version__"]

try:
    __version__ = importlib_metadata.version("vdreamer")
except importlib_metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev0"
