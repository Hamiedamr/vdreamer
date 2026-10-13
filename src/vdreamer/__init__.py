"""vdreamer — serve LangGraph agents as a production REST/SSE API.

The contract lives in ``docs/PRD.md`` (protocol version ``1.0``); the plan in
``docs/tasks.md``. Milestone 0 shipped the package skeleton and the CI entry
points (``--version``, ``validate``, ``doctor --offline``); Milestone 1
(Developer Contract Core) completes the agent-author surface: the full
``BaseAgent`` / ``BuildContext`` contract, the run context and implicit
helpers, tool policies with ``@vdreamer.tool``, discovery (specs,
``vdreamer.yaml`` registry, agents dir), the Postgres checkpointer factory,
and the ``AgentRunner`` run driver.

Agents import everything they need from here (``from vdreamer import
BaseAgent, BuildContext, Mode, FileRef``, ``@vdreamer.tool``, …); the
submodules stay importable for finer-grained access.
"""

from importlib import metadata as importlib_metadata

from vdreamer.agents import BaseAgent, BuildContext, Mode
from vdreamer.context import (
    MAX_SNIPPET_CHARS,
    ArtifactKind,
    ArtifactRef,
    FileRef,
    VDreamerContext,
    bind_context,
    current_context,
    emit_artifact,
    emit_citation,
    emit_log,
    emit_progress,
    require_approval,
)
from vdreamer.discovery import (
    AgentLoadError,
    RegistryError,
    discover_agents_dir,
    load_agent,
    load_registry,
    resolve_agent,
)
from vdreamer.persistence import (
    DEFAULT_DURABILITY,
    Durability,
    MissingPostgresDependencyError,
    PersistenceConfigError,
    dsn_from_env,
    durability_for,
    open_checkpointer,
)
from vdreamer.runtime import AgentRunner, RunInfo
from vdreamer.tools import (
    VDREAMER_METADATA_KEY,
    ApprovalSpec,
    InterruptPolicy,
    Policy,
    VDreamerToolMiddleware,
    redact_args,
    resolve_policy,
    tool,
)

__all__ = [
    "AgentLoadError",
    "AgentRunner",
    "ApprovalSpec",
    "ArtifactKind",
    "ArtifactRef",
    "BaseAgent",
    "BuildContext",
    "DEFAULT_DURABILITY",
    "Durability",
    "FileRef",
    "InterruptPolicy",
    "MAX_SNIPPET_CHARS",
    "MissingPostgresDependencyError",
    "Mode",
    "PersistenceConfigError",
    "Policy",
    "RegistryError",
    "RunInfo",
    "VDREAMER_METADATA_KEY",
    "VDreamerContext",
    "VDreamerToolMiddleware",
    "__version__",
    "bind_context",
    "current_context",
    "discover_agents_dir",
    "dsn_from_env",
    "durability_for",
    "emit_artifact",
    "emit_citation",
    "emit_log",
    "emit_progress",
    "load_agent",
    "load_registry",
    "open_checkpointer",
    "redact_args",
    "require_approval",
    "resolve_agent",
    "resolve_policy",
    "tool",
]

try:
    __version__ = importlib_metadata.version("vdreamer")
except importlib_metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev0"
