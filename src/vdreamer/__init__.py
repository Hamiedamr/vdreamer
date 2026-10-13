"""vdreamer — serve LangGraph agents as a production REST/SSE API.

The contract lives in ``docs/PRD.md`` (protocol version ``1.0``); the plan in
``docs/tasks.md``. Milestone 0 shipped the package skeleton and the CI entry
points (``--version``, ``validate``, ``doctor --offline``); Milestone 1
(Developer Contract Core) completes the agent-author surface: the full
``BaseAgent`` / ``BuildContext`` contract, the run context and implicit
helpers, tool policies with ``@vdreamer.tool``, discovery (specs,
``vdreamer.yaml`` registry, agents dir), the Postgres checkpointer factory,
and the ``AgentRunner`` run driver. Milestone 2 adds the protocol core
(WP1): protocol v1.0 event types, the ``EventEnvelope`` and its payload
models, and the event sinks (in-memory + the Postgres ``run_events`` log).

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
from vdreamer.eventlog import (
    RUN_EVENTS_DDL,
    EventSink,
    InMemoryEventSink,
    PostgresEventLog,
    SeqCounter,
)
from vdreamer.events import (
    DELTA_BATCH_INTERVAL_S,
    DELTA_BATCH_MAX_TOKENS,
    HEARTBEAT_INTERVAL_S,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_HEADER,
    SSE_BUFFERING_HEADER,
    SSE_BUFFERING_VALUE,
    ArtifactCreatedPayload,
    CitationPayload,
    CitationSource,
    ErrorInfo,
    EventEnvelope,
    EventType,
    InterruptRequestedPayload,
    InterruptResolvedPayload,
    LogPayload,
    MessageCompletedPayload,
    ProgressPayload,
    RunStatusPayload,
    TokenDeltaPayload,
    ToolCallPayload,
    ToolResultPayload,
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
from vdreamer.streaming import (
    STREAM_DRIVER_ENV_VAR,
    VALID_STREAM_DRIVERS,
    DeltaBatcher,
    StreamDriver,
    StreamSpec,
    ThinkTagSplitter,
    V2StreamDriver,
    V3StreamDriver,
    make_stream_driver,
    stream_driver_from_env,
)
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
    "ArtifactCreatedPayload",
    "ArtifactKind",
    "ArtifactRef",
    "BaseAgent",
    "BuildContext",
    "CitationPayload",
    "CitationSource",
    "DEFAULT_DURABILITY",
    "DELTA_BATCH_INTERVAL_S",
    "DELTA_BATCH_MAX_TOKENS",
    "DeltaBatcher",
    "Durability",
    "ErrorInfo",
    "EventEnvelope",
    "EventSink",
    "EventType",
    "FileRef",
    "HEARTBEAT_INTERVAL_S",
    "InMemoryEventSink",
    "InterruptPolicy",
    "InterruptRequestedPayload",
    "InterruptResolvedPayload",
    "LogPayload",
    "MAX_SNIPPET_CHARS",
    "MessageCompletedPayload",
    "MissingPostgresDependencyError",
    "Mode",
    "PersistenceConfigError",
    "Policy",
    "PROTOCOL_VERSION",
    "PROTOCOL_VERSION_HEADER",
    "PostgresEventLog",
    "ProgressPayload",
    "RegistryError",
    "RUN_EVENTS_DDL",
    "RunInfo",
    "RunStatusPayload",
    "SSE_BUFFERING_HEADER",
    "SSE_BUFFERING_VALUE",
    "STREAM_DRIVER_ENV_VAR",
    "SeqCounter",
    "StreamDriver",
    "StreamSpec",
    "ThinkTagSplitter",
    "TokenDeltaPayload",
    "ToolCallPayload",
    "ToolResultPayload",
    "V2StreamDriver",
    "V3StreamDriver",
    "VALID_STREAM_DRIVERS",
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
    "make_stream_driver",
    "open_checkpointer",
    "redact_args",
    "require_approval",
    "resolve_agent",
    "resolve_policy",
    "stream_driver_from_env",
    "tool",
]

try:
    __version__ = importlib_metadata.version("vdreamer")
except importlib_metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev0"
