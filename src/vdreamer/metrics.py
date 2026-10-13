"""Prometheus metrics (PRD §5.4) — one registry, every family the PRD names.

§5.4 lists: queue depth, run latency, SSE connections, model latency/tokens
per alias, tool latency/errors per source, MCP server health, sandbox
timeouts, outbox lag. All of them are *defined* here so the exposition format
is stable from Milestone 4 on; each is *recorded* where its subsystem lives:

=============================  ===========================================
family                          recorded by
=============================  ===========================================
``vdreamer_queue_depth``        gateway, refreshed from the queue per scrape
``vdreamer_run_duration_*``     worker, when a run settles
``vdreamer_sse_connections``    gateway SSE transport
``vdreamer_model_*``            models (Milestone 7)
``vdreamer_tool_*``             governed tool pipeline (Milestone 5)
``vdreamer_mcp_server_up``      MCP lifecycle (Milestone 5)
``vdreamer_sandbox_timeouts``   sandbox (Milestone 6)
``vdreamer_outbox_lag_seconds`` outbox delivery (Milestone 9)
=============================  ===========================================

A dedicated :data:`REGISTRY` (not ``prometheus_client``'s global default)
keeps the exposition to vdreamer's families and keeps tests isolated from
whatever else a process registers. The gateway serves it at
``/api/v1/metrics``; ``vdreamer worker --metrics-port`` serves a worker's.
Exposition is pull-only — nothing here ever sends data anywhere (§5.2).
"""

from __future__ import annotations

from typing import Final

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.exposition import CONTENT_TYPE_LATEST, generate_latest

#: The vdreamer metrics registry.
REGISTRY: Final[CollectorRegistry] = CollectorRegistry(auto_describe=True)

#: The exposition ``Content-Type`` (Prometheus text format 0.0.4).
CONTENT_TYPE: Final[str] = CONTENT_TYPE_LATEST

QUEUE_DEPTH: Final[Gauge] = Gauge(
    "vdreamer_queue_depth",
    "Runs per lifecycle status (PRD §3.2), refreshed on every scrape.",
    ["status"],
    registry=REGISTRY,
)

RUN_DURATION: Final[Histogram] = Histogram(
    "vdreamer_run_duration_seconds",
    "Wall time of one run attempt on a worker, from claim to settle.",
    ["agent", "status"],
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 900, 1800),
    registry=REGISTRY,
)

SSE_CONNECTIONS: Final[Gauge] = Gauge(
    "vdreamer_sse_connections",
    "Open SSE event streams on this gateway replica.",
    registry=REGISTRY,
)

MODEL_LATENCY: Final[Histogram] = Histogram(
    "vdreamer_model_latency_seconds",
    "Model call latency per models.yaml alias (recorded from Milestone 7).",
    ["alias"],
    registry=REGISTRY,
)

MODEL_TOKENS: Final[Counter] = Counter(
    "vdreamer_model_tokens",
    "Model tokens per alias and direction (input|output) (recorded from Milestone 7).",
    ["alias", "direction"],
    registry=REGISTRY,
)

TOOL_LATENCY: Final[Histogram] = Histogram(
    "vdreamer_tool_latency_seconds",
    "Governed tool call latency per source (custom|langchain|mcp:{server}) (Milestone 5).",
    ["source"],
    registry=REGISTRY,
)

TOOL_ERRORS: Final[Counter] = Counter(
    "vdreamer_tool_errors",
    "Governed tool failures per source and error.code (Milestone 5).",
    ["source", "code"],
    registry=REGISTRY,
)

MCP_SERVER_UP: Final[Gauge] = Gauge(
    "vdreamer_mcp_server_up",
    "1 when the MCP server is connected and its tools discovered, else 0 (Milestone 5).",
    ["server"],
    registry=REGISTRY,
)

SANDBOX_TIMEOUTS: Final[Counter] = Counter(
    "vdreamer_sandbox_timeouts",
    "Sandbox executions killed at timeout_s (Milestone 6).",
    registry=REGISTRY,
)

OUTBOX_LAG: Final[Gauge] = Gauge(
    "vdreamer_outbox_lag_seconds",
    "Age of the oldest undelivered outbox entry (Milestone 9).",
    registry=REGISTRY,
)


def render() -> bytes:
    """The current exposition of :data:`REGISTRY` (text format)."""
    return generate_latest(REGISTRY)
