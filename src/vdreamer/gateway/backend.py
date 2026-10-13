"""What one gateway process is built over: the agent registry plus the stores.

The gateway is stateless (§2.1): everything it knows lives in Postgres (A7)
— or, for ``vdreamer dev`` (D11) and the tests, in the in-memory twins of the
same stores. :class:`GatewayBackend` names those pieces; ``create_app`` turns
one into the REST/SSE surface. The two topologies differ only here:

=============  =========================  ===============================
piece          ``vdreamer serve``          ``vdreamer dev`` / tests
=============  =========================  ===============================
queue          ``PostgresRunQueue``        ``InMemoryRunQueue``
threads        ``PostgresThreadStore``     ``InMemoryThreadStore``
events         ``PostgresEventStore``      ``InMemoryEventLog``
live fan-out   ``EventHub`` fed by          the ``InMemoryEventLog``'s hub
               ``PostgresEventListener``
interrupts     ``PostgresInterruptStore``  ``InMemoryInterruptStore``
artifacts      ``UnavailableArtifactStore``  ``InMemoryArtifactStore``
               (until Milestone 6)
checkpointer   ``AsyncPostgresSaver``      ``AsyncSqliteSaver`` / in-memory
=============  =========================  ===============================

The agents are loaded for their *declarations* only (name, mode,
``input_schema``, ``access_roles``) — the gateway never builds a graph; that is
the worker's job (§2.2: "``build()`` runs once per worker").
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from vdreamer.agents import BaseAgent
from vdreamer.artifacts import ArtifactStore
from vdreamer.eventlog import EventHub, EventReader, EventSink
from vdreamer.interrupts import InterruptStore
from vdreamer.queue import RunQueue
from vdreamer.threads import ThreadStore

#: A readiness probe: returns when healthy, raises otherwise (``/readyz``).
Probe = Callable[[], Awaitable[None]]


@dataclass
class GatewayBackend:
    """The gateway's dependencies (see the module table)."""

    agents: Mapping[str, BaseAgent]
    queue: RunQueue
    threads: ThreadStore
    event_reader: EventReader
    event_writer: EventSink
    hub: EventHub
    interrupts: InterruptStore
    artifacts: ArtifactStore
    checkpointer: Any = None
    probes: Mapping[str, Probe] = field(default_factory=dict)
