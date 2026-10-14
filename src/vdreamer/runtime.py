"""The in-process run driver (PRD §2.2, §2.4, A42).

``AgentRunner`` is the Milestone-1 half of the runtime contract: it builds the
agent's graph once and caches it — "build() runs once per worker; graph cached"
(PRD §2.2) — then drives each run with the resolved ``durability`` (A42), the
lifecycle hooks (``on_run_start`` / ``on_run_end``), and the ``VDreamerContext``
bound so the implicit helpers (vdreamer.context) resolve the run from any node
or tool (PRD §2.2 line 70).

The worker/queue topology — claiming runs, lease heartbeats, crash recovery,
resume-from-checkpoint — lands in Milestone 3 and drives this same runner.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, Literal

from langchain_core.runnables import RunnableConfig

from vdreamer.agents import BaseAgent, BuildContext
from vdreamer.context import VDreamerContext, bind_context
from vdreamer.eventlog import EventSink
from vdreamer.governance import ToolKit
from vdreamer.streaming import StreamSpec, make_stream_driver, stream_driver_from_env
from vdreamer.validate import ensure_compiled

if TYPE_CHECKING:
    from vdreamer.mcp import MCPRegistry

#: The ``configurable`` key carrying the run id into every checkpoint (M4).
#: LangGraph copies ``configurable`` scalars into checkpoint metadata
#: (verified on langgraph 1.2.15), so a worker can tell whether a thread's
#: latest checkpoint was written by *this* run — and resume it from the
#: checkpoint instead of re-driving the input (PRD §3.2 recovery).
RUN_ID_CONFIGURABLE_KEY: Final[str] = "vdreamer_run_id"

#: The ``configurable`` key recording the MCP tool-snapshot hash a run was
#: built against (§2.7: "each run recording its tool-snapshot hash"); it lands
#: in the run's checkpoint metadata the same way the run id does.
TOOL_SNAPSHOT_CONFIGURABLE_KEY: Final[str] = "vdreamer_tool_snapshot"

#: ``configurable`` keys a client's run ``config`` may not set (§2.5): the
#: thread and checkpoint addressing are the server's, as is the run id above.
#: Keys starting with ``__`` are LangGraph-internal and reserved as well.
RESERVED_CONFIGURABLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "thread_id",
        "checkpoint_id",
        "checkpoint_ns",
        "checkpoint_map",
        RUN_ID_CONFIGURABLE_KEY,
        TOOL_SNAPSHOT_CONFIGURABLE_KEY,
    }
)


def is_reserved_configurable_key(key: str) -> bool:
    """Whether a client may not set ``key`` in ``config.configurable``."""
    return key in RESERVED_CONFIGURABLE_KEYS or key.startswith("__")


@dataclass(frozen=True)
class RunInfo:
    """What the lifecycle hooks receive (PRD §2.2).

    ``status`` is the hook phase: ``on_run_start`` gets ``"running"``;
    ``on_run_end`` gets a copy with ``"completed"`` or ``"failed"``.
    """

    run_id: str
    thread_id: str
    agent_name: str
    attempt: int = 0
    status: str = "running"


class AgentRunner:
    """Builds an agent's graph once, then drives runs of it in-process (PRD §2.2)."""

    def __init__(
        self,
        agent: BaseAgent,
        *,
        checkpointer: Any = None,
        store: Any = None,
        settings: Mapping[str, Any] | None = None,
        mcp: MCPRegistry | None = None,
        models: Mapping[str, Any] | None = None,
        resources: Mapping[str, Any] | None = None,
    ) -> None:
        self.agent = agent
        self._checkpointer = checkpointer
        self._compiled: Any = None
        self._mcp = mcp
        self._built_snapshot: str | None = None
        policies = dict(getattr(agent, "tool_policies", {}) or {})
        self._ctx = BuildContext(
            settings=settings if settings is not None else {},
            models=models if models is not None else {},
            resources=resources if resources is not None else {},
            tools=ToolKit(policies, mcp=mcp),
            checkpointer=checkpointer,
            store=store,
            tool_policies=policies,
        )

    def tool_snapshot(self) -> str | None:
        """The MCP tool-snapshot hash, or ``None`` without MCP servers (§2.7)."""
        return self._mcp.snapshot_hash() if self._mcp is not None else None

    def build(self) -> Any:
        """Build and compile the graph on the first call; cached afterwards (§2.2).

        ``StateGraph`` returns compile here with the runner's checkpointer;
        already-runnable graphs (compiled ``StateGraph``, functional-API
        ``@entrypoint``, ``create_agent`` / ``create_deep_agent`` prebuilts)
        pass through ``ensure_compiled`` untouched.

        With MCP servers, the cache is keyed by the tool snapshot: a refresh
        that changed the discovered tools rebuilds the graph for the next run
        ("changes apply to new runs", §2.7); runs in flight keep their graph.
        """
        snapshot = self.tool_snapshot()
        if self._compiled is None or snapshot != self._built_snapshot:
            self._compiled = ensure_compiled(
                self.agent.build(self._ctx), checkpointer=self._checkpointer
            )
            self._built_snapshot = snapshot
        return self._compiled

    @property
    def compiled(self) -> Any:
        """The cached compiled graph; ``None`` until the first ``build()``."""
        return self._compiled

    @property
    def has_checkpointer(self) -> bool:
        """Whether runs persist checkpoints (the worker's resume decisions need one)."""
        return self._checkpointer is not None

    async def aget_state(self, config: RunnableConfig) -> Any:
        """The thread's latest ``StateSnapshot`` (``values``, ``next``, ``interrupts``,
        ``metadata``) — LangGraph's own ``aget_state`` on the cached graph."""
        return await self.build().aget_state(config)

    def _resolve_durability(self, durability: Literal["sync", "async"] | None) -> str:
        """Resolve the durability for a run (A42).

        An explicit argument wins, else the agent's ``durability`` attribute
        (agents set ``durability = "async"`` to opt a whole agent in), else
        ``"sync"`` — the default, because crash recovery (§3.2) needs every
        step checkpointed. Anything else is a ``ValueError``.
        """
        resolved: str = (
            durability if durability is not None else str(getattr(self.agent, "durability", "sync"))
        )
        if resolved not in ("sync", "async"):
            raise ValueError(
                "durability must be 'sync' or 'async' (A42: agents may set "
                f"durability = 'async' to opt in), got {resolved!r}"
            )
        return resolved

    def _run_info(
        self,
        *,
        context: VDreamerContext | None,
        config: RunnableConfig | None,
    ) -> RunInfo:
        """The ``RunInfo`` for one run — the identity the hooks receive.

        ``context`` wins when given (the worker, Milestone 3, claims a queued
        run with its ``run_id``/``thread_id``/``attempt``); otherwise fall back
        to ``config["configurable"]``; otherwise generate ids — one execution
        is one ``run_id`` (PRD §3.1).
        """
        configurable: dict[str, Any] = {}
        if config is not None:
            from_config = config.get("configurable")
            if from_config is not None:
                configurable = dict(from_config)
        if context is not None:
            return RunInfo(
                run_id=context.run_id,
                thread_id=context.thread_id,
                agent_name=self.agent.name,
                attempt=context.attempt,
            )
        return RunInfo(
            run_id=str(configurable.get("run_id") or f"run_{uuid.uuid4().hex}"),
            thread_id=str(configurable.get("thread_id") or f"thread_{uuid.uuid4().hex}"),
            agent_name=self.agent.name,
            attempt=int(configurable.get("attempt") or 0),
        )

    async def _execute(
        self,
        built: Any,
        info: RunInfo,
        input: Any,
        config: RunnableConfig | None,
        context: VDreamerContext | None,
        durability: str,
    ) -> Any:
        """Start hook → graph → end hook, failing the run on any exception."""
        await self.agent.on_run_start(info)
        try:
            # context=/durability= verified against langgraph 1.2.15: a graph
            # without a context_schema ignores a non-None context, and
            # durability is a per-invocation ainvoke kwarg. A42's "sync" only
            # means anything with a checkpointer present — and 1.2.15 raises
            # on durability="sync" without one — so the kwarg is not
            # forwarded for a checkpointer-less runner.
            result = await built.ainvoke(
                input,
                config,
                context=context,
                durability=durability if self._checkpointer is not None else None,
            )
        except Exception:
            await self.agent.on_run_end(replace(info, status="failed"), "failed")
            raise
        await self.agent.on_run_end(replace(info, status="completed"), "completed")
        return result

    async def _execute_streaming(
        self,
        built: Any,
        info: RunInfo,
        input: Any,
        config: RunnableConfig | None,
        context: VDreamerContext | None,
        durability: str,
        sink: EventSink,
    ) -> Any:
        """Start hook → stream driver → sink → end hook (``arun(sink=...)``, WP2).

        Same hook and return-value contract as :meth:`_execute`: the driver
        (``vdreamer.streaming``, selected by ``VDREAMER_STREAM_DRIVER``, A31)
        yields protocol envelopes for the whole run — including the terminal
        ``run.status`` — and ``driver.result`` carries the final graph value
        the non-streaming path would return. When the graph raises, the
        driver emits the ``failed`` envelope first, then the exception
        propagates here so the hooks and the caller see it exactly as
        ``_execute`` would.
        """
        spec = StreamSpec(
            run_id=info.run_id,
            thread_id=info.thread_id,
            agent=info.agent_name,
            attempt=info.attempt,
            expose_thinking=bool(getattr(self.agent, "expose_thinking", True)),
            has_checkpointer=self._checkpointer is not None,
        )
        driver = make_stream_driver(stream_driver_from_env())
        await self.agent.on_run_start(info)
        try:
            async for envelope in driver.astream(
                built,
                input,
                config=config,
                context=context,
                durability=durability,
                spec=spec,
            ):
                await sink.write(envelope)
        except Exception:
            await self.agent.on_run_end(replace(info, status="failed"), "failed")
            raise
        await self.agent.on_run_end(replace(info, status="completed"), "completed")
        return driver.result

    async def arun(
        self,
        input: Any,
        *,
        config: RunnableConfig | None = None,
        context: VDreamerContext | None = None,
        durability: Literal["sync", "async"] | None = None,
        sink: EventSink | None = None,
    ) -> Any:
        """Run one invocation of the agent's graph end to end.

        Order of operations (PRD §2.2, A42):

        1. ``build()`` — cached, so the agent's ``build(ctx)`` runs at most
           once per runner.
        2. Resolve durability: the explicit argument, else the agent's
           ``durability`` attribute, else ``"sync"``; anything other than
           ``"sync"``/``"async"`` raises ``ValueError``.
        3. ``on_run_start(info)`` → ``ainvoke`` → ``on_run_end`` with
           ``"completed"``, or ``"failed"`` and the exception re-raised.
           When ``context`` is given, the whole invocation — hooks included —
           runs inside ``bind_context(context)``, so the implicit helpers
           resolve the run from any node or tool; without one, nothing binds
           and the helpers raise if a node reaches for them.

        With ``sink`` (Milestone 2, additive), step 3 instead streams the run
        through the selected driver (``VDREAMER_STREAM_DRIVER``, A31) and
        writes every protocol envelope to the sink in arrival order; hooks,
        exception semantics, and the return value (the final state) are
        identical to the non-streaming path. The driver is chosen per run —
        ``stream_driver_from_env()`` — and an invalid value raises
        ``ValueError`` at run start, never at import.

        ``context`` is forwarded to ``ainvoke`` whenever given — verified safe
        on langgraph 1.2.15: a graph that declares no ``context_schema`` runs
        unchanged, and one that does (``StateGraph(..., context_schema=...)``,
        or ``create_agent`` wired through ``ctx.agent_kwargs()``, Milestone 1
        follow-up) receives it as ``runtime.context`` in nodes. The resolved
        durability is forwarded only when the runner has a checkpointer: there
        is nothing to persist otherwise, and langgraph 1.2.15 raises on
        ``durability="sync"`` without a checkpointer (it also warns that the
        kwarg has no effect there).

        An interrupted run (``require_approval`` → LangGraph ``GraphInterrupt``)
        returns normally with the pending interrupt readable from
        ``aget_state(config).interrupts``; mapping that onto
        ``awaiting_approval`` (PRD §3.2) and resuming is the worker's job,
        Milestone 3. The streaming path emits one ``interrupt.requested``
        envelope per pending interrupt and ends on ``run.status``
        ``"awaiting_approval"`` (§3.2).

        Args:
            input: the graph input — or a ``Command(resume=...)`` continuation.
            config: the run config; ``config["configurable"]["thread_id"]`` is
                what keys the checkpointer.
            context: the run identity, bound for the implicit helpers.
            durability: per-run override of the agent's durability (A42).
            sink: when given, every protocol event of the run is written here
                (Milestone 2, WP2), in arrival order with per-run ``seq``.

        Returns:
            The graph result (``stream_mode="values"``), unchanged.
        """
        built = self.build()
        if self._built_snapshot is not None:
            configurable = dict((config or {}).get("configurable") or {})
            configurable[TOOL_SNAPSHOT_CONFIGURABLE_KEY] = self._built_snapshot
            config = {**(config or {}), "configurable": configurable}
        resolved = self._resolve_durability(durability)
        info = self._run_info(context=context, config=config)
        if sink is None:
            if context is None:
                return await self._execute(built, info, input, config, None, resolved)
            with bind_context(context):
                return await self._execute(built, info, input, config, context, resolved)
        if context is None:
            return await self._execute_streaming(
                built, info, input, config, None, resolved, sink
            )
        with bind_context(context):
            return await self._execute_streaming(
                built, info, input, config, context, resolved, sink
            )
