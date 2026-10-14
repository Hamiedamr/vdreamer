"""An agent's models and resources: config load, alias checks, open/close (§3.6, §3.7).

``models.yaml`` (``--models`` or ``$VDREAMER_MODELS``) and
``resources.yaml`` (``--resources`` or ``$VDREAMER_RESOURCES``) are loaded
once per process (:func:`load_backend_config`). For one agent,
:func:`open_agent_backends` then:

1. checks every alias in ``BaseAgent.models`` / ``BaseAgent.resources`` is
   declared (MCP servers count as resources; they reach the agent as tools,
   ``ctx.tools.mcp(...)``);
2. builds the chat models and loads the fastembed models once (A36);
3. probes the chat endpoints (the configured model must be served);
4. opens and probes the pooled resource clients (A35);

and yields the two :class:`~vdreamer.models.AliasMap` objects that become
``BuildContext.models`` / ``BuildContext.resources``. Everything opened is
closed when the block exits. The worker and ``vdreamer dev`` run this at
boot and refuse to start on a failure. ``vdreamer validate`` checks the
same aliases one by one and hands ``build()`` :class:`Unconnected`
placeholders for the resources.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vdreamer.adapters import (
    ResourceAdapter,
    ResourceUnavailable,
    missing_resources,
    open_resources,
    resource_adapters,
)
from vdreamer.models import (
    AliasMap,
    ChatModelConfig,
    ModelsConfig,
    ModelsError,
    ModelUnavailable,
    build_models,
    load_models,
    missing_aliases,
    probe_chat_model,
    resolve_models_path,
)
from vdreamer.network import NetworkAllowlist
from vdreamer.resources import ResourcesError, load_resources, resolve_resources_path


@dataclass
class BackendConfig:
    """The process's loaded ``models.yaml`` and ``resources.yaml``."""

    models: ModelsConfig | None = None
    resources: dict[str, dict[str, Any]] = field(default_factory=dict)
    adapters: dict[str, ResourceAdapter] = field(default_factory=dict)
    allowlist: NetworkAllowlist | None = None


def load_backend_config(
    models: str | Path | None = None,
    resources: str | Path | None = None,
    *,
    allowlist: NetworkAllowlist | None = None,
) -> BackendConfig:
    """Load both files (explicit path, else their env var, else none); validate."""
    models_path = resolve_models_path(models)
    resources_path = resolve_resources_path(resources)
    loaded = load_resources(resources_path) if resources_path is not None else {}
    return BackendConfig(
        models=load_models(models_path, allowlist=allowlist) if models_path is not None else None,
        resources=loaded,
        adapters=resource_adapters(loaded, allowlist=allowlist),
        allowlist=allowlist,
    )


def check_declared(agent: Any, config: BackendConfig) -> None:
    """Every alias the agent declares exists (§3.6: unknown aliases fail)."""
    models = list(getattr(agent, "models", ()) or ())
    unknown_models = missing_aliases(config.models, models)
    if unknown_models:
        where = config.models.origin if config.models is not None else (
            "no models.yaml was given (--models / $VDREAMER_MODELS)"
        )
        raise ModelsError(f"unknown model alias(es) {unknown_models}: {where}")
    declared = list(getattr(agent, "resources", ()) or ())
    unknown = missing_resources(config.resources, declared)
    if unknown:
        where = "resources.yaml" if config.resources else (
            "no resources.yaml was given (--resources / $VDREAMER_RESOURCES)"
        )
        raise ResourcesError(f"unknown resource alias(es) {unknown}: {where}")


class Unconnected:
    """``ctx.resources[alias]`` under ``vdreamer validate``: never contacted."""

    def __init__(self, alias: str) -> None:
        self._alias = alias

    def __getattr__(self, name: str) -> Any:
        raise ResourceUnavailable(
            f"resource {self._alias!r} is not connected under vdreamer validate"
        )

    def __repr__(self) -> str:
        return f"<unconnected resource {self._alias!r}>"


@dataclass
class AgentBackends:
    """What ``BuildContext.models`` / ``BuildContext.resources`` receive."""

    models: AliasMap
    resources: AliasMap


async def probe_models(
    config: BackendConfig, aliases: Sequence[str]
) -> dict[str, str]:
    """Probe each chat alias's endpoint; status lines by alias."""
    status: dict[str, str] = {}
    assert config.models is not None or not aliases
    for alias in aliases:
        assert config.models is not None
        model = config.models.aliases[alias]
        if isinstance(model, ChatModelConfig):
            status[alias] = await probe_chat_model(model, allowlist=config.allowlist)
    return status


@asynccontextmanager
async def open_agent_backends(
    agent: Any,
    config: BackendConfig,
    *,
    probe: bool = True,
    lazy_embeddings: bool = False,
    log: Callable[[str], None] | None = None,
) -> AsyncIterator[AgentBackends]:
    """The agent's models and resources, ready, for the block's lifetime.

    Raises ``ModelsError``/``ResourcesError`` (unknown alias, bad config) or
    ``ModelUnavailable``/``ResourceUnavailable`` (down, not served, missing
    files, missing ``*_env`` variable).
    """
    check_declared(agent, config)
    name = getattr(agent, "name", "") or type(agent).__name__
    owner = f"agent {name!r}"
    model_aliases = list(getattr(agent, "models", ()) or ())
    built = await asyncio.to_thread(
        build_models, config.models, model_aliases, lazy_embeddings=lazy_embeddings
    )
    if probe:
        for alias, line in (await probe_models(config, model_aliases)).items():
            if log is not None:
                log(f"model {alias}: {line}")
    if log is not None:
        for alias in model_aliases:
            assert config.models is not None
            if not isinstance(config.models.aliases[alias], ChatModelConfig):
                log(f"model {alias}: fastembed {config.models.aliases[alias].model} loaded")
    declared = [
        alias for alias in getattr(agent, "resources", ()) or () if alias in config.adapters
    ]
    def opened(adapter: ResourceAdapter) -> None:
        if log is not None:
            mode = ", read_only" if adapter.read_only else ""
            log(f"resource {adapter.alias}: {adapter.type} connected{mode}")

    async with open_resources(config.adapters, declared, probe=probe, on_open=opened) as clients:
        yield AgentBackends(AliasMap("model", owner, built), AliasMap("resource", owner, clients))


def empty_backends(agent: Any = None) -> AgentBackends:
    name = getattr(agent, "name", "") or "the agent"
    return AgentBackends(AliasMap("model", name, {}), AliasMap("resource", name, {}))


__all__ = [
    "AgentBackends",
    "BackendConfig",
    "ModelUnavailable",
    "ResourceUnavailable",
    "Unconnected",
    "check_declared",
    "empty_backends",
    "load_backend_config",
    "open_agent_backends",
    "probe_models",
]
