"""MCP servers as governed tool sources (PRD §2.7, A38–A40; Milestone 5).

The one module that touches ``langchain.mcp`` (beta, version-pinned through
the ``mcp`` extra, A38) and FastMCP. Servers are ``type: mcp`` entries in
``resources.yaml`` (A39):

.. code-block:: yaml

    netinv: {type: mcp, url: https://mcp.netinv.internal/mcp, bearer_env: NETINV_TOKEN,
             startup: required, refresh_s: 300, include: ["get_*", "search_*"],
             identity: service}
    pdfkit: {type: mcp, command: /opt/mcp/pdfkit/bin/server, args: ["--stdio"],
             env: {LOG_LEVEL: info}, startup: optional}

- **Connect** — one ``langchain.mcp.MCPAdapter`` per server at worker boot:
  ``url`` → Streamable HTTP with the ``bearer_env`` token on the client,
  ``command`` → stdio (the subprocess stays alive between calls).
- **Discover** — ``list_tools()`` cached per worker and refreshed every
  ``refresh_s``; tools are renamed ``{server}_{tool}`` and stamped with
  ``source = mcp:{server}``. :meth:`MCPRegistry.snapshot_hash` changes when
  the discovered set does — the runner rebuilds the graph for the next run
  and records the hash on it.
- **Failures** — a ``required`` server that is down fails readiness
  (:meth:`MCPRegistry.probes`); an ``optional`` one is excluded (its tools
  disappear) and logged. Both are retried on the refresh interval.
- **Elicitation** — ``MCPAdapter`` answers a server's input request with a
  LangGraph ``interrupt()``; ``streaming.normalize_interrupt`` maps it onto
  kind ``mcp_elicitation`` (§4.5).
- **Network rules (A40)** — ``NETWORK_ALLOWLIST`` is checked at config load
  *and* again at connect (with DNS resolution); a stdio ``command`` must be an
  absolute path to an installed binary, never a launcher that fetches
  packages (``npx``, ``uvx``, ``pip``, …); on POSIX the server runs under
  rlimits with only its declared ``env`` (``_mcp_launch.py``). The VM network
  policy stays authoritative — the allowlist is defense in depth.
- **Identity** — ``service`` (the server's own bearer) only; ``forward`` is P1.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Final, Literal

from vdreamer import metrics
from vdreamer.governance import SOURCE_METADATA_KEY
from vdreamer.network import (
    NETWORK_ALLOWLIST_ENV,
    EndpointNotAllowed,
    EndpointUnavailable,
    NetworkAllowlist,
)
from vdreamer.resources import ResourcesError, check_entry

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

_log = logging.getLogger(__name__)

#: Launchers that fetch and run packages at call time (A40): never allowed.
REJECTED_LAUNCHERS: Final = frozenset(
    {"npx", "uvx", "uv", "pip", "pip3", "pipx", "npm", "pnpm", "pnpx", "yarn", "bunx", "bun"}
)

DEFAULT_REFRESH_S: Final = 300.0
DEFAULT_CONNECT_TIMEOUT_S: Final = 30.0

_SERVER_NAME_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,62}$")
_KNOWN_KEYS: Final = frozenset(
    {
        "type",
        "url",
        "command",
        "args",
        "env",
        "bearer_env",
        "startup",
        "refresh_s",
        "include",
        "identity",
        "limits",
        "connect_timeout_s",
    }
)
_LIMIT_KEYS: Final = frozenset({"memory_mb", "cpu_s", "fsize_mb", "nofile"})


class MCPConfigError(ResourcesError):
    """A ``type: mcp`` entry breaks a rule (shape, A23 secrets, A40 network)."""


class MCPUnavailable(EndpointUnavailable):
    """A required MCP server is not connected (the readiness probe raises it)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPLimits:
    """rlimits for a stdio server (POSIX). ``None`` leaves a limit alone."""

    memory_mb: int | None = 2048
    cpu_s: int | None = None
    fsize_mb: int | None = 1024
    nofile: int | None = 1024

    def launcher_args(self) -> list[str]:
        args: list[str] = []
        for flag, value in (
            ("--memory-mb", self.memory_mb),
            ("--cpu-s", self.cpu_s),
            ("--fsize-mb", self.fsize_mb),
            ("--nofile", self.nofile),
        ):
            if value is not None:
                args += [flag, str(value)]
        return args


@dataclass(frozen=True)
class MCPServerConfig:
    """One validated ``type: mcp`` entry."""

    name: str
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    bearer_env: str | None = None
    startup: Literal["required", "optional"] = "required"
    refresh_s: float = DEFAULT_REFRESH_S
    include: tuple[str, ...] = ()
    identity: Literal["service"] = "service"
    limits: MCPLimits = field(default_factory=MCPLimits)
    connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S

    @property
    def transport(self) -> Literal["http", "stdio"]:
        return "http" if self.url is not None else "stdio"

    @property
    def required(self) -> bool:
        return self.startup == "required"


def _launcher_name(command: str) -> str:
    name = PurePath(command).name.lower()
    for suffix in (".exe", ".cmd", ".bat", ".ps1", ".sh"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def check_stdio_command(name: str, command: str, args: Sequence[str]) -> None:
    """A40: an absolute path to an installed binary, never a package launcher."""
    if not os.path.isabs(command):
        raise MCPConfigError(
            f"MCP server {name!r}: command {command!r} must be an absolute path to a "
            "preinstalled binary"
        )
    launcher = _launcher_name(command)
    module_launch = (
        launcher.startswith("python")
        and len(args) >= 2
        and args[0] == "-m"
        and _launcher_name(args[1]) in REJECTED_LAUNCHERS
    )
    if launcher in REJECTED_LAUNCHERS or module_launch:
        raise MCPConfigError(
            f"MCP server {name!r}: {command!r} fetches packages at runtime; install the "
            "server ahead of time and point command at its binary (A40)"
        )
    if not Path(command).is_file():
        raise MCPConfigError(f"MCP server {name!r}: command {command!r} does not exist")


def _check_url(name: str, url: str, allowlist: NetworkAllowlist | None) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise MCPConfigError(f"MCP server {name!r}: url {url!r} must be an http(s) URL")
    if parsed.username or parsed.password:
        raise MCPConfigError(
            f"MCP server {name!r}: credentials in the url put a secret in YAML (A23); "
            "use bearer_env"
        )
    if allowlist is not None:
        try:
            allowlist.check_config(parsed.hostname)
        except EndpointNotAllowed as exc:
            raise MCPConfigError(f"MCP server {name!r}: {exc}") from exc


def _str_list(name: str, key: str, value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MCPConfigError(f"MCP server {name!r}: {key} must be a list of strings")
    return tuple(str(item) for item in value)


def _limits(name: str, value: Any) -> MCPLimits:
    if value is None:
        return MCPLimits()
    if not isinstance(value, Mapping) or set(value) - _LIMIT_KEYS:
        raise MCPConfigError(
            f"MCP server {name!r}: limits takes {sorted(_LIMIT_KEYS)} (integers or null)"
        )
    defaults = MCPLimits()
    picked: dict[str, int | None] = {}
    for key in _LIMIT_KEYS:
        raw = value.get(key, getattr(defaults, key))
        picked[key] = None if raw is None else int(raw)
    return MCPLimits(**picked)


def parse_server(
    name: str, entry: Mapping[str, Any], *, allowlist: NetworkAllowlist | None = None
) -> MCPServerConfig:
    """Validate one ``type: mcp`` entry (shape, A23, A40)."""
    entry = check_entry(name, entry)
    if not _SERVER_NAME_RE.fullmatch(name):
        raise MCPConfigError(
            f"MCP server {name!r}: names prefix tool names ({name}_tool) — use letters, "
            "digits, '_' or '-', starting with a letter"
        )
    unknown = sorted(set(entry) - _KNOWN_KEYS)
    if unknown:
        raise MCPConfigError(f"MCP server {name!r}: unknown keys {unknown}")
    url, command = entry.get("url"), entry.get("command")
    if (url is None) == (command is None):
        raise MCPConfigError(
            f"MCP server {name!r}: set exactly one of url (HTTP) or command (stdio)"
        )
    startup = entry.get("startup", "required")
    if startup not in ("required", "optional"):
        raise MCPConfigError(f"MCP server {name!r}: startup must be 'required' or 'optional'")
    identity = entry.get("identity", "service")
    if identity == "forward":
        raise MCPConfigError(
            f"MCP server {name!r}: identity 'forward' (initiator headers) is not supported yet "
            "(P1); use 'service'"
        )
    if identity != "service":
        raise MCPConfigError(f"MCP server {name!r}: identity must be 'service'")
    bearer_env = entry.get("bearer_env")
    if bearer_env is not None and not isinstance(bearer_env, str):
        raise MCPConfigError(f"MCP server {name!r}: bearer_env names an environment variable")
    args = _str_list(name, "args", entry.get("args"))
    env_value = entry.get("env") or {}
    if not isinstance(env_value, Mapping):
        raise MCPConfigError(f"MCP server {name!r}: env must be a mapping")
    env = {str(key): str(value) for key, value in env_value.items()}
    if url is not None:
        if not isinstance(url, str):
            raise MCPConfigError(f"MCP server {name!r}: url must be a string")
        if args or env:
            raise MCPConfigError(f"MCP server {name!r}: args/env apply to stdio servers only")
        _check_url(name, url, allowlist)
    else:
        if not isinstance(command, str):
            raise MCPConfigError(f"MCP server {name!r}: command must be a string")
        if bearer_env is not None:
            raise MCPConfigError(f"MCP server {name!r}: bearer_env applies to HTTP servers only")
        check_stdio_command(name, command, args)
    try:
        refresh_s = float(entry.get("refresh_s", DEFAULT_REFRESH_S))
        connect_timeout_s = float(entry.get("connect_timeout_s", DEFAULT_CONNECT_TIMEOUT_S))
    except (TypeError, ValueError) as exc:
        raise MCPConfigError(f"MCP server {name!r}: refresh_s/connect_timeout_s: {exc}") from exc
    if refresh_s <= 0 or connect_timeout_s <= 0:
        raise MCPConfigError(f"MCP server {name!r}: refresh_s/connect_timeout_s must be > 0")
    return MCPServerConfig(
        name=name,
        url=url,
        command=command,
        args=args,
        env=env,
        bearer_env=bearer_env,
        startup=startup,
        refresh_s=refresh_s,
        include=_str_list(name, "include", entry.get("include")),
        limits=_limits(name, entry.get("limits")),
        connect_timeout_s=connect_timeout_s,
    )


def mcp_servers_from_resources(
    resources: Mapping[str, Mapping[str, Any]],
    *,
    allowlist: NetworkAllowlist | None = None,
) -> list[MCPServerConfig]:
    """Every ``type: mcp`` entry of a loaded ``resources.yaml``, validated."""
    return [
        parse_server(name, entry, allowlist=allowlist)
        for name, entry in resources.items()
        if entry.get("type") == "mcp"
    ]


# ---------------------------------------------------------------------------
# Runtime: one server, then the registry
# ---------------------------------------------------------------------------


def _import_adapter() -> Any:
    """``langchain.mcp`` — the ``mcp`` extra, imported on first use (beta warning muted)."""
    os.environ.setdefault("FASTMCP_CHECK_FOR_UPDATES", "off")  # zero egress (§5.2)
    try:
        from langchain_core._api import LangChainBetaWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LangChainBetaWarning)
            import langchain.mcp
    except ImportError as exc:  # pragma: no cover - the extra is installed in CI
        raise MCPConfigError(
            "MCP servers need the mcp extra: pip install 'vdreamer[mcp]'"
        ) from exc
    return langchain.mcp


def _stdio_launch(config: MCPServerConfig) -> tuple[str, list[str], dict[str, str]]:
    """``(command, args, env)``: POSIX goes through the rlimit launcher."""
    assert config.command is not None
    env = dict(config.env)
    if sys.platform == "win32":
        # No rlimits on Windows (dev hosts only, D19); the MCP SDK merges its
        # default environment, which carries what Windows binaries need.
        return config.command, list(config.args), env
    launcher = str(Path(__file__).with_name("_mcp_launch.py"))
    args = [
        "-I",
        launcher,
        *config.limits.launcher_args(),
        "--keep",
        ",".join(sorted(env)),
        "--",
        config.command,
        *config.args,
    ]
    return sys.executable, args, env


class MCPServer:
    """One configured server: its adapter, discovered tools, and health."""

    def __init__(self, config: MCPServerConfig, allowlist: NetworkAllowlist | None) -> None:
        self.config = config
        self._allowlist = allowlist
        self._client: Any = None
        self._adapter: Any = None
        self._tools: dict[str, BaseTool] = {}
        self._digest = ""
        self.up = False
        self.last_error: str | None = None
        self.checked_at = 0.0

    @property
    def name(self) -> str:
        return self.config.name

    async def _build_adapter(self) -> Any:
        mcp_module = _import_adapter()
        from fastmcp import Client

        config = self.config
        client: Any
        if config.transport == "http":
            from fastmcp.client.transports import StreamableHttpTransport

            assert config.url is not None
            parsed = urllib.parse.urlsplit(config.url)
            if self._allowlist is not None:
                assert parsed.hostname is not None
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                try:
                    await self._allowlist.check_connect(parsed.hostname, port)
                except EndpointUnavailable as exc:
                    raise MCPUnavailable(str(exc)) from exc
            token: str | None = None
            if config.bearer_env is not None:
                token = os.environ.get(config.bearer_env)
                if not token:
                    raise MCPUnavailable(f"${config.bearer_env} is not set")
            client = Client(StreamableHttpTransport(config.url), auth=token)
        else:
            from fastmcp.client.transports import StdioTransport

            command, args, env = _stdio_launch(config)
            client = Client(StdioTransport(command=command, args=args, env=env, keep_alive=True))
        self._client = client
        return mcp_module.MCPAdapter(client)

    def _qualify(self, tool: BaseTool) -> BaseTool:
        metadata = dict(tool.metadata or {})
        metadata[SOURCE_METADATA_KEY] = f"mcp:{self.name}"
        metadata["vdreamer_mcp"] = {"server": self.name, "tool": tool.name}
        return tool.model_copy(update={"name": f"{self.name}_{tool.name}", "metadata": metadata})

    def _included(self, raw: str, qualified: str, patterns: Sequence[str]) -> bool:
        return not patterns or any(
            fnmatch.fnmatchcase(raw, pattern) or fnmatch.fnmatchcase(qualified, pattern)
            for pattern in patterns
        )

    async def refresh(self) -> None:
        """Connect when needed, re-discover the tools; marks the server up or down."""
        self.checked_at = time.time()
        try:
            if self._adapter is None:
                self._adapter = await self._build_adapter()
            discovered = await asyncio.wait_for(
                self._adapter.list_tools(cache_mode="refresh"),
                timeout=self.config.connect_timeout_s,
            )
        except Exception as exc:
            await self._drop_client()
            self._mark_down(f"{type(exc).__name__}: {exc}")
            return
        tools: dict[str, BaseTool] = {}
        for tool in discovered:
            qualified = self._qualify(tool)
            if self._included(tool.name, qualified.name, self.config.include):
                tools[qualified.name] = qualified
        self._tools = tools
        self._digest = _digest(
            [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.args_schema
                    if isinstance(tool.args_schema, Mapping)
                    else repr(tool.args_schema),
                }
                for tool in sorted(tools.values(), key=lambda item: item.name)
            ]
        )
        if not self.up:
            _log.info("MCP server %r is up with %d tool(s)", self.name, len(tools))
        self.up = True
        self.last_error = None
        metrics.MCP_SERVER_UP.labels(server=self.name).set(1)

    def _mark_down(self, error: str) -> None:
        was_up = self.up
        self.up = False
        self.last_error = error
        metrics.MCP_SERVER_UP.labels(server=self.name).set(0)
        if self.config.required:
            _log.error("required MCP server %r is unavailable: %s", self.name, error)
        else:
            self._tools = {}
            self._digest = ""
            if was_up or self.checked_at:
                _log.warning(
                    "optional MCP server %r is unavailable and its tools are excluded: %s",
                    self.name,
                    error,
                )

    async def _drop_client(self) -> None:
        client, self._client, self._adapter = self._client, None, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()

    def tools(self, include: Sequence[str] | None = None) -> list[BaseTool]:
        """The discovered tools (``[]`` while an optional server is excluded)."""
        patterns = tuple(include or ())
        return [
            tool
            for qualified, tool in sorted(self._tools.items())
            if self._included((tool.metadata or {}).get("vdreamer_mcp", {}).get("tool", ""),
                              qualified, patterns)
        ]

    @property
    def digest(self) -> str:
        return self._digest

    async def probe(self) -> None:
        if not self.up:
            raise MCPUnavailable(f"MCP server {self.name!r} is down: {self.last_error}")

    async def close(self) -> None:
        await self._drop_client()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class MCPRegistry:
    """Every configured server for one worker (PRD §2.7).

    :meth:`start` connects each server and discovers its tools (concurrently);
    :meth:`run_refresh` re-discovers on each server's ``refresh_s`` until
    stopped; :meth:`close` stops the stdio subprocesses.
    """

    def __init__(
        self,
        configs: Sequence[MCPServerConfig],
        *,
        allowlist: NetworkAllowlist | None = None,
    ) -> None:
        self._servers = {config.name: MCPServer(config, allowlist) for config in configs}

    @property
    def servers(self) -> Mapping[str, MCPServer]:
        return self._servers

    async def start(self) -> None:
        await asyncio.gather(*(server.refresh() for server in self._servers.values()))

    async def run_refresh(self, stop: asyncio.Event) -> None:
        """Refresh each server on its interval until ``stop`` is set."""

        async def loop(server: MCPServer) -> None:
            while not stop.is_set():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=server.config.refresh_s)
                if not stop.is_set():
                    await server.refresh()

        await asyncio.gather(*(loop(server) for server in self._servers.values()))

    async def close(self) -> None:
        await asyncio.gather(*(server.close() for server in self._servers.values()))

    def tools(self, server: str, *, include: Sequence[str] | None = None) -> list[BaseTool]:
        """One server's governed-to-be tools, filtered by ``include`` globs."""
        try:
            entry = self._servers[server]
        except KeyError:
            raise LookupError(
                f"no MCP server {server!r} in resources.yaml (have: {sorted(self._servers)})"
            ) from None
        return entry.tools(include)

    def snapshot_hash(self) -> str:
        """Changes whenever any server's discovered tool set does (§2.7)."""
        return _digest({name: server.digest for name, server in sorted(self._servers.items())})

    def probes(self) -> dict[str, Callable[[], Awaitable[None]]]:
        """Readiness probes: one per ``required`` server (a down optional one is excluded)."""
        return {
            f"mcp:{name}": server.probe
            for name, server in self._servers.items()
            if server.config.required
        }


@contextlib.asynccontextmanager
async def open_mcp_registry(
    configs: Sequence[MCPServerConfig],
    *,
    allowlist: NetworkAllowlist | None = None,
    refresh: bool = True,
) -> AsyncIterator[MCPRegistry]:
    """Start the servers, keep them refreshed, close them on exit."""
    registry = MCPRegistry(configs, allowlist=allowlist)
    await registry.start()
    stop = asyncio.Event()
    refresher = asyncio.create_task(registry.run_refresh(stop)) if refresh and configs else None
    try:
        yield registry
    finally:
        stop.set()
        if refresher is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await refresher
        await registry.close()


__all__ = [
    "MCPConfigError",
    "MCPLimits",
    "MCPRegistry",
    "MCPServer",
    "MCPServerConfig",
    "MCPUnavailable",
    "NETWORK_ALLOWLIST_ENV",
    "NetworkAllowlist",
    "check_stdio_command",
    "mcp_servers_from_resources",
    "open_mcp_registry",
    "parse_server",
]
