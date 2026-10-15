"""``vdreamer doctor --offline`` — environment probe and zero-egress assertion.

Milestone 0 scope (PRD §2.3, §5.2): force the offline environment variables,
report interpreter and LangChain/LangGraph versions, import the stack under
a socket-level egress guard, and assert nothing tried to reach the network.
The VM network policy is the authoritative control (A40); this guard is
defense in depth and the thing this command asserts against. Backend,
model-file, MCP-server, and sandbox probes land with their milestones.

Milestone 4 adds the gateway stack (FastAPI, uvicorn, prometheus-client) to
the guarded imports and the vendored-asset check (§5.2, A13): every file
under ``static/vendor/`` must match its pinned sha256 and the docs shell must
reference no external origin (``vdreamer.vendored``).

Milestone 6 adds the artifact/files/sandbox stack to the guarded imports
(pyarrow, xlsxwriter and the MinIO client too, when their extras are
installed), the vendored chart JS to the asset check, and a sandbox probe:
a trivial program through ``vdreamer.sandbox`` under its rlimits, reporting
whether the network namespace and the dedicated uid took. Windows hosts
report the sandbox as unavailable (D19) without failing.

Milestone 7 adds the backend probes (§2.3 "probes all backends, model
files, MCP servers"), run after the egress assertion against the declared
endpoints only: every ``models.yaml`` alias (``--models`` or
``$VDREAMER_MODELS``) — a chat endpoint must answer and serve its model, a
fastembed model's files must be in the cache (located, not loaded) — and
every ``resources.yaml`` entry (``--resources`` or ``$VDREAMER_RESOURCES``):
each resource connects and passes its probe, each MCP server starts (a
``required`` one that does not is a failure). ``langchain_openai``,
``langchain_ollama`` and fastembed join the guarded imports.

Milestone 10 moves the forced variables to :mod:`vdreamer.offline` (every
command applies them now, not only ``doctor``; the list grew the telemetry
switches) and reports the tiktoken cache: a ``TIKTOKEN_CACHE_DIR`` that is
set but missing or empty fails; unset (with tiktoken installed) is a warning,
since only token counting would reach for it.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
from importlib import metadata as importlib_metadata
from typing import Any, cast

from vdreamer import __version__
from vdreamer.offline import (
    OFFLINE_ENV,
    TIKTOKEN_CACHE_ENV,
    apply_offline_env,
    tiktoken_cache_problem,
)

#: Reported in the version block (PRD §2.3: LangChain/LangGraph versions).
VERSION_PACKAGES: tuple[str, ...] = (
    "langchain",
    "langgraph",
    "langgraph-checkpoint",
    "langgraph-checkpoint-postgres",
    "fastapi",
    "uvicorn",
    "fastmcp",
    "pyarrow",
    "minio",
)


class EgressBlockedError(OSError):
    """Raised by the egress guard on a blocked socket call."""


class EgressGuard:
    """Record and block outbound socket calls while installed (§5.2, A40).

    Blocks the client-side paths (``connect``/``connect_ex`` on sockets,
    ``create_connection``, ``getaddrinfo``). Loopback-only usage is not
    exempted: Milestone 0 probes nothing, so any socket call is a violation.
    """

    def __init__(self) -> None:
        self.violations: list[str] = []
        self._originals: dict[str, Any] = {}

    def _blocked(self, label: str) -> Any:
        def blocked(*args: Any, **kwargs: Any) -> Any:
            target = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
            self.violations.append(f"{label}({target})")
            raise EgressBlockedError(f"egress attempt under --offline: {label}({target})")

        return blocked

    def install(self) -> None:
        self._originals = {
            "socket.socket.connect": socket.socket.connect,
            "socket.socket.connect_ex": socket.socket.connect_ex,
            "socket.create_connection": socket.create_connection,
            "socket.getaddrinfo": socket.getaddrinfo,
        }
        # `cast(Any, socket)`: mypy rejects assigning to method attributes,
        # but the guard's stand-ins are opaque (*args, **kwargs) by design.
        sock: Any = cast(Any, socket)
        sock.socket.connect = self._blocked("socket.socket.connect")
        sock.socket.connect_ex = self._blocked("socket.socket.connect_ex")
        sock.create_connection = self._blocked("socket.create_connection")
        sock.getaddrinfo = self._blocked("socket.getaddrinfo")

    def uninstall(self) -> None:
        if not self._originals:
            return
        sock: Any = cast(Any, socket)
        sock.socket.connect = self._originals["socket.socket.connect"]
        sock.socket.connect_ex = self._originals["socket.socket.connect_ex"]
        sock.create_connection = self._originals["socket.create_connection"]
        sock.getaddrinfo = self._originals["socket.getaddrinfo"]
        self._originals = {}


def _report_versions() -> None:
    print(f"python     {sys.version.split()[0]} ({sys.platform})")
    print(f"vdreamer   {__version__}")
    for package in VERSION_PACKAGES:
        try:
            version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            version = "not installed"
        print(f"{package}  {version}")


def _probe_sandbox() -> str | None:
    """Run ``print(42)`` in the sandbox (§2.3 "probes … sandbox"); the problem, or ``None``."""
    import asyncio
    import sys as _sys

    from vdreamer.sandbox import (
        SandboxUnavailable,
        run_confined,
        scratch_dir,
        unavailable_reason,
    )

    reason = unavailable_reason()
    if reason is not None:
        print(f"[--] sandbox: unavailable here: {reason}")
        return None
    try:
        with scratch_dir() as scratch:
            (scratch / "probe.py").write_text("print(42)\n", encoding="utf-8")
            done = asyncio.run(
                run_confined([_sys.executable, "-I", "-B", "probe.py"], cwd=scratch, timeout_s=30)
            )
    except (SandboxUnavailable, OSError) as exc:
        return f"sandbox: {exc}"
    if done.exit_code != 0 or done.stdout.strip() != "42":
        return f"sandbox: probe failed (exit {done.exit_code}): {done.stderr.strip()[-300:]}"
    network = "off" if done.network_isolated else "NOT isolated (no namespace permitted)"
    print(f"[ok] sandbox: probe ran as uid {done.uid}, network {network}")
    return None


async def _probe_backends(models: str | None, resources: str | None) -> list[str]:
    """Probe every declared model alias, resource, and MCP server; the problems."""
    from vdreamer.adapters import probe_resource
    from vdreamer.backends import load_backend_config
    from vdreamer.mcp import mcp_servers_from_resources, open_mcp_registry
    from vdreamer.models import probe_model
    from vdreamer.network import NetworkAllowlist

    allowlist = NetworkAllowlist.from_env()
    try:
        config = load_backend_config(models, resources, allowlist=allowlist)
        mcp_configs = mcp_servers_from_resources(config.resources, allowlist=allowlist)
    except ValueError as exc:  # ModelsError / ResourcesError
        return [f"config: {exc}"]
    problems: list[str] = []

    def report(label: str, ok: bool, detail: str) -> None:
        print(f"[{'ok' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            problems.append(f"{label}: {detail}")

    if config.models is not None:
        for alias in config.models.aliases:
            try:
                report(f"model {alias}", True, await probe_model(
                    config.models, alias, allowlist=allowlist
                ))
            except Exception as exc:
                report(f"model {alias}", False, str(exc))
    for alias, adapter in config.adapters.items():
        try:
            report(f"resource {alias}", True, await probe_resource(adapter))
        except Exception as exc:
            report(f"resource {alias}", False, str(exc))
    if mcp_configs:
        async with open_mcp_registry(mcp_configs, allowlist=allowlist, refresh=False) as registry:
            for server in registry.servers.values():
                label = f"mcp {server.name}"
                if server.up:
                    report(label, True, f"up, {len(server.tools())} tool(s)")
                elif server.config.required:
                    report(label, False, f"required server down: {server.last_error}")
                else:
                    print(f"[--] {label}: optional server down: {server.last_error}")
    if config.models is None and not config.resources:
        print("[--] backends: no models.yaml / resources.yaml given (--models, --resources)")
    return problems


def run_offline(models: str | None = None, resources: str | None = None) -> int:
    """CLI entry point for ``doctor --offline``. 0 = healthy and zero egress."""
    apply_offline_env()

    print("vdreamer doctor --offline")
    _report_versions()

    guard = EgressGuard()
    guard.install()
    problems: list[str] = []
    postgres_note: str | None = None
    try:
        import langchain  # noqa: F401 — imported under the guard on purpose
        import langgraph  # noqa: F401
        import uvicorn  # noqa: F401

        try:
            import langgraph.checkpoint.postgres  # noqa: F401
        except ImportError as exc:
            # psycopg needs libpq: the [postgres] extra ships it (psycopg[binary]),
            # else the host's. Only serve/worker need it; dev runs on SQLite (D11).
            postgres_note = (
                f"postgres: unavailable ({type(exc).__name__}: {str(exc).splitlines()[0]}); "
                "serve/worker need `vdreamer[postgres]` or a system libpq, dev does not"
            )

        import vdreamer.gateway  # noqa: F401 — FastAPI app factory + routes
        import vdreamer.governance  # noqa: F401 — the governed tool pipeline (M5)
        import vdreamer.mcp  # noqa: F401
        if importlib.util.find_spec("fastmcp") is not None:
            from vdreamer.mcp import _import_adapter

            _import_adapter()  # langchain.mcp + FastMCP (the mcp extra), M5
        import vdreamer.adapters  # noqa: F401 — models and resources (M7)
        import vdreamer.files  # noqa: F401 — artifacts, files, sandbox (M6)
        import vdreamer.models  # noqa: F401 — langchain_openai + langchain_ollama
        import vdreamer.sandbox  # noqa: F401
        import vdreamer.tables  # noqa: F401
        for optional in (
            "pyarrow.parquet",
            "pyarrow.csv",
            "xlsxwriter",
            "minio",
            "fastembed",
            "qdrant_client",
            "neo4j",
            "sqlalchemy.ext.asyncio",
        ):
            if importlib.util.find_spec(optional.split(".", 1)[0]) is not None:
                importlib.import_module(optional)
    except Exception as exc:
        problems.append(f"stack import failed: {type(exc).__name__}: {exc}")
    finally:
        guard.uninstall()

    for key, value in OFFLINE_ENV.items():
        effective = os.environ.get(key, "")
        print(f"[{'ok' if effective == value else 'FAIL'}] {key}={effective!r}")
        if effective != value:
            problems.append(f"{key}={effective!r} (expected {value!r})")

    if guard.violations:
        for violation in guard.violations:
            problems.append(f"egress attempt: {violation}")
        print(f"[FAIL] egress: {len(guard.violations)} blocked socket call(s)")
    else:
        print("[ok] egress: zero outbound socket calls during the probe")

    if postgres_note is not None:
        print(f"[--] {postgres_note}")

    tiktoken_problem = tiktoken_cache_problem()
    if tiktoken_problem is None:
        cache = os.environ.get(TIKTOKEN_CACHE_ENV) or "tiktoken not installed"
        print(f"[ok] tiktoken cache: {cache}")
    elif os.environ.get(TIKTOKEN_CACHE_ENV, "").strip():
        problems.append(f"tiktoken cache: {tiktoken_problem}")
    else:
        print(f"[--] tiktoken cache: {tiktoken_problem}")

    from vdreamer.vendored import VENDORED_ASSETS, check_vendored_assets

    asset_problems = check_vendored_assets()
    if asset_problems:
        problems.extend(asset_problems)
        print(f"[FAIL] vendored assets: {len(asset_problems)} problem(s)")
    else:
        print(
            f"[ok] vendored assets: {len(VENDORED_ASSETS)} file(s) match their sha256; "
            "the docs shell references no external origin"
        )

    sandbox_problem = _probe_sandbox()
    if sandbox_problem is not None:
        problems.append(sandbox_problem)

    import asyncio

    problems.extend(asyncio.run(_probe_backends(models, resources)))

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        print("doctor: FAIL")
        return 1
    print("[ok] stack imports without egress")
    print("doctor: PASS")
    return 0
