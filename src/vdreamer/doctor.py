"""``vdreamer doctor --offline`` — environment probe and zero-egress assertion.

Milestone 0 scope (PRD §2.3, §5.2): force the offline environment variables,
report interpreter and LangChain/LangGraph versions, import the stack under
a socket-level egress guard, and assert nothing tried to reach the network.
The VM network policy is the authoritative control (A40); this guard is
defense in depth and the thing this command asserts against. Backend,
model-file, MCP-server, and sandbox probes land with their milestones.
"""

from __future__ import annotations

import os
import socket
import sys
from importlib import metadata as importlib_metadata
from typing import Any, cast

from vdreamer import __version__

#: Forced before importing the stack (PRD §5.2): LangSmith tracing off,
#: HuggingFace/Transformers offline.
OFFLINE_ENV: dict[str, str] = {
    "LANGSMITH_TRACING": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

#: Reported in the version block (PRD §2.3: LangChain/LangGraph versions).
VERSION_PACKAGES: tuple[str, ...] = (
    "langchain",
    "langgraph",
    "langgraph-checkpoint",
    "langgraph-checkpoint-postgres",
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


def run_offline() -> int:
    """CLI entry point for ``doctor --offline``. 0 = healthy and zero egress."""
    for key, value in OFFLINE_ENV.items():
        os.environ[key] = value

    print("vdreamer doctor --offline")
    _report_versions()

    guard = EgressGuard()
    guard.install()
    problems: list[str] = []
    try:
        import langchain  # noqa: F401 — imported under the guard on purpose
        import langgraph  # noqa: F401
        import langgraph.checkpoint.postgres  # noqa: F401
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

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        print("doctor: FAIL")
        return 1
    print("[ok] stack imports without egress")
    print("doctor: PASS")
    return 0
