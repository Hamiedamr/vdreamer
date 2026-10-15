"""Exec an MCP stdio server under rlimits with only its declared env (POSIX).

Run as a script by ``vdreamer.mcp`` (``python -I _mcp_launch.py ...``); it
imports nothing from vdreamer. The MCP SDK merges a default environment
(``PATH``, ``HOME``, …) into every stdio child, so the launcher receives the
declared keys by name and drops everything else before ``execve``.

    _mcp_launch.py [--memory-mb N] [--cpu-s N] [--fsize-mb N] [--nofile N]
                   [--keep KEY,KEY] -- /abs/command arg...
"""

from __future__ import annotations

import os
import sys


def _limit(name: str, value: int) -> None:
    if sys.platform == "win32":  # vdreamer.mcp never launches this on Windows
        raise SystemExit("rlimits need POSIX")
    import resource

    which = getattr(resource, name)
    _soft, hard = resource.getrlimit(which)
    # An unprivileged process may lower its hard limit, never raise it.
    capped = value if hard == resource.RLIM_INFINITY else min(value, hard)
    resource.setrlimit(which, (capped, capped))


def main(argv: list[str]) -> int:
    try:
        split = argv.index("--")
    except ValueError:
        print("usage: _mcp_launch.py [options] -- command [args...]", file=sys.stderr)
        return 2
    options, command = argv[:split], argv[split + 1 :]
    if not command:
        print("_mcp_launch.py: no command", file=sys.stderr)
        return 2
    values: dict[str, str] = {}
    for index in range(0, len(options) - 1, 2):
        values[options[index]] = options[index + 1]

    _limit("RLIMIT_CORE", 0)
    if "--memory-mb" in values:
        _limit("RLIMIT_AS", int(values["--memory-mb"]) * 1024 * 1024)
    if "--cpu-s" in values:
        _limit("RLIMIT_CPU", int(values["--cpu-s"]))
    if "--fsize-mb" in values:
        _limit("RLIMIT_FSIZE", int(values["--fsize-mb"]) * 1024 * 1024)
    if "--nofile" in values:
        _limit("RLIMIT_NOFILE", int(values["--nofile"]))

    keep = {key for key in values.get("--keep", "").split(",") if key}
    env = {key: value for key, value in os.environ.items() if key in keep}
    os.execve(command[0], command, env)
    return 127  # pragma: no cover - execve does not return


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
