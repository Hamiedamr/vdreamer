"""POSIX launcher for sandboxed children: limits first, then ``execve``.

``vdreamer.sandbox`` starts ``python -I -B -m vdreamer._sandbox_launch CONFIG``
(``start_new_session=True``, every other fd closed). This process is fresh and
single-threaded — which is why the work happens here and not in a
``preexec_fn`` inside the multi-threaded worker — and in order it:

1. drops the network where the kernel permits: a new network namespace
   (``unshare(CLONE_NEWNET)`` as root, else ``CLONE_NEWUSER|CLONE_NEWNET``
   when unprivileged user namespaces are enabled), leaving only a downed
   loopback;
2. switches to the dedicated uid/gid (root only; supplementary groups
   cleared) when one is configured;
3. applies the rlimits (CPU, address space, file size, nproc, open files, no
   core dumps), changes into the scratch directory, sets ``umask 077``;
4. reports ``{network_isolated, uid}`` on the status fd and closes it;
5. ``execve``\\ s the target with exactly the given env — nothing inherited.

Any failure before ``execve`` exits 70 with the reason on stderr; a sandbox
must never fall back to running unconfined.
"""

from __future__ import annotations

import json
import os
import sys

if sys.platform == "win32":  # pragma: no cover - POSIX-only module
    raise SystemExit("vdreamer._sandbox_launch is POSIX-only")

import ctypes
import resource

_CLONE_NEWNET = 0x40000000
_CLONE_NEWUSER = 0x10000000

_LIMITS = {
    "cpu": resource.RLIMIT_CPU,
    "as": resource.RLIMIT_AS,
    "fsize": resource.RLIMIT_FSIZE,
    "nproc": resource.RLIMIT_NPROC,
    "nofile": resource.RLIMIT_NOFILE,
    "core": resource.RLIMIT_CORE,
}


def _unshare(flags: int) -> bool:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return int(libc.unshare(flags)) == 0
    except (OSError, AttributeError):
        return False


def _fail(message: str) -> None:
    sys.stderr.write(f"vdreamer sandbox: {message}\n")
    sys.stderr.flush()
    os._exit(70)


def main() -> None:
    config = json.loads(sys.argv[1])
    uid = config.get("uid")
    gid = config.get("gid")
    isolated = False
    if config.get("isolate_network", True) and sys.platform.startswith("linux"):
        if os.geteuid() == 0:
            isolated = _unshare(_CLONE_NEWNET)
        elif uid is None:
            isolated = _unshare(_CLONE_NEWUSER | _CLONE_NEWNET)
    if uid is not None:
        if os.geteuid() != 0:
            _fail(f"switching to uid {uid} needs root")
        try:
            os.setgroups([])
            os.setgid(int(gid if gid is not None else uid))
            os.setuid(int(uid))
        except OSError as exc:
            _fail(f"cannot switch to uid {uid}: {exc}")
        if os.getuid() != int(uid) or os.geteuid() != int(uid):
            _fail("uid switch did not take")
    for name, value in config.get("rlimits", {}).items():
        try:
            resource.setrlimit(_LIMITS[name], (int(value), int(value)))
        except (KeyError, ValueError, OSError) as exc:
            _fail(f"cannot set rlimit {name}={value}: {exc}")
    try:
        os.chdir(config["cwd"])
    except OSError as exc:
        _fail(f"cannot enter the scratch dir: {exc}")
    os.umask(0o077)
    status_fd = config.get("status_fd")
    if isinstance(status_fd, int):
        try:
            status = {"network_isolated": isolated, "uid": os.getuid()}
            os.write(status_fd, json.dumps(status).encode())
        finally:
            os.close(status_fd)
    argv = [str(arg) for arg in config["argv"]]
    env = {str(k): str(v) for k, v in config.get("env", {}).items()}
    try:
        os.execve(argv[0], argv, env)
    except OSError as exc:
        _fail(f"cannot exec {argv[0]}: {exc}")


if __name__ == "__main__":
    main()
