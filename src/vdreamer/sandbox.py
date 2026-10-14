"""The code sandbox: ``ctx.sandbox.run(code, inputs, timeout_s=60)`` (PRD §3.5, D19, A19).

Generated code runs as a worker subprocess under ``vdreamer._sandbox_launch``:

- **dedicated uid** — ``VDREAMER_SANDBOX_UID`` (a uid or user name; the
  worker must run as root to switch). Without it the child keeps the
  worker's uid, the one non-production compromise; the result says so;
- **rlimits** — CPU (``timeout_s`` + 1 s), 2 GB address space, file size,
  nproc (with a dedicated uid only: ``RLIMIT_NPROC`` counts the whole uid),
  open files, no core dumps;
- **wiped scratch dir** — a fresh directory under ``VDREAMER_SANDBOX_ROOT``
  (default: the system temp dir) holding the code, its inputs, and ``out/``;
  deleted afterwards whatever happened;
- **empty env** — nothing inherited; only sandbox-local ``HOME``/``TMPDIR``,
  ``MPLBACKEND=Agg``, and single-threaded BLAS settings;
- **network off where namespaces permit** — a new network namespace
  (containers are P1).

The interpreter is the worker environment's Python (``sys.executable -I``),
so pandas, pyarrow, matplotlib, plotly and numpy come from the ``[sandbox]``
extra. Regular files the code leaves in ``./out/`` become artifacts, by
extension: PNG/JPEG/SVG → ``image``, HTML → ``html``, JSON objects →
``json``, Parquet/CSV → ``table``, anything else → ``file``. Symlinks,
hard links and special files there are ignored — they could point at the
worker's own files.

POSIX only (D19): on Windows dev hosts ``run`` raises :class:`SandboxUnavailable`.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import mimetypes
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pydantic

from vdreamer import metrics
from vdreamer.artifacts import ArtifactKind, ArtifactRef, aemit_artifact, run_blocking

if TYPE_CHECKING:
    from vdreamer.context import VDreamerContext

SANDBOX_UID_ENV: Final = "VDREAMER_SANDBOX_UID"
SANDBOX_GID_ENV: Final = "VDREAMER_SANDBOX_GID"
SANDBOX_ROOT_ENV: Final = "VDREAMER_SANDBOX_ROOT"

DEFAULT_TIMEOUT_S: Final = 60.0
#: What is kept of each output stream (the rest is drained and dropped).
STREAM_CAP_BYTES: Final = 64 * 1024
MAX_OUTPUT_FILES: Final = 32
MAX_OUTPUT_BYTES: Final = 256 * 1024 * 1024
_INPUT_NAME: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RESERVED_NAMES: Final = frozenset({"out", "main.py", ".tmp"})


class SandboxUnavailable(RuntimeError):
    """The sandbox cannot run here (not POSIX, or the uid switch is impossible)."""


@dataclass(frozen=True)
class SandboxLimits:
    """The rlimits (A19). ``cpu_s=None`` means ``ceil(timeout_s) + 1``."""

    memory_bytes: int = 2 * 1024 * 1024 * 1024
    file_size_bytes: int = 256 * 1024 * 1024
    nproc: int = 64
    nofile: int = 256
    cpu_s: int | None = None


class SandboxResult(pydantic.BaseModel):
    """What one sandbox run produced."""

    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    duration_ms: int
    artifacts: list[ArtifactRef] = pydantic.Field(default_factory=list)
    #: Output files left out (symlinks, special files, over the caps).
    skipped_outputs: list[str] = pydantic.Field(default_factory=list)
    network_isolated: bool = False
    uid: int | None = None
    dedicated_uid: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def unavailable_reason() -> str | None:
    """Why the sandbox cannot run on this host, or ``None`` when it can."""
    if os.name != "posix":
        return "the sandbox needs a POSIX worker (D19); Windows hosts cannot run it"
    return None


def _sandbox_ids(environ: Mapping[str, str] | None = None) -> tuple[int | None, int | None]:
    """The configured ``(uid, gid)``; ``(None, None)`` keeps the worker's own."""
    env = os.environ if environ is None else environ
    raw = env.get(SANDBOX_UID_ENV, "").strip()
    if not raw:
        return None, None
    if raw.isdigit():
        uid = int(raw)
        gid: int | None = None
    else:
        if sys.platform == "win32":
            raise SandboxUnavailable(unavailable_reason() or "POSIX only")
        import pwd

        try:
            entry = pwd.getpwnam(raw)
        except KeyError as exc:
            raise SandboxUnavailable(f"{SANDBOX_UID_ENV}: no such user {raw!r}") from exc
        uid, gid = entry.pw_uid, entry.pw_gid
    raw_gid = env.get(SANDBOX_GID_ENV, "").strip()
    if raw_gid:
        gid = int(raw_gid)
    return uid, gid if gid is not None else uid


def sandbox_root(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    configured = env.get(SANDBOX_ROOT_ENV, "").strip()
    return Path(configured) if configured else Path(tempfile.gettempdir()) / "vdreamer-sandbox"


def sandbox_env(scratch: Path) -> dict[str, str]:
    """The child's whole environment: sandbox-local paths, nothing inherited."""
    tmp = scratch / ".tmp"
    return {
        "HOME": str(scratch),
        "TMPDIR": str(tmp),
        "MPLCONFIGDIR": str(tmp),
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "LANG": "C.UTF-8",
    }


@dataclass
class Completed:
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    duration_ms: int
    network_isolated: bool
    uid: int | None


async def _drain(stream: asyncio.StreamReader | None) -> bytes:
    if stream is None:
        return b""
    kept = bytearray()
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return bytes(kept)
        if len(kept) < STREAM_CAP_BYTES:
            kept.extend(chunk[: STREAM_CAP_BYTES - len(kept)])


def _kill_uid(uid: int) -> None:
    """Kill whatever the dedicated uid still runs (a child may have left the group)."""
    if sys.platform == "win32":
        return
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        with contextlib.suppress(OSError):
            if entry.stat().st_uid == uid:
                os.kill(int(entry.name), signal.SIGKILL)


async def run_confined(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float,
    limits: SandboxLimits | None = None,
    allow_unconfined: bool = False,
) -> Completed:
    """Run ``argv`` (a Python command line) confined in ``cwd``; never raises on timeout.

    ``allow_unconfined`` lets ``safe_extract`` — whose caps are enforced in
    its own code — run on hosts without the launcher (Windows dev); the
    sandbox proper never sets it.
    """
    caps = limits or SandboxLimits()
    reason = unavailable_reason()
    started = time.monotonic()
    env = sandbox_env(cwd)
    (cwd / ".tmp").mkdir(exist_ok=True)
    uid, gid = (None, None) if reason is not None else _sandbox_ids()
    status_r = status_w = -1
    if reason is None:
        cpu = caps.cpu_s if caps.cpu_s is not None else int(timeout_s) + 2
        rlimits: dict[str, int] = {
            "cpu": cpu,
            "as": caps.memory_bytes,
            "fsize": caps.file_size_bytes,
            "nofile": caps.nofile,
            "core": 0,
        }
        if sys.platform == "win32":  # unavailable_reason() is None only on POSIX
            raise SandboxUnavailable("POSIX only")
        if uid is not None:
            rlimits["nproc"] = caps.nproc
            if os.geteuid() != 0:
                raise SandboxUnavailable(
                    f"{SANDBOX_UID_ENV} is set but the worker is not root: it cannot switch uid"
                )
            for path in [cwd, *cwd.rglob("*")]:
                os.chown(path, uid, gid if gid is not None else uid)
        status_r, status_w = os.pipe()
        config = {
            "argv": list(argv),
            "env": env,
            "cwd": str(cwd),
            "uid": uid,
            "gid": gid,
            "rlimits": rlimits,
            "status_fd": status_w,
            "isolate_network": True,
        }
        command = [sys.executable, "-I", "-B", "-m", "vdreamer._sandbox_launch", json.dumps(config)]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={},
            start_new_session=True,
            pass_fds=(status_w,),
        )
        os.close(status_w)
    elif allow_unconfined:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env={**env, "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")},
        )
    else:
        raise SandboxUnavailable(reason)
    out_task = asyncio.ensure_future(_drain(process.stdout))
    err_task = asyncio.ensure_future(_drain(process.stderr))
    timed_out = False
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_s)
    except TimeoutError:
        timed_out = True
        with contextlib.suppress(ProcessLookupError, PermissionError):
            if sys.platform != "win32" and reason is None:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        await process.wait()
    finally:
        if uid is not None:
            _kill_uid(uid)
    stdout, stderr = await out_task, await err_task
    status: dict[str, Any] = {}
    if status_r >= 0:
        with contextlib.suppress(OSError, ValueError):
            raw = os.read(status_r, 4096)
            status = json.loads(raw) if raw else {}
        os.close(status_r)
    if timed_out:
        metrics.SANDBOX_TIMEOUTS.inc()
    return Completed(
        exit_code=None if timed_out else process.returncode,
        timed_out=timed_out,
        stdout=stdout.decode("utf-8", "replace"),
        stderr=stderr.decode("utf-8", "replace"),
        duration_ms=int((time.monotonic() - started) * 1000),
        network_isolated=bool(status.get("network_isolated")),
        uid=status.get("uid") if isinstance(status.get("uid"), int) else None,
    )


@contextlib.contextmanager
def scratch_dir() -> Any:
    """A fresh scratch directory, wiped on exit whatever happened."""
    root = sandbox_root()
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _read_output(path: Path) -> bytes | None:
    """A regular, singly-linked file's bytes — opened without following symlinks."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        import stat as stat_mod

        if not stat_mod.S_ISREG(info.st_mode) or info.st_nlink != 1:
            return None
        return handle.read()


def _output_artifact(name: str, data: bytes) -> tuple[ArtifactKind, Any]:
    """The artifact kind and data for one ``out/`` file, by extension."""
    suffix = Path(name).suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".svg"):
        return ArtifactKind.IMAGE, data
    if suffix in (".html", ".htm"):
        return ArtifactKind.HTML, data.decode("utf-8", "replace")
    if suffix == ".json":
        with contextlib.suppress(ValueError):
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return ArtifactKind.JSON, parsed
    if suffix in (".parquet", ".csv"):
        with contextlib.suppress(Exception):
            if suffix == ".parquet":
                import pyarrow.parquet as pq

                return ArtifactKind.TABLE, pq.read_table(io.BytesIO(data))
            import pyarrow.csv as pacsv

            return ArtifactKind.TABLE, pacsv.read_csv(io.BytesIO(data))
    return ArtifactKind.FILE, data


async def collect_outputs(out_dir: Path) -> tuple[list[ArtifactRef], list[str]]:
    """Turn ``out/``'s regular files into artifacts (D7 order, one by one)."""
    refs: list[ArtifactRef] = []
    skipped: list[str] = []
    total = 0
    if not out_dir.is_dir() or out_dir.is_symlink():
        return refs, skipped
    for root, dirs, files in os.walk(out_dir, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(root) / d).is_symlink())
        for filename in sorted(files):
            path = Path(root) / filename
            relative = path.relative_to(out_dir).as_posix()
            if len(refs) >= MAX_OUTPUT_FILES:
                skipped.append(relative)
                continue
            data = await asyncio.to_thread(_read_output, path)
            if data is None or total + len(data) > MAX_OUTPUT_BYTES:
                skipped.append(relative)
                continue
            total += len(data)
            kind, payload = await asyncio.to_thread(_output_artifact, relative, data)
            metadata = {"source": "sandbox", "path": relative}
            try:
                ref = await aemit_artifact(relative, kind, payload, metadata, filename=filename)
            except (TypeError, ValueError):
                mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                ref = await aemit_artifact(
                    relative, ArtifactKind.FILE, data, metadata, mime_type=mime, filename=filename
                )
            refs.append(ref)
    return refs, skipped


InputValue = Any  # bytes | str | os.PathLike | FileRef | ManifestEntry


class Sandbox:
    """``ctx.sandbox`` — bound to one run's context."""

    def __init__(self, ctx: VDreamerContext) -> None:
        self._ctx = ctx

    async def _write_inputs(self, scratch: Path, inputs: Mapping[str, InputValue]) -> None:
        from vdreamer.files import Files

        files = Files(self._ctx)
        for name, value in inputs.items():
            if not _INPUT_NAME.fullmatch(name) or name in _RESERVED_NAMES:
                raise ValueError(f"invalid sandbox input name {name!r} (a plain file name)")
            target = scratch / name
            if isinstance(value, (bytes, bytearray, memoryview)):
                target.write_bytes(bytes(value))
            elif isinstance(value, str):
                target.write_text(value, encoding="utf-8")
            elif isinstance(value, os.PathLike):
                await asyncio.to_thread(shutil.copyfile, os.fspath(value), target)
            else:
                await files.adownload(value, target)

    async def arun(
        self,
        code: str,
        inputs: Mapping[str, InputValue] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        *,
        limits: SandboxLimits | None = None,
    ) -> SandboxResult:
        """Run ``code`` (a Python program) confined; ``./out/`` files become artifacts.

        ``inputs`` maps plain file names to bytes, text, a local path, a
        ``FileRef``, or a ``safe_extract`` manifest entry; each lands in the
        working directory under its name.
        """
        reason = unavailable_reason()
        if reason is not None:
            raise SandboxUnavailable(reason)
        with scratch_dir() as scratch:
            (scratch / "out").mkdir()
            (scratch / "main.py").write_text(code, encoding="utf-8")
            await self._write_inputs(scratch, inputs or {})
            done = await run_confined(
                [sys.executable, "-I", "-B", "main.py"],
                cwd=scratch,
                timeout_s=timeout_s,
                limits=limits,
            )
            refs, skipped = await collect_outputs(scratch / "out")
        uid, _ = _sandbox_ids()
        return SandboxResult(
            exit_code=done.exit_code,
            timed_out=done.timed_out,
            stdout=done.stdout,
            stderr=done.stderr,
            duration_ms=done.duration_ms,
            artifacts=refs,
            skipped_outputs=skipped,
            network_isolated=done.network_isolated,
            uid=done.uid,
            dedicated_uid=uid is not None,
        )

    def run(
        self,
        code: str,
        inputs: Mapping[str, InputValue] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        *,
        limits: SandboxLimits | None = None,
    ) -> SandboxResult:
        """The synchronous :meth:`arun` (any node or tool)."""
        return run_blocking(self.arun(code, inputs, timeout_s, limits=limits))
