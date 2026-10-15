"""``vdreamer dev --reload`` — restart the dev process when its sources change (§2.3).

A supervisor, not an in-process reloader: it runs ``python -m vdreamer dev …``
(the same arguments, minus ``--reload``) as a child and restarts it when a
watched file changes. Everything the child knew lives in the SQLite ``--db``
file (D11), so a restart loses no thread, run or interrupt — a run in flight
is recovered like any worker crash (A6).

Watched: ``*.py``, ``*.yaml``/``*.yml`` and ``*.json`` under the working
directory, the ``PYTHONPATH`` entries, ``--agents-dir`` and the directories of
``--registry``/``--models``/``--resources`` — minus virtualenvs, VCS and cache
directories, and the dev state directory. Polling (no watcher dependency, no
platform specifics); changes are debounced so an editor's save burst restarts
once.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Final

WATCHED_SUFFIXES: Final[frozenset[str]] = frozenset({".py", ".yaml", ".yml", ".json"})

IGNORED_DIRS: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hg",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "node_modules",
        ".vdreamer",
        "dist",
        "build",
    }
)

POLL_INTERVAL_S: Final = 0.5

Snapshot = dict[str, int]


def snapshot(roots: Iterable[Path], ignored: Iterable[Path] = ()) -> Snapshot:
    """``{path: mtime_ns}`` of every watched file under ``roots``."""
    skip = {path.resolve() for path in ignored}
    files: Snapshot = {}
    for root in roots:
        if root.is_file():
            if root.suffix in WATCHED_SUFFIXES:
                files[str(root.resolve())] = root.stat().st_mtime_ns
            continue
        if not root.is_dir():
            continue
        for directory, subdirs, names in os.walk(root):
            here = Path(directory)
            subdirs[:] = [
                name
                for name in subdirs
                if name not in IGNORED_DIRS
                and not name.endswith(".egg-info")
                and (here / name).resolve() not in skip
            ]
            for name in names:
                path = here / name
                if path.suffix in WATCHED_SUFFIXES:
                    try:
                        files[str(path.resolve())] = path.stat().st_mtime_ns
                    except OSError:  # vanished between listing and stat
                        continue
    return files


def changes(before: Snapshot, after: Snapshot) -> list[str]:
    """The files added, removed or modified between two snapshots."""
    changed = {path for path in before.keys() ^ after.keys()}
    changed |= {path for path in before.keys() & after.keys() if before[path] != after[path]}
    return sorted(changed)


def watch_roots(
    *,
    cwd: Path,
    agents_dir: str | None = None,
    files: Sequence[str | None] = (),
    pythonpath: str | None = None,
) -> list[Path]:
    """The directories (and files) ``dev --reload`` watches."""
    roots = [cwd]
    for entry in (pythonpath or "").split(os.pathsep):
        if entry.strip():
            roots.append(Path(entry))
    if agents_dir:
        roots.append(Path(agents_dir))
    for path in files:
        if path:
            roots.append(Path(path).parent)
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if not any(resolved == seen or seen in resolved.parents for seen in unique):
            unique = [seen for seen in unique if resolved not in seen.parents]
            unique.append(resolved)
    return unique


def child_argv(argv: Sequence[str]) -> list[str]:
    """``python -m vdreamer <argv without --reload>``."""
    return [sys.executable, "-m", "vdreamer", *[arg for arg in argv if arg != "--reload"]]


Spawn = Callable[[list[str]], Any]


def _stop(process: Any, timeout_s: float = 5.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def supervise(
    command: list[str],
    roots: Sequence[Path],
    *,
    ignored: Sequence[Path] = (),
    interval_s: float = POLL_INTERVAL_S,
    spawn: Spawn = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    max_restarts: int | None = None,
) -> int:
    """Run ``command`` and restart it on every change under ``roots``.

    Returns when the child exits on its own (its exit code), or after
    ``max_restarts`` restarts (tests). Ctrl-C stops the child and returns 130.
    """
    restarts = 0
    process = spawn(command)
    seen = snapshot(roots, ignored)
    try:
        while True:
            sleep(interval_s)
            code = process.poll()
            if code is not None:
                return int(code)
            current = snapshot(roots, ignored)
            changed = changes(seen, current)
            if not changed:
                continue
            sleep(interval_s)  # debounce an editor's save burst
            current = snapshot(roots, ignored)
            shown = ", ".join(Path(path).name for path in changed[:3])
            more = f" (+{len(changed) - 3} more)" if len(changed) > 3 else ""
            print(f"vdreamer dev: {shown}{more} changed — restarting", flush=True)
            _stop(process)
            restarts += 1
            if max_restarts is not None and restarts > max_restarts:
                return 0
            process = spawn(command)
            seen = current
    except KeyboardInterrupt:
        _stop(process)
        return 130


def run_dev_reload(
    argv: Sequence[str],
    *,
    db: str,
    agents_dir: str | None,
    registry: str | None,
    models: str | None,
    resources: str | None,
) -> int:
    """CLI entry point for ``vdreamer dev --reload``."""
    roots = watch_roots(
        cwd=Path.cwd(),
        agents_dir=agents_dir,
        files=(registry, models, resources),
        pythonpath=os.environ.get("PYTHONPATH"),
    )
    print("vdreamer dev: --reload watching " + ", ".join(root.as_posix() for root in roots))
    state_dir = Path(db).parent.resolve()
    ignored = [] if state_dir in roots else [state_dir]  # never ignore a whole root
    return supervise(child_argv(argv), roots, ignored=ignored)


__all__ = [
    "IGNORED_DIRS",
    "changes",
    "child_argv",
    "run_dev_reload",
    "snapshot",
    "supervise",
    "watch_roots",
]
