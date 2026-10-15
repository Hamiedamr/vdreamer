"""Load a project's ``.env`` file into the environment.

Every ``vdreamer`` command reads ``.env`` from the working directory when
there is one (or the file ``--env-file`` names), so the token, DSNs and the
keys ``models.yaml`` / ``resources.yaml`` reference through ``*_env`` can live
next to the project instead of in the shell. Variables already set in the
environment win: a deployment's real environment is never overridden by a
file. The zero-egress switches (:mod:`vdreamer.offline`) are applied after
this, so a ``.env`` cannot turn telemetry back on.

The format is the common one: ``KEY=value`` per line, an optional ``export``
prefix, ``#`` comments (a ``#`` after unquoted text needs a space before it),
blank lines, and single- or double-quoted values (double quotes understand
``\\n``, ``\\t``, ``\\"`` and ``\\\\``). There is no variable expansion.
"""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

#: The file read from the working directory when ``--env-file`` is not given.
DEFAULT_ENV_FILE = ".env"

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


class EnvFileError(ValueError):
    """A ``.env`` line that is not ``KEY=value``; the message names the line."""


@dataclass
class EnvFile:
    """What one file held, and which of its keys were applied."""

    path: Path
    values: dict[str, str] = field(default_factory=dict)
    applied: list[str] = field(default_factory=list)


def _unquote(raw: str, where: str) -> str:
    if raw[:1] == "'":
        end = raw.find("'", 1)
        if end < 0:
            raise EnvFileError(f"{where}: unterminated single quote")
        return raw[1:end]
    if raw[:1] == '"':
        out: list[str] = []
        index = 1
        while index < len(raw):
            char = raw[index]
            if char == "\\" and index + 1 < len(raw):
                nxt = raw[index + 1]
                out.append(_ESCAPES.get(nxt, "\\" + nxt))
                index += 2
                continue
            if char == '"':
                return "".join(out)
            out.append(char)
            index += 1
        raise EnvFileError(f"{where}: unterminated double quote")
    comment = re.search(r"\s#", raw)
    return (raw[: comment.start()] if comment else raw).strip()


def parse_env_file(path: Path) -> dict[str, str]:
    """``KEY → value`` from one file, in file order (later lines win)."""
    values: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _LINE.match(line)
        if match is None:
            raise EnvFileError(f"{path}:{number}: expected KEY=value")
        key, raw = match.groups()
        values[key] = _unquote(raw, f"{path}:{number}")
    return values


def load_env_file(
    path: str | Path | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
    cwd: Path | None = None,
) -> EnvFile | None:
    """Apply ``path`` (default: ``.env`` in the working directory) to ``environ``.

    Keys already in ``environ`` are left alone. Returns ``None`` when no path
    was given and the working directory has no ``.env``.

    Raises:
        FileNotFoundError: an explicit ``path`` that does not exist.
        EnvFileError: a malformed line.
    """
    env = environ if environ is not None else os.environ
    if path is None:
        candidate = (cwd or Path.cwd()) / DEFAULT_ENV_FILE
        if not candidate.is_file():
            return None
        file = candidate
    else:
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"env file not found: {file}")
    loaded = EnvFile(file, parse_env_file(file))
    for key, value in loaded.values.items():
        if key not in env:
            env[key] = value
            loaded.applied.append(key)
    return loaded
