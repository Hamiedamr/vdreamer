"""``resources.yaml`` loading (PRD §3.7, A35).

The file maps an alias to one resource entry, each with a ``type``:
``{vectors: {type: qdrant, ...}, netinv: {type: mcp, ...}}``. Milestone 5
consumes the ``type: mcp`` entries (``vdreamer.mcp``, §2.7, A39); the other
adapters (qdrant, neo4j, sql) and ``ctx.resources[alias]`` land with
Milestone 7 — until then their entries are loaded, shape-checked, and left
alone.

Secrets never appear in YAML (A23): an entry references environment
variables through ``*_env`` keys, and a key that would hold a secret inline
(``password``, ``token``, ``api_key``, …) is refused at load.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import yaml

#: Where the CLI looks when ``--resources`` is not given.
RESOURCES_ENV: Final = "VDREAMER_RESOURCES"

#: Keys that would carry a secret inline (A23: reference ``*_env`` instead).
SECRET_KEYS: Final = frozenset(
    {"password", "token", "secret", "api_key", "apikey", "bearer", "authorization", "credentials"}
)


class ResourcesError(ValueError):
    """``resources.yaml`` is missing, unparsable, or has the wrong shape."""


def check_entry(alias: str, entry: Any) -> dict[str, Any]:
    """One entry's common shape: a mapping with a string ``type`` and no inline secret."""
    if not isinstance(entry, Mapping):
        raise ResourcesError(f"resource {alias!r}: expected a mapping, got {type(entry).__name__}")
    kind = entry.get("type")
    if not isinstance(kind, str) or not kind:
        raise ResourcesError(f"resource {alias!r}: needs a 'type' (e.g. mcp, qdrant, neo4j, sql)")
    inline = sorted(str(key) for key in entry if str(key).lower() in SECRET_KEYS)
    if inline:
        raise ResourcesError(
            f"resource {alias!r}: {', '.join(inline)} would put a secret in YAML (A23); "
            "reference an environment variable with a *_env key instead"
        )
    return dict(entry)


def parse_resources(data: Any, *, origin: str = "resources.yaml") -> dict[str, dict[str, Any]]:
    """Validate a loaded document: ``{alias: {type: ..., ...}}``."""
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ResourcesError(f"{origin}: expected a mapping of alias -> entry")
    resources: dict[str, dict[str, Any]] = {}
    for alias, entry in data.items():
        if not isinstance(alias, str) or not alias:
            raise ResourcesError(f"{origin}: aliases must be non-empty strings, got {alias!r}")
        resources[alias] = check_entry(alias, entry)
    return resources


def load_resources(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate ``resources.yaml`` (``yaml.safe_load``)."""
    file = Path(path)
    try:
        raw = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ResourcesError(f"cannot read {file}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ResourcesError(f"{file}: invalid YAML: {exc}") from exc
    return parse_resources(data, origin=str(file))


def resolve_resources_path(path: str | Path | None) -> Path | None:
    """The explicit path, else ``$VDREAMER_RESOURCES``, else none."""
    if path is not None:
        return Path(path)
    from_env = os.environ.get(RESOURCES_ENV)
    return Path(from_env) if from_env else None
