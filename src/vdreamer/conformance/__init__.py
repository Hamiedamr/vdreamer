"""The protocol 1.0 conformance kit: recorded SSE streams + expected client state.

Client teams test their SSE handling and event fold against these files —
vdreamer ships no client code, so the kit *is* the executable contract.
``vdreamer export-schema`` writes it to ``<out>/conformance/``; every release
attaches it. ``README.md`` (beside the recordings) defines the file format and
the fold; ``docs/protocol.md`` the normative client behavior around it.

The recordings are real gateway output, captured from the in-process REST/SSE
stack by the repo's recorder (``tests/conformance/record_kit.py``) with every
generated id and timestamp normalized (``run_1``, ``int_1``, …) so a
re-recording diffs cleanly. The one exception, ``forward_compat``, is
hand-written: the things a 1.0 client must survive that 1.0 never emits.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, Final

#: The recordings, in the order the README presents them.
RECORDINGS: Final[tuple[str, ...]] = (
    "chat",
    "reconnect",
    "tool_overflow",
    "interrupt_require_approval",
    "interrupt_tool_review",
    "interrupt_mcp_elicitation",
    "interrupt_generic",
    "artifacts",
    "citations",
    "forward_compat",
)


def kit_files() -> dict[str, bytes]:
    """Every kit file by name (``<recording>.json`` and ``README.md``)."""
    root = resources.files(__package__).joinpath("kit")
    files: dict[str, bytes] = {}
    for entry in root.iterdir():
        if entry.name.endswith((".json", ".md")):
            files[entry.name] = entry.read_bytes()
    return dict(sorted(files.items()))


def load_recording(name: str) -> dict[str, Any]:
    """One recording, parsed."""
    data = resources.files(__package__).joinpath("kit", f"{name}.json").read_bytes()
    loaded: dict[str, Any] = json.loads(data)
    return loaded


__all__ = ["RECORDINGS", "kit_files", "load_recording"]
