"""Zero-egress runtime settings: applied by every ``vdreamer`` command.

The VM network policy is the authoritative control; these settings make
the libraries vdreamer loads never *try*: LangSmith tracing is forced off, the
HuggingFace/Transformers stacks run offline, and every telemetry or
update-check switch the dependency tree honours is turned off. They are
*forced* (not defaults): an operator's ``LANGSMITH_TRACING=true`` would be a
configuration error in an air-gapped deployment, so it is overridden.

``TIKTOKEN_CACHE_DIR`` is the one setting that points at data rather than
switching something off: ``langchain-openai`` counts tokens with tiktoken,
which downloads its encodings on first use unless they are in this cache.
``vdreamer bundle`` fills ``<out>/tiktoken/`` on the connected host; the
operator ships it and sets the variable. :func:`tiktoken_cache_problem`
is what ``doctor --offline`` reports.

``vdreamer.models`` also sets the two HF variables right before it loads
fastembed, for library users who never go through the CLI.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Final

#: Forced before the stack is imported. ``doctor --offline`` asserts each.
OFFLINE_ENV: Final[dict[str, str]] = {
    # LangSmith tracing off (both spellings LangChain reads).
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false",
    # HuggingFace hub / Transformers: never reach the hub (fastembed loads local files only).
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    # FastMCP's update check (the [mcp] extra).
    "FASTMCP_CHECK_FOR_UPDATES": "off",
    # The cross-tool opt-out (https://consoledonottrack.com — an identifier, never fetched).
    "DO_NOT_TRACK": "1",
}

TIKTOKEN_CACHE_ENV: Final = "TIKTOKEN_CACHE_DIR"

#: The encodings ``vdreamer bundle`` pre-fetches into the tiktoken cache.
TIKTOKEN_ENCODINGS: Final[tuple[str, ...]] = ("cl100k_base", "o200k_base")


def apply_offline_env(environ: MutableMapping[str, str] | None = None) -> None:
    """Force :data:`OFFLINE_ENV` into ``environ`` (the process environment)."""
    env = environ if environ is not None else os.environ
    env.update(OFFLINE_ENV)


def tiktoken_cache_problem(environ: MutableMapping[str, str] | None = None) -> str | None:
    """Why tiktoken could still reach the network, or ``None`` when it cannot.

    Set and non-empty: fine. Set but missing or empty: a problem (the bundle
    was not shipped). Unset: fine only when tiktoken is not installed.
    """
    env = environ if environ is not None else os.environ
    value = env.get(TIKTOKEN_CACHE_ENV, "").strip()
    if value:
        path = Path(value)
        if not path.is_dir():
            return f"{TIKTOKEN_CACHE_ENV}={value!r} is not a directory"
        if not any(path.iterdir()):
            return f"{TIKTOKEN_CACHE_ENV}={value!r} is empty (ship `vdreamer bundle`'s tiktoken/)"
        return None
    import importlib.util

    if importlib.util.find_spec("tiktoken") is None:
        return None
    return (
        f"{TIKTOKEN_CACHE_ENV} is unset: tiktoken would download its encodings on first use "
        "(set it to the `vdreamer bundle` tiktoken/ directory)"
    )


__all__ = [
    "OFFLINE_ENV",
    "TIKTOKEN_CACHE_ENV",
    "TIKTOKEN_ENCODINGS",
    "apply_offline_env",
    "tiktoken_cache_problem",
]
