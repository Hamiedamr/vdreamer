"""Gateway kit for the tutorial tests: real gateway, real workers, ``vdreamer run``.

Unit tests drive a lesson's agent in-process with
``vdreamer_tutorials.testing.AgentSession``. Integration tests here serve it the
way ``vdreamer dev`` does — a uvicorn gateway over TCP, a worker per agent, the
run queue, the event log, interrupts and artifact stores — and talk to it with
``vdreamer run`` (``run_terminal``), reusing the repo's acceptance kit
(``tests/acceptance/_uc_kit.py``) and API kit (``tests/api/_api.py``).

Only the chat model is scripted (``ScriptedChatModel``). A fresh stack per
``served(...)`` block keeps each test's script independent.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
REPO = TESTS.parent
TUTORIALS = REPO / "tutorials"
LESSONS = TUTORIALS / "src" / "vdreamer_tutorials"
for kit_dir in (TESTS / "api", TESTS / "acceptance"):
    if str(kit_dir) not in sys.path:
        sys.path.insert(0, str(kit_dir))

from _api import SERVICE_TOKEN, TOKEN, make_stack  # noqa: E402
from _uc_kit import CliResult, serve, vdreamer_run, wait_status  # noqa: E402

from vdreamer import BaseAgent  # noqa: E402

__all__ = [
    "LESSONS",
    "REPO",
    "SERVICE_TOKEN",
    "TESTS",
    "TOKEN",
    "TUTORIALS",
    "CliResult",
    "served",
    "vdreamer_run",
    "wait_status",
]

#: Per agent name: a factory for the runner's backends, e.g. ``{"models": {"chat": m}}``.
Backends = Mapping[str, Callable[[], Mapping[str, Any]]]


def _opener(make: Callable[[], Mapping[str, Any]]) -> Any:
    @asynccontextmanager
    async def backends() -> AsyncIterator[Mapping[str, Any]]:
        yield make()

    return backends


@contextmanager
def served(agents: Sequence[BaseAgent], backends: Backends | None = None) -> Iterator[str]:
    """Serve ``agents`` behind a real gateway; yields the base URL.

    ``backends[name]()`` returns the agent's ``AgentRunner`` options — usually
    ``{"models": {"chat": scripted(...)}}``, optionally a ``store``.
    """
    stack = make_stack(
        workers=True,
        extra_agents=agents,
        worker_agents=[agent.name for agent in agents],
        poll_interval_s=0.05,
        runner_options={name: _opener(make) for name, make in (backends or {}).items()},
    )
    try:
        with serve(stack) as url:
            yield url
    finally:
        stack.close()
