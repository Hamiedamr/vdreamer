"""``vdreamer migrate`` — apply the vdreamer schema (PRD §2.3; M3 WP3).

Creates the Milestone-3 tables — ``runs`` (:data:`vdreamer.queue.RUNS_DDL`)
and ``run_events`` (:data:`vdreamer.eventlog.RUN_EVENTS_DDL`) — plus the
checkpointer's own tables (``AsyncPostgresSaver.setup()`` runs its
migrations). Every step is ``CREATE ... IF NOT EXISTS`` idempotent: running
``migrate`` again on a migrated database is a no-op.

The DSN comes from the environment only (A23): ``VDREAMER_PG_DSN``, resolved
by :func:`vdreamer.persistence.dsn_from_env` — never a command-line argument,
never YAML. Later milestones extend this command as their tables land
(artifacts, interrupts); the DDL constants they reuse already live beside
their writers.
"""

from __future__ import annotations

import asyncio
import sys

from vdreamer.eventlog import PostgresEventLog
from vdreamer.persistence import (
    PG_DSN_ENV_VAR,
    MissingPostgresDependencyError,
    PersistenceConfigError,
    dsn_from_env,
    open_checkpointer,
)
from vdreamer.queue import PostgresRunQueue


async def _apply(dsn: str) -> None:
    """Apply the schema to ``dsn``, printing one line per applied piece."""
    await PostgresRunQueue(dsn).setup()
    print("[ok] runs table + active-thread index (vdreamer.queue.RUNS_DDL)")

    log = await PostgresEventLog.connect(dsn)
    try:
        await log.setup()
    finally:
        await log.close()
    print("[ok] run_events table (vdreamer.eventlog.RUN_EVENTS_DDL)")

    # The checkpointer's migrations are langgraph's to run — reuse the same
    # factory the worker uses so ``migrate`` and ``worker`` never diverge.
    async with open_checkpointer(dsn):
        print("[ok] checkpointer tables (AsyncPostgresSaver.setup)")


def run_migrate() -> int:
    """CLI entry point for ``vdreamer migrate``. 0 = schema applied (or already there)."""
    try:
        dsn = dsn_from_env(PG_DSN_ENV_VAR)
    except PersistenceConfigError as exc:
        print(f"migrate: {exc}", file=sys.stderr)
        return 1
    try:
        asyncio.run(_apply(dsn))
    except MissingPostgresDependencyError as exc:
        print(f"migrate: {exc}", file=sys.stderr)
        return 1
    print("migrate: PASS")
    return 0
