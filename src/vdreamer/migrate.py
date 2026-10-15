"""``vdreamer migrate`` — apply the vdreamer schema.

Creates the vdreamer tables — ``runs`` (:data:`vdreamer.queue.RUNS_DDL`, which
also upgrades an older table in place), ``run_events``
(:data:`vdreamer.eventlog.RUN_EVENTS_DDL`), ``threads``
(:data:`vdreamer.threads.THREADS_DDL`), ``interrupts``
(:data:`vdreamer.interrupts.INTERRUPTS_DDL`), ``outbox``
(:data:`vdreamer.outbox.OUTBOX_DDL`) and ``artifacts``
(:data:`vdreamer.artifacts.ARTIFACTS_DDL`) — plus the checkpointer's own
tables (``AsyncPostgresSaver.setup()`` runs its
migrations). Every step is ``CREATE ... IF NOT EXISTS`` idempotent: running
``migrate`` again on a migrated database is a no-op.

The DSN comes from the environment only: ``VDREAMER_PG_DSN``, resolved
by :func:`vdreamer.persistence.dsn_from_env` — never a command-line argument,
never YAML. With ``VDREAMER_MINIO_*`` set it also creates the MinIO bucket
and its 30-day lifecycle rule on ``artifacts/`` and ``extracted/``.
Later milestones extend this command as their tables land; the DDL constants
live beside their writers.
"""

from __future__ import annotations

import asyncio
import sys

from vdreamer.artifacts import PostgresArtifactStore, UnavailableArtifactStore
from vdreamer.eventlog import PostgresEventLog
from vdreamer.interrupts import PostgresInterruptStore
from vdreamer.objects import (
    MINIO_ENDPOINT_ENV,
    RETENTION_DAYS,
    MinioConfigError,
    MinioObjectStore,
)
from vdreamer.outbox import PostgresOutboxStore
from vdreamer.persistence import (
    PG_DSN_ENV_VAR,
    MissingPostgresDependencyError,
    PersistenceConfigError,
    dsn_from_env,
    open_checkpointer,
    open_store,
)
from vdreamer.queue import PostgresRunQueue
from vdreamer.threads import PostgresThreadStore


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

    await PostgresThreadStore(dsn).setup()
    print("[ok] threads table (vdreamer.threads.THREADS_DDL)")

    await PostgresInterruptStore(dsn).setup()
    print("[ok] interrupts table (vdreamer.interrupts.INTERRUPTS_DDL)")

    await PostgresOutboxStore(dsn).setup()
    print("[ok] outbox table (vdreamer.outbox.OUTBOX_DDL)")

    objects = MinioObjectStore.from_env()
    await PostgresArtifactStore(dsn, objects or UnavailableArtifactStore().objects).setup()
    print("[ok] artifacts table (vdreamer.artifacts.ARTIFACTS_DDL)")
    if objects is None:
        print(f"[--] MinIO bucket skipped ({MINIO_ENDPOINT_ENV} unset)")
    else:
        await asyncio.to_thread(objects.setup)
        print(f"[ok] MinIO bucket {objects.bucket!r} + {RETENTION_DAYS}-day lifecycle")

    # The checkpointer's migrations are langgraph's to run — reuse the same
    # factory the worker uses so ``migrate`` and ``worker`` never diverge.
    async with open_checkpointer(dsn):
        print("[ok] checkpointer tables (AsyncPostgresSaver.setup)")
    async with open_store(dsn):
        print("[ok] store tables (AsyncPostgresStore.setup)")


def run_migrate() -> int:
    """CLI entry point for ``vdreamer migrate``. 0 = schema applied (or already there)."""
    try:
        dsn = dsn_from_env(PG_DSN_ENV_VAR)
    except PersistenceConfigError as exc:
        print(f"migrate: {exc}", file=sys.stderr)
        return 1
    try:
        asyncio.run(_apply(dsn))
    except (MissingPostgresDependencyError, MinioConfigError) as exc:
        print(f"migrate: {exc}", file=sys.stderr)
        return 1
    print("migrate: PASS")
    return 0
