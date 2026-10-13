"""``vdreamer`` CLI (PRD §2.3).

Milestone 0 surface: ``--version``, ``validate``, ``doctor --offline`` — the
pieces the CI matrix depends on. Milestone 1 wires the D2 discovery flags into
``validate``: ``--registry PATH`` / ``--agents-dir PATH`` resolve a bare agent
name (``vdreamer.discovery.resolve_agent``), while a ``pkg.module:agent`` spec
stays explicit and wins. Milestone 3 adds the topology entrypoints: ``worker``
(one process, one agent, Postgres queue + checkpointer — A7) and ``dev`` (the
D11 single-process mode on SQLite + in-memory queue), plus ``migrate`` (the
idempotent schema application, DSN from ``VDREAMER_PG_DSN``, A23). ``serve``
lands with Milestone 4, ``run`` with Milestone 11, and
``export-schema``/``bundle`` with Milestone 10.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from vdreamer import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="vdreamer",
        description="Serve LangGraph agents as a production REST/SSE API (protocol version 1.0).",
    )
    parser.add_argument("--version", action="version", version=f"vdreamer {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate an agent against the vdreamer contract",
        description=(
            "Load 'pkg.module:agent' (or resolve a bare name via --registry / --agents-dir, D2) "
            "and run the developer-contract checks."
        ),
    )
    validate_parser.add_argument(
        "agent",
        metavar="pkg.module:agent",
        help="agent spec (module path and attribute), or a bare name with --registry/--agents-dir",
    )
    validate_parser.add_argument(
        "--registry",
        metavar="PATH",
        default=None,
        help="path to a vdreamer.yaml registry mapping names to import specs (D2)",
    )
    validate_parser.add_argument(
        "--agents-dir",
        metavar="PATH",
        default=None,
        help="flat directory of agent modules to resolve bare names against (D2)",
    )

    doctor_parser = subparsers.add_parser(
        "doctor", help="probe the deployment environment", description="Probe the environment."
    )
    doctor_parser.add_argument(
        "--offline",
        action="store_true",
        help="force offline env vars and assert zero egress (required in Milestone 0)",
    )

    subparsers.add_parser(
        "migrate",
        help="apply the vdreamer schema (runs, run_events, checkpointer)",
        description=(
            "Apply the vdreamer schema to Postgres: the runs table and its "
            "active-thread index, the run_events log, and the checkpointer's own "
            "migrations (AsyncPostgresSaver.setup). All steps are idempotent. "
            "The DSN comes from VDREAMER_PG_DSN — never the command line (A23)."
        ),
    )

    def _add_topology_agent_flags(parser: argparse.ArgumentParser) -> None:
        """The D2 discovery flags shared by the worker/dev entrypoints."""
        parser.add_argument(
            "--agent",
            metavar="NAME",
            required=True,
            help="agent name (resolved via --registry/--agents-dir) or a pkg.module:agent spec",
        )
        parser.add_argument(
            "--registry",
            metavar="PATH",
            default=None,
            help="path to a vdreamer.yaml registry mapping names to import specs (D2)",
        )
        parser.add_argument(
            "--agents-dir",
            metavar="PATH",
            default=None,
            help="flat directory of agent modules to resolve bare names against (D2)",
        )

    worker_parser = subparsers.add_parser(
        "worker",
        help="run the worker loop for one agent (PRD §2.1)",
        description=(
            "Serve one agent: claim queued runs from the Postgres queue (A7), drive "
            "them through the agent's graph, and write the protocol events to the "
            "run_events log. The DSN comes from VDREAMER_PG_DSN (A23); run "
            "`vdreamer migrate` once before the first worker."
        ),
    )
    _add_topology_agent_flags(worker_parser)
    worker_parser.add_argument(
        "--claimant",
        metavar="ID",
        default=None,
        help="lease claim stamp (default: worker-{hostname}-{pid})",
    )

    dev_parser = subparsers.add_parser(
        "dev",
        help="single-process dev mode: SQLite + in-memory queue (D11)",
        description=(
            "Run everything in one process with no infrastructure: the agent's "
            "checkpoints on a local SQLite file (D11), the queue and the event "
            "sinks in memory. With --input, enqueue and drive exactly one run and "
            "report the outcome (a smoke of the whole lifecycle); without it, run "
            "the worker loop until Ctrl-C. Hot reload is future work."
        ),
    )
    _add_topology_agent_flags(dev_parser)
    dev_parser.add_argument(
        "--db",
        metavar="PATH",
        default="./.vdreamer/dev.db",
        help="SQLite checkpoint file (default ./.vdreamer/dev.db)",
    )
    dev_parser.add_argument(
        "--input",
        metavar="JSON",
        default=None,
        help="enqueue and run one run with this JSON input, then exit",
    )
    dev_parser.add_argument(
        "--thread",
        metavar="ID",
        default=None,
        help="thread id for the --input run (default: none)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "validate":
        from vdreamer.validate import run_validate

        return run_validate(args.agent, registry=args.registry, agents_dir=args.agents_dir)

    if args.command == "doctor":
        if not args.offline:
            parser.error("doctor requires --offline; online probes land with Milestone 10")
        from vdreamer.doctor import run_offline

        return run_offline()

    if args.command == "migrate":
        from vdreamer.migrate import run_migrate

        return run_migrate()

    if args.command == "worker":
        from vdreamer.entrypoints import run_worker

        return run_worker(
            agent=args.agent,
            registry=args.registry,
            agents_dir=args.agents_dir,
            claimant=args.claimant,
        )

    if args.command == "dev":
        from vdreamer.entrypoints import run_dev

        return run_dev(
            agent=args.agent,
            registry=args.registry,
            agents_dir=args.agents_dir,
            db=args.db,
            input_json=args.input,
            thread_id=args.thread,
        )

    parser.error(f"unknown command {args.command!r}")
