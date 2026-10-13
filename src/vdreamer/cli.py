"""``vdreamer`` CLI (PRD §2.3).

Milestone 0 surface: ``--version``, ``validate``, ``doctor --offline`` — the
pieces the CI matrix depends on. Milestone 1 wires the D2 discovery flags into
``validate``: ``--registry PATH`` / ``--agents-dir PATH`` resolve a bare agent
name (``vdreamer.discovery.resolve_agent``), while a ``pkg.module:agent`` spec
stays explicit and wins. ``dev``/``serve``/``worker`` (with their own registry/
agents-dir flags) land with Milestone 3, ``run`` with Milestone 11, and
``migrate``/``export-schema``/``bundle`` with Milestone 10.
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

    parser.error(f"unknown command {args.command!r}")
