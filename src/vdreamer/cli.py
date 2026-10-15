"""``vdreamer`` CLI (PRD §2.3).

Milestone 0 surface: ``--version``, ``validate``, ``doctor --offline`` — the
pieces the CI matrix depends on. Milestone 1 wires the D2 discovery flags into
``validate``: ``--registry PATH`` / ``--agents-dir PATH`` resolve a bare agent
name (``vdreamer.discovery.resolve_agent``), while a ``pkg.module:agent`` spec
stays explicit and wins. Milestone 3 adds the topology entrypoints: ``worker``
(one process, one agent, Postgres queue + checkpointer — A7) and ``dev`` (the
D11 single-process mode on SQLite + in-memory queue), plus ``migrate`` (the
idempotent schema application, DSN from ``VDREAMER_PG_DSN``, A23). Milestone
4 adds ``serve`` (the REST/SSE gateway, D20) and turns ``dev`` without
``--input`` into gateway + worker in one process. Milestone 7 adds
``--models`` (models.yaml) next to ``--resources`` on ``validate``,
``doctor``, ``worker`` and ``dev``. ``run`` lands with Milestone 11, and
``export-schema``/``bundle`` with Milestone 10.

Milestone 10 also applies the zero-egress environment (:mod:`vdreamer.offline`)
before every command but ``bundle`` — the one that downloads, on a connected
host — and adds ``dev --reload`` (:mod:`vdreamer.reload`).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from vdreamer import __version__


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    """``--models`` / ``--resources`` (§3.6, §3.7)."""
    parser.add_argument(
        "--models",
        metavar="PATH",
        default=None,
        help="models.yaml: model aliases (default $VDREAMER_MODELS)",
    )
    parser.add_argument(
        "--resources",
        metavar="PATH",
        default=None,
        help="resources.yaml: agent resources and MCP servers (default $VDREAMER_RESOURCES)",
    )


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
    _add_config_flags(validate_parser)
    validate_parser.add_argument(
        "--no-probe",
        action="store_true",
        help="skip the network: endpoints and resources are not contacted (model files are "
        "still checked)",
    )

    doctor_parser = subparsers.add_parser(
        "doctor", help="probe the deployment environment", description="Probe the environment."
    )
    doctor_parser.add_argument(
        "--offline",
        action="store_true",
        help="force the offline env vars and assert zero egress (the only mode; kept explicit "
        "for CI)",
    )
    _add_config_flags(doctor_parser)

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

    def _add_resources_flag(parser: argparse.ArgumentParser) -> None:
        _add_config_flags(parser)

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
    worker_parser.add_argument(
        "--metrics-port",
        metavar="PORT",
        type=int,
        default=None,
        help="serve this worker's Prometheus metrics on PORT (default: off)",
    )
    worker_parser.add_argument(
        "--metrics-host",
        metavar="ADDR",
        default="127.0.0.1",
        help="interface for --metrics-port (default 127.0.0.1; wildcards refused)",
    )
    _add_resources_flag(worker_parser)

    def _add_bind_flags(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--host",
            metavar="ADDR",
            default="127.0.0.1",
            help="internal interface to bind (default 127.0.0.1; 0.0.0.0/:: refused, A14)",
        )
        parser.add_argument(
            "--port", metavar="PORT", type=int, default=8000, help="port (default 8000)"
        )

    serve_parser = subparsers.add_parser(
        "serve",
        help="run the REST/SSE gateway (PRD §2.5)",
        description=(
            "Serve the /api/v1 REST/SSE contract for the agents named by --agent "
            "(repeatable), or every agent of --registry / --agents-dir (D2). Stateless: "
            "everything lives in Postgres (VDREAMER_PG_DSN, A23); run `vdreamer migrate` "
            "first and one `vdreamer worker` per agent. Requires VDREAMER_INTERNAL_TOKEN "
            "(and optionally VDREAMER_INTERNAL_TOKEN_PREVIOUS for rotation, A14)."
        ),
    )
    serve_parser.add_argument(
        "--agent",
        metavar="NAME",
        action="append",
        default=[],
        help="an agent to serve: a pkg.module:agent spec or a name (repeatable)",
    )
    serve_parser.add_argument(
        "--registry",
        metavar="PATH",
        default=None,
        help="path to a vdreamer.yaml registry mapping names to import specs (D2)",
    )
    serve_parser.add_argument(
        "--agents-dir",
        metavar="PATH",
        default=None,
        help="flat directory of agent modules (D2)",
    )
    _add_bind_flags(serve_parser)

    dev_parser = subparsers.add_parser(
        "dev",
        help="single-process dev mode: gateway + worker on one SQLite file (D11)",
        description=(
            "Run everything in one process with no infrastructure: the agent's "
            "runs, events, threads, interrupts and checkpoints on one local SQLite "
            "file (D11), surviving restarts. With --input, enqueue and drive exactly one run and "
            "report the outcome (a smoke of the whole lifecycle); without it, serve the "
            "REST/SSE gateway and run the worker loop until Ctrl-C. The token comes from "
            "VDREAMER_INTERNAL_TOKEN, or a one-off token is printed. --reload restarts the "
            "process when the agent's sources or config change (state survives in --db)."
        ),
    )
    _add_topology_agent_flags(dev_parser)
    dev_parser.add_argument(
        "--db",
        metavar="PATH",
        default="./.vdreamer/dev.db",
        help="SQLite state + checkpoint file (default ./.vdreamer/dev.db)",
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
    dev_parser.add_argument(
        "--reload",
        action="store_true",
        help="restart when *.py / *.yaml / *.json sources change (not with --input)",
    )
    _add_resources_flag(dev_parser)
    _add_bind_flags(dev_parser)

    export_parser = subparsers.add_parser(
        "export-schema",
        help="write the protocol contract: OpenAPI 3.1, event JSON Schema, inputs, TS types",
        description=(
            "Write the client contract to --out: openapi.json (OpenAPI 3.1), "
            "events.schema.json (the SSE envelope), agents/<name>.input.schema.json, "
            "vdreamer.d.ts (TypeScript declarations, no npm package) and manifest.json "
            "(versions + sha256). Agents come from --agent / --registry / --agents-dir; "
            "with none, only the generic contract is written. Deterministic and offline."
        ),
    )
    export_parser.add_argument(
        "--agent",
        metavar="NAME",
        action="append",
        default=[],
        help="an agent whose input schema to export: a pkg.module:agent spec or a name "
        "(repeatable)",
    )
    export_parser.add_argument(
        "--registry", metavar="PATH", default=None, help="a vdreamer.yaml registry (D2)"
    )
    export_parser.add_argument(
        "--agents-dir", metavar="PATH", default=None, help="a flat directory of agent modules (D2)"
    )
    export_parser.add_argument(
        "--out", metavar="DIR", default="vdreamer-schema", help="output directory "
        "(default ./vdreamer-schema)"
    )

    bundle_parser = subparsers.add_parser(
        "bundle",
        help="build the offline install set on a connected host (wheelhouse + hashed lock)",
        description=(
            "Run on a connected host with plain pip (no uv): `python -m pip download` "
            "vdreamer (with --extra ...) and your agents' requirements (-r) into "
            "OUT/wheelhouse, constrained by the release constraints.txt (-c), then write "
            "a hash-pinned OUT/requirements.lock, the tiktoken encodings (OUT/tiktoken) "
            "and, with --models, the fastembed aliases (OUT/fastembed). On the air-gapped "
            "host: pip install --no-index --find-links wheelhouse --require-hashes "
            "-r requirements.lock."
        ),
    )
    bundle_parser.add_argument(
        "--out", metavar="DIR", default="vdreamer-bundle", help="output directory "
        "(default ./vdreamer-bundle)"
    )
    bundle_parser.add_argument(
        "--vdreamer",
        metavar="SPEC",
        default=None,
        help="the vdreamer to bundle: a requirement or a local wheel path "
        "(default vdreamer==<this version>)",
    )
    bundle_parser.add_argument(
        "--extra",
        metavar="NAME",
        action="append",
        default=[],
        help="a vdreamer extra to include: postgres, minio, fastembed, qdrant, neo4j, mcp, "
        "sandbox, sql (repeatable)",
    )
    bundle_parser.add_argument(
        "-r",
        "--requirement",
        metavar="FILE",
        action="append",
        default=[],
        help="your agents' requirements file (repeatable)",
    )
    bundle_parser.add_argument(
        "-c",
        "--constraints",
        metavar="FILE",
        default=None,
        help="the release's constraints.txt (the tested dependency set)",
    )
    bundle_parser.add_argument(
        "--platform",
        metavar="TAG",
        action="append",
        default=[],
        help="target platform tag, e.g. manylinux2014_x86_64 (repeatable; binary wheels only)",
    )
    bundle_parser.add_argument(
        "--python-version", metavar="X.Y", default=None, help="target Python version"
    )
    bundle_parser.add_argument(
        "--implementation", metavar="IMPL", default=None, help="target implementation (cp)"
    )
    bundle_parser.add_argument(
        "--abi", metavar="ABI", action="append", default=[], help="target ABI (repeatable)"
    )
    bundle_parser.add_argument(
        "--index-url", metavar="URL", default=None, help="the package index to download from"
    )
    bundle_parser.add_argument(
        "--models",
        metavar="PATH",
        default=None,
        help="models.yaml: pre-fetch its fastembed aliases into OUT/fastembed",
    )
    bundle_parser.add_argument(
        "--no-tiktoken",
        action="store_true",
        help="do not fetch the tiktoken encodings into OUT/tiktoken",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "bundle":
        # The one command that downloads (on a connected host): no offline env.
        from pathlib import Path

        from vdreamer.bundle import BundleOptions, run_bundle

        return run_bundle(
            BundleOptions(
                out=Path(args.out),
                vdreamer=args.vdreamer,
                extras=tuple(args.extra),
                requirements=tuple(args.requirement),
                constraints=args.constraints,
                platforms=tuple(args.platform),
                python_version=args.python_version,
                implementation=args.implementation,
                abis=tuple(args.abi),
                index_url=args.index_url,
                models=args.models,
                tiktoken=not args.no_tiktoken,
            )
        )

    # §5.2: tracing off, HF offline, no telemetry — before any command imports the stack.
    from vdreamer.offline import apply_offline_env

    apply_offline_env()

    if args.command == "validate":
        from vdreamer.validate import run_validate

        return run_validate(
            args.agent,
            registry=args.registry,
            agents_dir=args.agents_dir,
            resources=args.resources,
            models=args.models,
            probe=not args.no_probe,
        )

    if args.command == "doctor":
        # An air-gapped product has one mode: the probes always run under the
        # egress guard; --offline stays as the explicit (CI) spelling.
        from vdreamer.doctor import run_offline

        return run_offline(models=args.models, resources=args.resources)

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
            metrics_host=args.metrics_host,
            metrics_port=args.metrics_port,
            resources=args.resources,
            models=args.models,
        )

    if args.command == "serve":
        from vdreamer.entrypoints import run_serve

        return run_serve(
            agents=args.agent,
            registry=args.registry,
            agents_dir=args.agents_dir,
            host=args.host,
            port=args.port,
        )

    if args.command == "dev" and args.reload:
        if args.input is not None:
            parser.error("dev --reload serves the gateway; it cannot be combined with --input")
        import sys

        from vdreamer.reload import run_dev_reload

        return run_dev_reload(
            list(argv) if argv is not None else sys.argv[1:],
            db=args.db,
            agents_dir=args.agents_dir,
            registry=args.registry,
            models=args.models,
            resources=args.resources,
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
            host=args.host,
            port=args.port,
            resources=args.resources,
            models=args.models,
        )

    if args.command == "export-schema":
        from vdreamer.export_schema import run_export_schema

        return run_export_schema(
            agents=args.agent, registry=args.registry, agents_dir=args.agents_dir, out=args.out
        )

    parser.error(f"unknown command {args.command!r}")
