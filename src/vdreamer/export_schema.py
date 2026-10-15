"""``vdreamer export-schema`` — the protocol contract as files (PRD §2.3, §5.3, §6.1).

What client teams build against, written to ``--out`` (default
``./vdreamer-schema``) and attached to every release:

- ``openapi.json`` — the REST surface, OpenAPI 3.1 (the same document
  ``GET /api/v1/openapi.json`` serves);
- ``events.schema.json`` — the SSE event envelope, JSON Schema 2020-12 (the
  same document as ``GET /api/v1/protocol/schema.json``);
- ``agents/<name>.input.schema.json`` — each agent's ``input_schema`` (the
  body ``/trigger`` and ``/threads/{id}/runs`` validate against);
- ``vdreamer.d.ts`` — TypeScript declarations generated from all of the above
  (:mod:`vdreamer.tsgen`; no npm package, A45);
- ``manifest.json`` — protocol and vdreamer versions, the agents, and every
  file's sha256.

Output is deterministic: two exports of the same code are byte-identical, so a
release diff of these files *is* the contract diff. Agents are optional — with
none, the generic contract is still exported. Nothing here starts a server or
touches a store (in-memory stand-ins build the app), so it runs offline.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from vdreamer import __version__
from vdreamer.agents import BaseAgent
from vdreamer.events import PROTOCOL_VERSION, EventType
from vdreamer.tsgen import declarations, namespace, to_ts, type_name

DEFAULT_OUT: Final = "vdreamer-schema"

_PLACEHOLDER_TOKEN: Final = "export-schema-placeholder-token"


def _dump(document: Any) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def openapi_document(agents: Mapping[str, BaseAgent]) -> dict[str, Any]:
    """The gateway's OpenAPI 3.1 document, built over in-memory stand-ins."""
    from vdreamer.artifacts import InMemoryArtifactStore
    from vdreamer.eventlog import InMemoryEventLog
    from vdreamer.gateway import GatewayBackend, GatewaySettings, create_app
    from vdreamer.interrupts import InMemoryInterruptStore
    from vdreamer.queue import InMemoryRunQueue
    from vdreamer.threads import InMemoryThreadStore

    log = InMemoryEventLog()
    backend = GatewayBackend(
        agents=dict(agents),
        queue=InMemoryRunQueue(),
        threads=InMemoryThreadStore(),
        event_reader=log,
        event_writer=log,
        hub=log.hub,
        interrupts=InMemoryInterruptStore(),
        artifacts=InMemoryArtifactStore(),
    )
    app = create_app(backend, GatewaySettings(tokens=(_PLACEHOLDER_TOKEN,)))
    return dict(app.openapi())


def agent_input_schema(agent: BaseAgent) -> dict[str, Any] | None:
    """The agent's ``input_schema`` as JSON Schema (validation mode), or ``None``."""
    if agent.input_schema is None:
        return None
    schema = dict(agent.input_schema.model_json_schema(mode="validation"))
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema.setdefault("title", f"{agent.name} input")
    return schema


def _envelope_ts(payload_defs: Mapping[str, Any], payload_refs: Mapping[str, str]) -> str:
    """The envelope, the event-type union, and the known-event discriminated union."""
    event_types = [event.value for event in EventType]
    known = "\n".join(
        f'  | Envelope<"{event}", Events.{payload_refs[event]}>' for event in event_types
    )
    return (
        f'export type ProtocolVersion = "{PROTOCOL_VERSION}";\n\n'
        "/** Every event type protocol 1.0 defines. Clients ignore any other (§6.1). */\n"
        "export type EventType =\n"
        + "\n".join(f'  | "{event}"' for event in event_types)
        + ";\n\n"
        "/** One SSE event (§3.3). The SSE id is `${run_id}:${seq}`; dedup by `seq`. */\n"
        "export interface Envelope<T extends string = string, P = Record<string, unknown>> {\n"
        "  v: ProtocolVersion;\n"
        "  type: T;\n"
        "  run_id: string;\n"
        "  thread_id: string;\n"
        "  agent: string;\n"
        "  seq: number;\n"
        "  ts: string;\n"
        "  ns: string[];\n"
        "  payload: P;\n"
        "}\n\n"
        "/** The events this protocol version knows, discriminated on `type`. */\n"
        f"export type KnownEvent =\n{known};\n\n"
        "/** What a stream may carry: a known event, or one from a newer minor to ignore. */\n"
        "export type VdreamerEvent = KnownEvent | Envelope<string, Record<string, unknown>>;\n"
    )


def typescript(
    openapi: Mapping[str, Any],
    events: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
) -> str:
    """``vdreamer.d.ts``: ``Api`` (OpenAPI components), ``Events`` (payloads), ``Agents``."""
    components = dict((openapi.get("components") or {}).get("schemas") or {})
    payload_defs = dict(events.get("$defs") or {})
    payload_refs: dict[str, str] = {}
    for clause in events.get("allOf") or ():
        event = clause["if"]["properties"]["type"]["const"]
        ref = clause["then"]["properties"]["payload"]["$ref"]
        payload_refs[event] = type_name(ref.rsplit("/", 1)[-1])

    parts = [
        "// Generated by `vdreamer export-schema` — do not edit.\n"
        f"// vdreamer {__version__}, protocol {PROTOCOL_VERSION}. Types only: no runtime code.\n",
        _envelope_ts(payload_defs, payload_refs),
        "/** Event payloads (§3.3), named as in events.schema.json. */\n"
        + namespace("Events", declarations(payload_defs, indent="  ")),
        "/** REST request/response bodies, named as in openapi.json components. */\n"
        + namespace("Api", declarations(components, indent="  ")),
    ]
    agent_blocks: list[str] = []
    for name in sorted(inputs):
        schema = dict(inputs[name])
        defs = dict(schema.pop("$defs", None) or {})
        body = f"    export type Input = {to_ts(schema, '    ')};\n"
        if defs:
            body += "\n" + declarations(defs, indent="    ")
        agent_blocks.append(
            f"  /** `{name}`: the body of /agents/{name}/trigger and its runs. */\n"
            + namespace(name, body, indent="  ", declare=False)
        )
    parts.append(
        "/** Per-agent run inputs (each agent's input_schema). */\n"
        + namespace("Agents", "\n".join(agent_blocks))
    )
    return "\n".join(parts)


def build_export(agents: Mapping[str, BaseAgent]) -> dict[str, bytes]:
    """Every exported file, by relative path (``/``-separated)."""
    from vdreamer.protocol_schema import event_json_schema

    openapi = openapi_document(agents)
    events = event_json_schema()
    inputs = {
        name: schema
        for name, agent in sorted(agents.items())
        if (schema := agent_input_schema(agent)) is not None
    }
    files: dict[str, bytes] = {
        "openapi.json": _dump(openapi),
        "events.schema.json": _dump(events),
        "vdreamer.d.ts": typescript(openapi, events, inputs).encode(),
    }
    for name, schema in inputs.items():
        files[f"agents/{name}.input.schema.json"] = _dump(schema)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "vdreamer_version": __version__,
        "openapi_version": openapi.get("openapi"),
        "agents": {
            name: {"mode": agent.mode.value, "input_schema": name in inputs}
            for name, agent in sorted(agents.items())
        },
        "files": {path: hashlib.sha256(data).hexdigest() for path, data in sorted(files.items())},
    }
    files["manifest.json"] = _dump(manifest)
    return files


def write_export(files: Mapping[str, bytes], out: Path) -> None:
    """Write ``files`` under ``out``, replacing what an earlier export left there."""
    out.mkdir(parents=True, exist_ok=True)
    for stale in sorted(out.glob("agents/*.input.schema.json")):
        if stale.relative_to(out).as_posix() not in files:
            stale.unlink()
    for path, data in files.items():
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def run_export_schema(
    *,
    agents: list[str],
    registry: str | None = None,
    agents_dir: str | None = None,
    out: str = DEFAULT_OUT,
) -> int:
    """CLI entry point for ``vdreamer export-schema``."""
    from vdreamer.discovery import AgentLoadError
    from vdreamer.entrypoints import ServeAgentsError, resolve_serve_agents

    loaded: dict[str, BaseAgent] = {}
    if agents or registry or agents_dir:
        try:
            loaded = resolve_serve_agents(agents, registry=registry, agents_dir=agents_dir)
        except (AgentLoadError, ServeAgentsError) as exc:
            print(f"export-schema: {exc}", file=sys.stderr)
            return 1
    files = build_export(loaded)
    target = Path(out)
    write_export(files, target)
    for path in sorted(files):
        print(f"[ok] {(target / path).as_posix()}")
    names = ", ".join(sorted(loaded)) or "none (generic contract only)"
    print(f"export-schema: protocol {PROTOCOL_VERSION}, agents: {names}")
    return 0


__all__ = [
    "DEFAULT_OUT",
    "agent_input_schema",
    "build_export",
    "openapi_document",
    "run_export_schema",
    "typescript",
    "write_export",
]
