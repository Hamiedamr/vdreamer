"""JSON Schema → TypeScript declarations, for ``vdreamer export-schema`` (PRD §6.1).

vdreamer ships no npm packages and no client SDK (§7.2, A45): client teams get
``.d.ts`` *types* generated from the same schemas the gateway serves, so a
TypeScript client compiles against the protocol without a code generator of
its own. This is a deliberately small translator — the subset pydantic and
FastAPI emit (draft 2020-12 / OpenAPI 3.1):

- ``type`` (incl. lists of types), ``const``, ``enum``, ``anyOf``/``oneOf``
  (unions), ``allOf`` (intersections), ``$ref`` to ``#/$defs/X``,
  ``#/definitions/X`` and ``#/components/schemas/X``;
- objects with ``properties``/``required`` and ``additionalProperties``
  (a schema → an index signature, ``true``/absent → open);
- arrays with ``items`` and ``prefixItems`` (tuples).

Anything it does not understand becomes ``unknown`` — never a wrong type.
Output is deterministic (sorted definitions), so a diff of two releases'
``.d.ts`` is a diff of the contract.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Final

_IDENT: Final = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_REF_PREFIXES: Final[tuple[str, ...]] = ("#/$defs/", "#/definitions/", "#/components/schemas/")

_PRIMITIVES: Final[dict[str, str]] = {
    "string": "string",
    "number": "number",
    "integer": "number",
    "boolean": "boolean",
    "null": "null",
}


def type_name(raw: str) -> str:
    """A TypeScript identifier for a schema/definition name (``Foo-Bar`` → ``Foo_Bar``)."""
    name = re.sub(r"[^A-Za-z0-9_$]", "_", raw)
    if not name or name[0].isdigit():
        name = f"_{name}"
    return name


def _key(name: str) -> str:
    return name if _IDENT.fullmatch(name) else json.dumps(name)


def _literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return json.dumps(value)
    return "unknown"


def _union(parts: list[str]) -> str:
    unique: list[str] = []
    for part in parts:
        if part not in unique:
            unique.append(part)
    if "unknown" in unique:
        return "unknown"
    return " | ".join(unique) if unique else "never"


def _wrap(ts: str) -> str:
    """Parenthesize a union/intersection used as an array element."""
    return f"({ts})" if (" | " in ts or " & " in ts) else ts


def _comment(schema: Mapping[str, Any], indent: str) -> str:
    text = schema.get("description") or schema.get("title")
    if not isinstance(text, str) or not text.strip():
        return ""
    lines = text.strip().replace("*/", "* /").splitlines()
    if len(lines) == 1:
        return f"{indent}/** {lines[0]} */\n"
    body = "".join(f"{indent} * {line}".rstrip() + "\n" for line in lines)
    return f"{indent}/**\n{body}{indent} */\n"


def to_ts(schema: Any, indent: str = "") -> str:
    """The TypeScript type expression for one schema."""
    if schema is True or schema == {}:
        return "unknown"
    if not isinstance(schema, Mapping):
        return "unknown"
    ref = schema.get("$ref")
    if isinstance(ref, str):
        for prefix in _REF_PREFIXES:
            if ref.startswith(prefix):
                return type_name(ref[len(prefix) :])
        return "unknown"
    if "const" in schema:
        return _literal(schema["const"])
    if isinstance(schema.get("enum"), list):
        return _union([_literal(value) for value in schema["enum"]])
    for combinator in ("anyOf", "oneOf"):
        if isinstance(schema.get(combinator), list):
            return _union([to_ts(part, indent) for part in schema[combinator]])
    if isinstance(schema.get("allOf"), list):
        parts = [to_ts(part, indent) for part in schema["allOf"]]
        parts = [part for part in parts if part != "unknown"]
        if not parts:
            return "unknown"
        return parts[0] if len(parts) == 1 else " & ".join(_wrap(part) for part in parts)

    kind = schema.get("type")
    if isinstance(kind, list):
        return _union([to_ts({**schema, "type": entry}, indent) for entry in kind])
    if kind in _PRIMITIVES:
        return _PRIMITIVES[kind]
    if kind == "array" or "items" in schema or "prefixItems" in schema:
        tuple_items = schema.get("prefixItems")
        if isinstance(tuple_items, list):
            return "[" + ", ".join(to_ts(item, indent) for item in tuple_items) + "]"
        items = schema.get("items")
        return f"{_wrap(to_ts(items, indent))}[]" if items is not None else "unknown[]"
    if kind == "object" or "properties" in schema:
        return _object(schema, indent)
    return "unknown"


def _object(schema: Mapping[str, Any], indent: str) -> str:
    properties = schema.get("properties")
    extra = schema.get("additionalProperties", True)
    inner = indent + "  "
    lines: list[str] = []
    if isinstance(properties, Mapping):
        required = set(schema.get("required") or ())
        for name, prop in properties.items():
            optional = "" if name in required else "?"
            comment = _comment(prop, inner) if isinstance(prop, Mapping) else ""
            lines.append(f"{comment}{inner}{_key(name)}{optional}: {to_ts(prop, inner)};")
    if extra is not False:
        value = "unknown" if extra is True else to_ts(extra, inner)
        if lines and value != "unknown":
            value = f"{value} | unknown"  # the named properties must stay assignable
        if lines or value != "unknown":
            lines.append(f"{inner}[key: string]: {value};")
        else:
            return "Record<string, unknown>"
    if not lines:
        return "Record<string, never>"
    return "{\n" + "\n".join(lines) + f"\n{indent}}}"


def declarations(definitions: Mapping[str, Any], *, indent: str = "") -> str:
    """``export type`` declarations for every named schema, sorted by name."""
    out: list[str] = []
    for raw in sorted(definitions):
        schema = definitions[raw]
        comment = _comment(schema, indent) if isinstance(schema, Mapping) else ""
        out.append(f"{comment}{indent}export type {type_name(raw)} = {to_ts(schema, indent)};\n")
    return "\n".join(out)


def namespace(name: str, body: str, *, indent: str = "", declare: bool = True) -> str:
    """Wrap declarations (already indented one level deeper) in a namespace."""
    keyword = "export declare namespace" if declare else "export namespace"
    return f"{indent}{keyword} {type_name(name)} {{\n{body}{indent}}}\n"


__all__ = ["declarations", "namespace", "to_ts", "type_name"]
