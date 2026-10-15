"""Per-kind artifact encoding and previews.

- ``table`` — ``pyarrow.Table``, ``pandas.DataFrame``, or an iterable of row
  mappings → Parquet; preview: schema + 10 rows + ``row_count``.
- ``image`` — bytes or a path holding PNG, JPEG, or SVG → as-is;
  preview: ``{width, height}``.
- ``html`` — a ``str`` (may embed ``artifact://{id}`` links) → as-is;
  preview: ``{title, summary?}``.
- ``json`` — a Pydantic model or a dict, ≤ 1 MB → JSON; preview: the full
  body when ≤ 64 KB.
- ``file`` — bytes or a path, plus a mime type → as-is; preview:
  ``{filename, mime}``.

No auto-detection: data the kind does not accept is a ``TypeError`` (and an
image that is not PNG/JPEG/SVG is one too). :func:`encode` turns accepted
data into an :class:`Encoded` — bytes, or a spooled file for tables — plus the
mime type, object-key extension, and preview; ``vdreamer.artifacts`` stores it.
"""

from __future__ import annotations

import html as html_lib
import json
import mimetypes
import os
import re
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pydantic

from vdreamer import tables

#: ``json`` artifacts hold at most 1 MB; the preview carries ≤ 64 KB bodies.
JSON_MAX_BYTES: Final = 1024 * 1024
JSON_PREVIEW_MAX_BYTES: Final = 64 * 1024
HTML_TITLE_MAX_CHARS: Final = 200
HTML_SUMMARY_MAX_CHARS: Final = 300

#: Mime type → object-key extension.
EXTENSIONS: Final[dict[str, str]] = {
    "application/json": "json",
    "text/plain": "txt",
    "text/html": "html",
    "text/csv": "csv",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/svg+xml": "svg",
    "application/pdf": "pdf",
    "application/zip": "zip",
    tables.PARQUET_MIME: "parquet",
}
_SAFE_EXT: Final = re.compile(r"^[A-Za-z0-9]{1,10}$")


def extension_for(mime_type: str, filename: str | None = None) -> str:
    """The object key's extension: the filename's own when safe, else from the mime."""
    if filename:
        suffix = Path(filename).suffix.lstrip(".")
        if _SAFE_EXT.fullmatch(suffix):
            return suffix.lower()
    base = mime_type.split(";", 1)[0].strip().lower()
    if base in EXTENSIONS:
        return EXTENSIONS[base]
    guessed = mimetypes.guess_extension(base) or ""
    guessed = guessed.lstrip(".")
    return guessed if _SAFE_EXT.fullmatch(guessed) else "bin"


@dataclass
class Encoded:
    """One artifact ready to store: ``data`` bytes, or a spooled ``path`` (tables)."""

    kind: str
    mime_type: str
    ext: str
    size: int
    preview: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    filename: str | None = None
    data: bytes | None = None
    path: Path | None = None

    def cleanup(self) -> None:
        """Remove the spooled file, if any."""
        if self.path is not None:
            self.path.unlink(missing_ok=True)


def _as_bytes(data: Any, kind: str) -> tuple[bytes, str | None]:
    """``(bytes, source filename)`` from bytes or a path; anything else is a ``TypeError``."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data), None
    if isinstance(data, (str, os.PathLike)):
        path = Path(data)
        return path.read_bytes(), path.name
    raise TypeError(f"{kind} data must be bytes or a path, got {type(data).__name__}")


# -- image ------------------------------------------------------------------

_SVG_ROOT: Final = re.compile(rb"<svg\b[^>]*>", re.IGNORECASE | re.DOTALL)
_SVG_START: Final = re.compile(
    rb"^(?:\xef\xbb\xbf)?\s*(?:<\?xml[^>]*\?>\s*)?(?:<!--.*?-->\s*|<!DOCTYPE[^>]*>\s*)*<svg\b",
    re.IGNORECASE | re.DOTALL,
)


def _svg_length(tag: bytes, name: bytes) -> float | None:
    match = re.search(rb"\b" + name + rb"\s*=\s*[\"']\s*([0-9.]+)\s*(px)?\s*[\"']", tag)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _svg_size(data: bytes) -> tuple[int | None, int | None]:
    match = _SVG_ROOT.search(data)
    if match is None:
        return None, None
    tag = match.group(0)
    width, height = _svg_length(tag, b"width"), _svg_length(tag, b"height")
    if width is None or height is None:
        box = re.search(rb"\bviewBox\s*=\s*[\"']([^\"']+)[\"']", tag)
        if box is not None:
            parts = re.split(rb"[\s,]+", box.group(1).strip())
            if len(parts) == 4:
                try:
                    width = width if width is not None else float(parts[2])
                    height = height if height is not None else float(parts[3])
                except ValueError:
                    pass
    return (
        round(width) if width is not None else None,
        round(height) if height is not None else None,
    )


def _jpeg_size(data: bytes) -> tuple[int | None, int | None]:
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        (length,) = struct.unpack(">H", data[index + 2 : index + 4])
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return width, height
        index += 2 + length
    return None, None


def image_info(data: bytes) -> tuple[str, int | None, int | None]:
    """``(mime, width, height)`` for PNG/JPEG/SVG bytes; anything else is a ``TypeError``."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return "image/png", width, height
    if data.startswith(b"\xff\xd8\xff"):
        width, height = _jpeg_size(data)
        return "image/jpeg", width, height
    if _SVG_START.match(data[:4096]):
        width, height = _svg_size(data)
        return "image/svg+xml", width, height
    raise TypeError("image data must be PNG, JPEG, or SVG (no auto-detection)")


# -- html -------------------------------------------------------------------

_TITLE: Final = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_DESCRIPTION: Final = re.compile(
    r"<meta\s+[^>]*name\s*=\s*[\"']description[\"'][^>]*content\s*=\s*[\"']([^\"']*)[\"']",
    re.IGNORECASE,
)


def html_preview(text: str, fallback_title: str) -> dict[str, Any]:
    title_match = _TITLE.search(text)
    title = ""
    if title_match:
        title = html_lib.unescape(re.sub(r"\s+", " ", title_match.group(1))).strip()
    preview: dict[str, Any] = {"title": (title or fallback_title)[:HTML_TITLE_MAX_CHARS]}
    summary_match = _DESCRIPTION.search(text)
    if summary_match:
        summary = html_lib.unescape(summary_match.group(1)).strip()
        if summary:
            preview["summary"] = summary[:HTML_SUMMARY_MAX_CHARS]
    return preview


# -- json -------------------------------------------------------------------


def json_preview(body: Any, size: int) -> dict[str, Any]:
    """The full body when it is ≤ 64 KB, else only that it was cut."""
    return {"body": body} if size <= JSON_PREVIEW_MAX_BYTES else {"truncated": True}


def _json_bytes(data: Any) -> tuple[bytes, Any]:
    if isinstance(data, pydantic.BaseModel):
        body: Any = data.model_dump(mode="json")
    elif isinstance(data, Mapping):
        body = dict(data)
    else:
        raise TypeError(f"json data must be a Pydantic model or a dict, got {type(data).__name__}")
    try:
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"json data is not JSON-serializable: {exc}") from exc
    if len(encoded) > JSON_MAX_BYTES:
        raise ValueError(
            f"json artifacts hold at most {JSON_MAX_BYTES} bytes ({len(encoded)} given); "
            "store larger documents as a file artifact"
        )
    return encoded, json.loads(encoded)


# -- the entry point --------------------------------------------------------


def encode(
    kind: str,
    data: Any,
    *,
    name: str,
    metadata: Mapping[str, Any] | None = None,
    mime_type: str | None = None,
    filename: str | None = None,
) -> Encoded:
    """Encode ``data`` as a ``kind`` artifact (``TypeError`` when the kind does not take it)."""
    meta = dict(metadata or {})
    if data is None:
        raise TypeError("artifact data must not be None")
    if kind == "table":
        fd, spool = tempfile.mkstemp(prefix="vdreamer-table-", suffix=".parquet")
        os.close(fd)
        path = Path(spool)
        try:
            written = tables.write_parquet(data, path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        meta["truncated"] = written.truncated
        return Encoded(
            kind=kind,
            mime_type=tables.PARQUET_MIME,
            ext="parquet",
            size=path.stat().st_size,
            preview=written.preview,
            metadata=meta,
            filename=filename or f"{name}.parquet",
            path=path,
        )
    if kind == "image":
        raw, source = _as_bytes(data, kind)
        mime, width, height = image_info(raw)
        ext = EXTENSIONS[mime]
        return Encoded(
            kind=kind,
            mime_type=mime,
            ext=ext,
            size=len(raw),
            preview={"width": width, "height": height},
            metadata=meta,
            filename=filename or source or f"{name}.{ext}",
            data=raw,
        )
    if kind == "html":
        if not isinstance(data, str):
            raise TypeError(f"html data must be a str, got {type(data).__name__}")
        raw = data.encode("utf-8")
        return Encoded(
            kind=kind,
            mime_type="text/html; charset=utf-8",
            ext="html",
            size=len(raw),
            preview=html_preview(data, name),
            metadata=meta,
            filename=filename or f"{name}.html",
            data=raw,
        )
    if kind == "json":
        raw, body = _json_bytes(data)
        return Encoded(
            kind=kind,
            mime_type="application/json",
            ext="json",
            size=len(raw),
            preview=json_preview(body, len(raw)),
            metadata=meta,
            filename=filename or f"{name}.json",
            data=raw,
        )
    if kind == "file":
        raw, source = _as_bytes(data, kind)
        shown = filename or source or name
        mime = mime_type or mimetypes.guess_type(shown)[0] or "application/octet-stream"
        return Encoded(
            kind=kind,
            mime_type=mime,
            ext=extension_for(mime, shown),
            size=len(raw),
            preview={"filename": shown, "mime": mime},
            metadata=meta,
            filename=shown,
            data=raw,
        )
    raise TypeError(f"unknown artifact kind {kind!r} (table/image/html/json/file)")
