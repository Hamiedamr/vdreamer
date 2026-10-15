"""The artifact proxy.

The gateway proxies artifact bytes so URLs stay same-origin behind the
Web Backend. Access: the owner of the artifact's thread, an approver
granted one of that run's interrupts (its ``approver_role`` holder), or a
holder of ``artifact:read:all``. An artifact the caller may not read is
a ``404``, like any other owner-scoped resource. A record whose object is
gone (30-day lifecycle) answers ``410 artifact_expired``.

- ``/preview`` — the stored preview; for tables a page of rows,
  ``?offset&limit&sort`` (``limit ≤ 1000``; ``sort=col`` or ``-col``, comma
  separated) read from the Parquet object.
- ``/raw`` — the bytes inline under their own media type. HTML has its
  ``artifact://{id}`` links rewritten to ``/raw`` URLs and is served under a
  CSP allowing no external origin and ``connect-src 'none'``; chart JS is
  vendored at ``/static/vendor/``.
- ``/download`` — the bytes as an attachment; tables convert while streaming
  with ``?format=csv|xlsx|parquet`` (XLSX beyond 1,048,575 rows → ``422``).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import tempfile
import urllib.parse
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from vdreamer import tables
from vdreamer.artifacts import API_PREFIX as ARTIFACT_API_PREFIX
from vdreamer.artifacts import ArtifactRecord, artifact_urls
from vdreamer.gateway.auth import API_PREFIX, Identity, current_identity
from vdreamer.gateway.backend import GatewayBackend
from vdreamer.gateway.common import get_backend, get_settings, jsonable
from vdreamer.gateway.errors import ApiError, ErrorCode
from vdreamer.gateway.schemas import ArtifactOut, ArtifactPreview, ErrorResponse
from vdreamer.gateway.settings import GatewaySettings
from vdreamer.interrupts import Viewer

router = APIRouter(prefix=API_PREFIX, tags=["artifacts"])

Backend = Annotated[GatewayBackend, Depends(get_backend)]
Settings = Annotated[GatewaySettings, Depends(get_settings)]
Caller = Annotated[Identity, Depends(current_identity)]

_E = {"model": ErrorResponse}

#: Served with every ``html`` artifact: no external origin, ``connect-src
#: 'none'``. Inline script/style stay allowed — report HTML embeds its charts —
#: and clients frame it ``sandbox="allow-scripts"`` without same-origin.
HTML_ARTIFACT_CSP: Final[str] = (
    "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; font-src 'self' data:; connect-src 'none'; "
    "frame-ancestors 'self'; base-uri 'none'; form-action 'none'"
)

#: ``/preview`` page bound (``limit ≤ 1000``).
MAX_PREVIEW_LIMIT: Final[int] = 1000

#: ``artifact://{id}`` references inside HTML artifacts.
_ARTIFACT_LINK: Final = re.compile(r"artifact://(art_[A-Za-z0-9]+)")

_FORMAT_MIME: Final[dict[str, str]] = {
    "csv": "text/csv; charset=utf-8",
    "xlsx": tables.XLSX_MIME,
    "parquet": tables.PARQUET_MIME,
}


async def _readable_or_404(
    backend: GatewayBackend, settings: GatewaySettings, artifact_id: str, caller: Identity
) -> ArtifactRecord:
    record = await backend.artifacts.get(artifact_id)
    if record is not None:
        if settings.artifact_read_all_role in caller.roles:
            return record
        thread = await backend.threads.get(record.thread_id)
        if thread is not None and thread.owner_id == caller.user_id:
            return record
        granted = await backend.interrupts.list(
            status=None,
            run_id=record.run_id,
            visible_to=Viewer(user_id=caller.user_id, roles=tuple(caller.roles)),
            limit=1,
        )
        if any(item.approver_role in caller.roles for item in granted):
            return record
    raise ApiError(404, ErrorCode.ARTIFACT_NOT_FOUND, f"no artifact {artifact_id!r}")


def _expired(record: ArtifactRecord) -> ApiError:
    return ApiError(
        410,
        ErrorCode.ARTIFACT_EXPIRED,
        f"artifact {record.artifact_id!r} is no longer stored (retention)",
    )


async def _bytes_or_410(backend: GatewayBackend, record: ArtifactRecord) -> bytes:
    data = await backend.artifacts.read_bytes(record)
    if data is None:
        raise _expired(record)
    return data


@contextlib.asynccontextmanager
async def _local_object(backend: GatewayBackend, record: ArtifactRecord) -> AsyncIterator[Path]:
    """A local path to the bytes — the dev file itself, or a temp download (MinIO)."""
    local = backend.artifacts.local_path(record)
    if local is not None:
        yield local
        return
    fd, name = tempfile.mkstemp(prefix="vdreamer-artifact-")
    os.close(fd)
    path = Path(name)
    try:
        if not await backend.artifacts.fetch(record, path):
            raise _expired(record)
        yield path
    finally:
        path.unlink(missing_ok=True)


async def _owned_object(backend: GatewayBackend, record: ArtifactRecord) -> tuple[Path, bool]:
    """``(path, is_temp)``: like :func:`_local_object`, but the caller deletes temp files."""
    local = backend.artifacts.local_path(record)
    if local is not None:
        return local, False
    fd, name = tempfile.mkstemp(prefix="vdreamer-artifact-")
    os.close(fd)
    path = Path(name)
    if not await backend.artifacts.fetch(record, path):
        path.unlink(missing_ok=True)
        raise _expired(record)
    return path, True


def _filename(record: ArtifactRecord) -> str:
    return record.filename or record.name or record.artifact_id


def _content_disposition(disposition: str, filename: str) -> str:
    """RFC 6266: an ASCII-safe ``filename`` plus the exact UTF-8 ``filename*``."""
    fallback = re.sub(r"[^A-Za-z0-9._-]", "_", filename) or "artifact"
    quoted = urllib.parse.quote(filename, safe="")
    return f"{disposition}; filename=\"{fallback}\"; filename*=UTF-8''{quoted}"


def rewrite_artifact_links(text: str) -> str:
    """``artifact://{id}`` → ``/api/v1/artifacts/{id}/raw``."""
    return _ARTIFACT_LINK.sub(
        lambda match: f"{ARTIFACT_API_PREFIX}/artifacts/{match.group(1)}/raw", text
    )


def _with_extension(filename: str, ext: str) -> str:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return f"{stem}.{ext}"


def _file_response(
    path: Path, is_temp: bool, media_type: str, headers: dict[str, str]
) -> FileResponse:
    cleanup = BackgroundTask(path.unlink, missing_ok=True) if is_temp else None
    return FileResponse(path, media_type=media_type, headers=headers, background=cleanup)


@router.get(
    "/artifacts/{artifact_id}",
    response_model=ArtifactOut,
    responses={400: _E, 401: _E, 404: _E},
)
async def get_artifact(
    artifact_id: str, backend: Backend, settings: Settings, caller: Caller
) -> ArtifactOut:
    """The artifact's metadata — the ``artifact.created`` field set."""
    record = await _readable_or_404(backend, settings, artifact_id, caller)
    view_url, download_url = artifact_urls(record.artifact_id)
    return ArtifactOut(
        artifact_id=record.artifact_id,
        question_id=record.run_id,
        session_id=record.thread_id,
        kind=record.kind,
        renderer=record.renderer,
        name=record.name,
        mime_type=record.mime_type,
        size_bytes=record.size_bytes,
        preview=jsonable(dict(record.preview)),
        view_url=view_url,
        download_url=download_url,
        metadata=jsonable(dict(record.metadata)),
    )


@router.get(
    "/artifacts/{artifact_id}/preview",
    response_model=ArtifactPreview,
    responses={400: _E, 401: _E, 404: _E, 410: _E, 422: _E},
)
async def preview_artifact(
    artifact_id: str,
    backend: Backend,
    settings: Settings,
    caller: Caller,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_PREVIEW_LIMIT)] = 100,
    sort: Annotated[str | None, Query(max_length=1000)] = None,
) -> ArtifactPreview:
    """The stored preview; tables also return one page of rows."""
    record = await _readable_or_404(backend, settings, artifact_id, caller)
    base = ArtifactPreview(
        artifact_id=record.artifact_id,
        kind=record.kind,
        preview=jsonable(dict(record.preview)),
        offset=offset,
        limit=limit,
        sort=sort,
    )
    if record.kind != "table":
        if sort is not None:
            raise ApiError(422, ErrorCode.INVALID_REQUEST, "sort applies to table artifacts only")
        return base
    async with _local_object(backend, record) as path:
        try:
            page = await asyncio.to_thread(tables.page, path, offset=offset, limit=limit, sort=sort)
        except tables.SortError as exc:
            raise ApiError(422, ErrorCode.INVALID_REQUEST, str(exc)) from None
        except (tables.TablesUnavailable, tables.CorruptTable) as exc:
            raise ApiError(422, ErrorCode.UNSUPPORTED_FORMAT, str(exc)) from None
    return base.model_copy(
        update={"columns": page["schema"], "rows": page["rows"], "row_count": page["row_count"]}
    )


@router.get(
    "/artifacts/{artifact_id}/raw",
    responses={200: {"content": {"application/octet-stream": {}}}, 401: _E, 404: _E, 410: _E},
)
async def raw_artifact(
    artifact_id: str, backend: Backend, settings: Settings, caller: Caller
) -> Response:
    """The bytes, inline, as their own media type; HTML rewritten and under the artifact CSP."""
    record = await _readable_or_404(backend, settings, artifact_id, caller)
    headers = {
        "Content-Disposition": _content_disposition("inline", _filename(record)),
        "X-Content-Type-Options": "nosniff",
    }
    if record.kind == "html":
        data = await _bytes_or_410(backend, record)
        headers["Content-Security-Policy"] = HTML_ARTIFACT_CSP
        body = rewrite_artifact_links(data.decode("utf-8", "replace"))
        return Response(content=body.encode("utf-8"), media_type=record.mime_type, headers=headers)
    path, is_temp = await _owned_object(backend, record)
    if record.mime_type.startswith("image/svg"):
        headers["Content-Security-Policy"] = HTML_ARTIFACT_CSP
    return _file_response(path, is_temp, record.mime_type, headers)


def _iter_then_delete(chunks: Iterator[bytes], path: Path, is_temp: bool) -> Iterator[bytes]:
    try:
        yield from chunks
    finally:
        if is_temp:
            path.unlink(missing_ok=True)


@router.get(
    "/artifacts/{artifact_id}/download",
    responses={
        200: {"content": {"application/octet-stream": {}}},
        401: _E,
        404: _E,
        410: _E,
        422: _E,
    },
)
async def download_artifact(
    artifact_id: str,
    backend: Backend,
    settings: Settings,
    caller: Caller,
    format: Annotated[str | None, Query(pattern="^(csv|xlsx|parquet)$")] = None,
) -> Response:
    """The bytes as an attachment; tables convert with ``?format=csv|xlsx|parquet``."""
    record = await _readable_or_404(backend, settings, artifact_id, caller)
    if format is not None and record.kind != "table":
        raise ApiError(
            422,
            ErrorCode.UNSUPPORTED_FORMAT,
            f"format {format!r} is not available for this {record.kind} artifact",
            {"available": []},
        )
    filename = _filename(record)
    if format in (None, "parquet"):
        path, is_temp = await _owned_object(backend, record)
        headers = {"Content-Disposition": _content_disposition("attachment", filename)}
        return _file_response(path, is_temp, record.mime_type, headers)
    path, is_temp = await _owned_object(backend, record)
    try:
        rows = await asyncio.to_thread(tables.row_count, path)
    except (tables.TablesUnavailable, tables.CorruptTable) as exc:
        if is_temp:
            path.unlink(missing_ok=True)
        raise ApiError(422, ErrorCode.UNSUPPORTED_FORMAT, str(exc)) from None
    headers = {
        "Content-Disposition": _content_disposition(
            "attachment", _with_extension(filename, str(format))
        )
    }
    if format == "csv":
        return StreamingResponse(
            _iter_then_delete(tables.iter_csv(path), path, is_temp),
            media_type=_FORMAT_MIME["csv"],
            headers=headers,
        )
    if rows > tables.XLSX_MAX_ROWS:
        if is_temp:
            path.unlink(missing_ok=True)
        raise ApiError(
            422,
            ErrorCode.UNSUPPORTED_FORMAT,
            f"{rows} rows exceed the XLSX limit of {tables.XLSX_MAX_ROWS}; use csv or parquet",
            {"row_count": rows, "max_rows": tables.XLSX_MAX_ROWS, "available": ["csv", "parquet"]},
        )
    fd, name = tempfile.mkstemp(prefix="vdreamer-xlsx-", suffix=".xlsx")
    os.close(fd)
    xlsx = Path(name)
    try:
        await asyncio.to_thread(tables.write_xlsx, path, xlsx)
    except BaseException:
        xlsx.unlink(missing_ok=True)
        raise
    finally:
        if is_temp:
            path.unlink(missing_ok=True)
    return _file_response(xlsx, True, _FORMAT_MIME["xlsx"], headers)
