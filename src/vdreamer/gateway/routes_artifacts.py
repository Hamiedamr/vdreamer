"""The artifact proxy (PRD §2.5, §3.4, A8, A9, A25).

The gateway proxies artifact bytes (A8) so URLs stay same-origin behind the
Web Backend (A25). Access: the owner of the artifact's thread, or a holder of
``artifact:read:all`` (A8; "granted approver" access lands with Milestone 8).
An artifact the caller may not read is a ``404``, like any other
owner-scoped resource. A record whose object is gone (30-day lifecycle, A9)
answers ``410 artifact_expired``.

Milestone 6 fills in what this milestone only fixes the shape of: paged and
sorted table ``/preview`` over Parquet, CSV/XLSX conversion on ``/download``,
and ``artifact://`` rewriting in HTML.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Query, Response

from vdreamer.artifacts import ArtifactRecord
from vdreamer.gateway.auth import API_PREFIX, Identity, current_identity
from vdreamer.gateway.backend import GatewayBackend
from vdreamer.gateway.common import get_backend, get_settings, jsonable
from vdreamer.gateway.errors import ApiError, ErrorCode
from vdreamer.gateway.schemas import ArtifactOut, ArtifactPreview, ErrorResponse
from vdreamer.gateway.settings import GatewaySettings

router = APIRouter(prefix=API_PREFIX, tags=["artifacts"])

Backend = Annotated[GatewayBackend, Depends(get_backend)]
Settings = Annotated[GatewaySettings, Depends(get_settings)]
Caller = Annotated[Identity, Depends(current_identity)]

_E = {"model": ErrorResponse}

#: Served with every ``html`` artifact (§3.4): no external origin, ``connect-src
#: 'none'``. Inline script/style stay allowed — report HTML embeds its charts —
#: and clients frame it ``sandbox="allow-scripts"`` without same-origin (§6.3).
HTML_ARTIFACT_CSP: Final[str] = (
    "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; font-src 'self' data:; connect-src 'none'; "
    "frame-ancestors 'self'; base-uri 'none'; form-action 'none'"
)

#: ``/preview`` page bound (§3.4: ``limit ≤ 1000``).
MAX_PREVIEW_LIMIT: Final[int] = 1000

#: ``/download`` formats per kind available in this milestone — the stored
#: bytes only. CSV/XLSX conversion of tables lands with Milestone 6.
_NATIVE_FORMATS: Final[dict[str, str]] = {"table": "parquet"}


def artifact_urls(artifact_id: str) -> tuple[str, str]:
    """``(view_url, download_url)`` — the gateway routes (§3.4)."""
    return (
        f"{API_PREFIX}/artifacts/{artifact_id}/raw",
        f"{API_PREFIX}/artifacts/{artifact_id}/download",
    )


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
    raise ApiError(404, ErrorCode.ARTIFACT_NOT_FOUND, f"no artifact {artifact_id!r}")


async def _bytes_or_410(backend: GatewayBackend, record: ArtifactRecord) -> bytes:
    data = await backend.artifacts.read_bytes(record)
    if data is None:
        raise ApiError(
            410,
            ErrorCode.ARTIFACT_EXPIRED,
            f"artifact {record.artifact_id!r} is no longer stored (retention, A9)",
        )
    return data


def _filename(record: ArtifactRecord) -> str:
    return record.filename or record.name or record.artifact_id


def _content_disposition(disposition: str, filename: str) -> str:
    """RFC 6266: an ASCII-safe ``filename`` plus the exact UTF-8 ``filename*``."""
    fallback = re.sub(r"[^A-Za-z0-9._-]", "_", filename) or "artifact"
    quoted = urllib.parse.quote(filename, safe="")
    return f"{disposition}; filename=\"{fallback}\"; filename*=UTF-8''{quoted}"


@router.get(
    "/artifacts/{artifact_id}",
    response_model=ArtifactOut,
    responses={400: _E, 401: _E, 404: _E},
)
async def get_artifact(
    artifact_id: str, backend: Backend, settings: Settings, caller: Caller
) -> ArtifactOut:
    """The artifact's metadata — the ``artifact.created`` field set (§3.4)."""
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
    responses={400: _E, 401: _E, 404: _E, 422: _E},
)
async def preview_artifact(
    artifact_id: str,
    backend: Backend,
    settings: Settings,
    caller: Caller,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_PREVIEW_LIMIT)] = 100,
    sort: str | None = None,
) -> ArtifactPreview:
    """The stored preview block (§3.4); paged/sorted table rows land with Milestone 6."""
    record = await _readable_or_404(backend, settings, artifact_id, caller)
    return ArtifactPreview(
        artifact_id=record.artifact_id,
        kind=record.kind,
        preview=jsonable(dict(record.preview)),
        offset=offset,
        limit=limit,
    )


@router.get(
    "/artifacts/{artifact_id}/raw",
    responses={200: {"content": {"application/octet-stream": {}}}, 401: _E, 404: _E, 410: _E},
)
async def raw_artifact(
    artifact_id: str, backend: Backend, settings: Settings, caller: Caller
) -> Response:
    """The bytes, inline, as their own media type; HTML under the §3.4 CSP."""
    record = await _readable_or_404(backend, settings, artifact_id, caller)
    data = await _bytes_or_410(backend, record)
    headers = {"Content-Disposition": _content_disposition("inline", _filename(record))}
    if record.kind == "html":
        headers["Content-Security-Policy"] = HTML_ARTIFACT_CSP
    return Response(content=data, media_type=record.mime_type, headers=headers)


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
    """The bytes as an attachment. ``?format=`` must be the stored format until M6."""
    record = await _readable_or_404(backend, settings, artifact_id, caller)
    if format is not None and _NATIVE_FORMATS.get(record.kind) != format:
        raise ApiError(
            422,
            ErrorCode.UNSUPPORTED_FORMAT,
            f"format {format!r} is not available for this {record.kind} artifact",
            {"available": [_NATIVE_FORMATS[record.kind]] if record.kind in _NATIVE_FORMATS else []},
        )
    data = await _bytes_or_410(backend, record)
    headers = {"Content-Disposition": _content_disposition("attachment", _filename(record))}
    return Response(content=data, media_type=record.mime_type, headers=headers)
