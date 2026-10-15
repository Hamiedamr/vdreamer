"""Inbound files: ``ctx.files.open(ref)`` and ``ctx.files.safe_extract``.

Upstream systems upload to the object store under ``inputs/`` and pass a
``FileRef`` in the run's input (the gateway refuses any other prefix).
Agents read through ``ctx.files``:

- ``open(ref)`` → a binary file object (``aopen`` / ``read_bytes`` /
  ``aread_bytes`` alongside);
- ``safe_extract(ref, allow=[...]) -> Manifest`` unpacks a ZIP **in the
  sandbox** (``vdreamer._extract`` under ``vdreamer.sandbox``'s launcher):
  caps on sizes, entry count, compression ratio and nesting (≤ 2);
  traversal, absolute paths, symlinks and encryption are rejected
  (:class:`UnsafeArchiveError`); names decode UTF-8 → CP437 → CP1256. Kept
  files go to ``extracted/{thread_id}/{run_id}/…`` and the manifest lists
  ``{path, object_key, mime, size, sha256}`` per file plus the skipped types.

``ctx.files`` reads keys under ``inputs/`` and under the current run's own
``extracted/`` prefix — never another run's, never artifacts.
"""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING, Final

import pydantic

from vdreamer._extract import ExtractLimits, limits_dict
from vdreamer.artifact_data import extension_for
from vdreamer.artifacts import current_artifact_store, run_blocking
from vdreamer.context import FileRef
from vdreamer.objects import EXTRACTED_PREFIX, INPUTS_PREFIX, ObjectStore, check_key

if TYPE_CHECKING:
    from vdreamer.context import VDreamerContext

#: How long extraction may take before the child is killed.
EXTRACT_TIMEOUT_S: Final = 300.0


class ManifestEntry(pydantic.BaseModel):
    """One extracted file: its archive path and where its bytes now live."""

    path: str
    object_key: str
    mime: str
    size: int
    sha256: str


class SkippedEntry(pydantic.BaseModel):
    """An entry ``allow`` did not admit (or a duplicate path)."""

    path: str
    mime: str
    reason: str


class Manifest(pydantic.BaseModel):
    """``safe_extract``'s result: the kept files (iterate it) and what was skipped."""

    files: list[ManifestEntry] = pydantic.Field(default_factory=list)
    skipped: list[SkippedEntry] = pydantic.Field(default_factory=list)

    def __iter__(self) -> Iterator[ManifestEntry]:  # type: ignore[override]
        return iter(self.files)

    def __len__(self) -> int:
        return len(self.files)


class UnsafeArchiveError(Exception):
    """``safe_extract`` rejected the archive; ``code`` is stable (``traversal``, ``ratio``, …)."""

    def __init__(self, code: str, message: str, entry: str | None = None) -> None:
        super().__init__(f"{code}: {message}" + (f" ({entry})" if entry else ""))
        self.code = code
        self.entry = entry


FileLike = FileRef | ManifestEntry | str


class Files:
    """``ctx.files`` — bound to one run's context."""

    def __init__(self, ctx: VDreamerContext) -> None:
        self._ctx = ctx

    @property
    def objects(self) -> ObjectStore:
        store = current_artifact_store()
        if store is None:
            raise RuntimeError("ctx.files needs a worker-bound object store (none is bound)")
        return store.objects

    def key(self, ref: FileLike) -> str:
        """The object key ``ref`` names, if this run may read it."""
        if isinstance(ref, FileRef):
            return ref.object_key
        key = ref.object_key if isinstance(ref, ManifestEntry) else ref
        check_key(key)
        own = f"{EXTRACTED_PREFIX}{self._ctx.thread_id}/{self._ctx.run_id}/"
        if key.startswith(INPUTS_PREFIX) or key.startswith(own):
            return key
        raise PermissionError(f"ctx.files cannot read {key!r} (inputs/ and this run's extracts)")

    # -- reading -------------------------------------------------------------

    async def aread_bytes(self, ref: FileLike) -> bytes:
        key = self.key(ref)
        data = await self.objects.aget_bytes(key)
        if data is None:
            raise FileNotFoundError(f"no object {key!r}")
        return data

    def read_bytes(self, ref: FileLike) -> bytes:
        return run_blocking(self.aread_bytes(ref))

    async def adownload(self, ref: FileLike, dest: str | Path) -> Path:
        """Copy the object to ``dest`` (no whole-file buffering on MinIO)."""
        key = self.key(ref)
        if not await self.objects.aget_to_file(key, dest):
            raise FileNotFoundError(f"no object {key!r}")
        return Path(dest)

    async def aopen(self, ref: FileLike) -> IO[bytes]:
        """A binary file object for ``ref`` (spooled to a temp file when it is remote)."""
        key = self.key(ref)
        local = self.objects.local_path(key)
        if local is not None:
            return open(local, "rb")
        spool = tempfile.TemporaryFile()
        try:
            data = await self.objects.aget_bytes(key)
            if data is None:
                raise FileNotFoundError(f"no object {key!r}")
            spool.write(data)
            spool.seek(0)
        except BaseException:
            spool.close()
            raise
        return spool

    def open(self, ref: FileLike) -> IO[bytes]:
        """``ctx.files.open(ref)`` — a binary file object; close it (or use ``with``)."""
        return run_blocking(self.aopen(ref))

    # -- safe_extract ------------------------------------------------------

    async def asafe_extract(
        self,
        ref: FileLike,
        allow: Sequence[str] | None = None,
        *,
        limits: ExtractLimits | None = None,
        timeout_s: float = EXTRACT_TIMEOUT_S,
    ) -> Manifest:
        """Unpack the ZIP ``ref`` in the sandbox; see the module docstring.

        ``allow`` holds filename globs (``"*.jpg"``) and mime patterns
        (``"image/*"``); ``None`` keeps every type.

        Raises:
            UnsafeArchiveError: the archive was rejected (nothing is kept).
        """
        from vdreamer.sandbox import run_confined, scratch_dir

        key = self.key(ref)
        with scratch_dir() as scratch:
            archive = scratch / "archive.zip"
            await self.adownload(key, archive)
            spec = {
                "archive": str(archive),
                "out": str(scratch / "out"),
                "allow": list(allow) if allow is not None else None,
                "limits": limits_dict(limits or ExtractLimits()),
            }
            spec_path = scratch / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            done = await run_confined(
                [sys.executable, "-I", "-B", "-m", "vdreamer._extract", str(spec_path)],
                cwd=scratch,
                timeout_s=timeout_s,
                allow_unconfined=True,
            )
            result_path = spec_path.with_suffix(".result.json")
            if done.timed_out:
                raise UnsafeArchiveError("timeout", f"extraction exceeded {timeout_s:g}s")
            if not result_path.is_file():
                raise UnsafeArchiveError(
                    "resources",
                    f"the extractor died (exit {done.exit_code}): {done.stderr[-500:]}",
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not result.get("ok"):
                error = result.get("error") or {}
                raise UnsafeArchiveError(
                    str(error.get("code") or "rejected"),
                    str(error.get("message") or "archive rejected"),
                    error.get("entry"),
                )
            batch = uuid.uuid4().hex[:12]
            prefix = f"{EXTRACTED_PREFIX}{self._ctx.thread_id}/{self._ctx.run_id}/{batch}/"
            entries: list[ManifestEntry] = []
            for item in result["files"]:
                source = scratch / "out" / str(item["file"])
                if source.is_symlink() or not source.is_file():
                    continue
                object_key = check_key(
                    f"{prefix}{len(entries):05d}.{extension_for(item['mime'], item['path'])}"
                )
                await self.objects.aput_file(object_key, source, str(item["mime"]))
                entries.append(
                    ManifestEntry(
                        path=str(item["path"]),
                        object_key=object_key,
                        mime=str(item["mime"]),
                        size=int(item["size"]),
                        sha256=str(item["sha256"]),
                    )
                )
            skipped = [SkippedEntry(**item) for item in result.get("skipped", [])]
        return Manifest(files=entries, skipped=skipped)

    def safe_extract(
        self,
        ref: FileLike,
        allow: Sequence[str] | None = None,
        *,
        limits: ExtractLimits | None = None,
        timeout_s: float = EXTRACT_TIMEOUT_S,
    ) -> Manifest:
        """``ctx.files.safe_extract(ref, allow=[...]) -> Manifest`` (synchronous)."""
        return run_blocking(self.asafe_extract(ref, allow, limits=limits, timeout_s=timeout_s))
