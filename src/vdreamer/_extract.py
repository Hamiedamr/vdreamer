"""The ``safe_extract`` engine (PRD §3.5, A34 — security-critical).

Runs as ``python -I -m vdreamer._extract SPEC.json`` in a sandboxed child
(rlimits, dedicated uid, no network — see ``vdreamer.sandbox``), so the
caps below are enforced twice: by this code while it streams, and by the
kernel if it is ever wrong. It is also importable, for tests.

What it guarantees, for one ZIP archive:

- **Rejects the whole archive** on any entry with a traversal component
  (``..``), an absolute path (``/x``, ``\\x``, ``C:x``), a symlink, a NUL or
  control character, or encryption; on more than ``max_entries`` entries; on
  nesting deeper than ``max_depth`` (the outer archive is depth 1); when an
  entry or the whole archive inflates past its byte cap or past
  ``max_ratio`` × its compressed size; and on a corrupt or unsupported
  member. Sizes are counted while inflating — declared sizes are not trusted.
- **Decodes names** UTF-8 when the archive flags them so; otherwise tries
  UTF-8, then CP437, then CP1256 (Arabic Windows), as PRD §3.5 orders them.
  CP437 decodes any byte string, so it only wins when every non-ASCII byte is
  one of its accented letters (0x80–0xAF); box-drawing and Greek ranges mean
  the name was CP1256.
- **Never writes an archive path to disk:** each kept entry lands at
  ``out/{n:05d}``; the manifest maps it back to its archive path. Nested ZIPs
  are expanded in place (``a/inner.zip/b.txt``), never kept themselves.
- **Lists** every entry whose type ``allow`` does not admit as skipped, with
  its mime type.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import json
import mimetypes
import os
import re
import stat
import sys
import tempfile
import zipfile
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

_CHUNK: Final = 64 * 1024
_ZIP_MAGIC: Final = b"PK\x03\x04"
_CONTROL: Final = re.compile(r"[\x00-\x1f\x7f]")
_DRIVE: Final = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class ExtractLimits:
    """The A34 caps. Defaults suit a site-survey ZIP of photos and PDFs."""

    max_total_bytes: int = 1024 * 1024 * 1024
    max_file_bytes: int = 256 * 1024 * 1024
    max_entries: int = 10_000
    max_ratio: float = 100.0
    #: Ratio checks start above this many inflated bytes (tiny files compress wildly).
    ratio_floor_bytes: int = 1024 * 1024
    max_depth: int = 2


class UnsafeArchive(Exception):
    """The archive was rejected; ``code`` is stable, ``entry`` the offending path."""

    def __init__(self, code: str, message: str, entry: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.entry = entry

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "entry": self.entry}


def decode_name(info: zipfile.ZipInfo) -> str:
    """The entry name per §3.5: flagged UTF-8, else UTF-8 → CP437 → CP1256."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        raw = info.orig_filename.encode("cp437")
    except UnicodeEncodeError:  # zipfile already decoded it some other way
        return info.orig_filename
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if all(byte < 0x80 or 0x80 <= byte <= 0xAF for byte in raw):
        return raw.decode("cp437")
    try:
        return raw.decode("cp1256")
    except UnicodeDecodeError:
        return raw.decode("cp437")


def check_name(name: str) -> str:
    """The entry's normalized relative path, or :class:`UnsafeArchive`."""
    if _CONTROL.search(name):
        raise UnsafeArchive("invalid_name", "entry name contains a control character", name)
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or _DRIVE.match(normalized):
        raise UnsafeArchive("absolute_path", f"absolute entry path {name!r}", name)
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise UnsafeArchive("traversal", f"entry path climbs out of the archive: {name!r}", name)
    if not parts:
        raise UnsafeArchive("invalid_name", f"empty entry path {name!r}", name)
    return "/".join(parts)


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return info.create_system == 3 and stat.S_ISLNK(mode)


def mime_of(path: str) -> str:
    return mimetypes.guess_type(path, strict=False)[0] or "application/octet-stream"


def allowed(path: str, mime: str, allow: Sequence[str] | None) -> bool:
    """``allow`` holds filename globs (``*.jpg``) and mime patterns (``image/*``)."""
    if allow is None:
        return True
    base = path.rsplit("/", 1)[-1].lower()
    for pattern in allow:
        rule = pattern.strip().lower()
        if "/" in rule:
            if fnmatch.fnmatchcase(mime.lower(), rule):
                return True
        elif fnmatch.fnmatchcase(base, rule):
            return True
    return False


class _Extractor:
    def __init__(self, out_dir: Path, allow: Sequence[str] | None, limits: ExtractLimits) -> None:
        self.out = out_dir
        self.allow = allow
        self.limits = limits
        self.total = 0
        self.entries = 0
        self.files: list[dict[str, Any]] = []
        self.skipped: list[dict[str, Any]] = []
        self.seen: set[str] = set()

    def _inflate(
        self, source: IO[bytes], sink: IO[bytes], *, entry: str, compressed: int
    ) -> tuple[int, str]:
        """Copy ``source`` → ``sink``, enforcing the byte and ratio caps as it goes."""
        limits = self.limits
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = source.read(_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            self.total += len(chunk)
            if size > limits.max_file_bytes:
                raise UnsafeArchive(
                    "file_too_large", f"entry inflates past {limits.max_file_bytes} bytes", entry
                )
            if self.total > limits.max_total_bytes:
                raise UnsafeArchive(
                    "total_too_large",
                    f"archive inflates past {limits.max_total_bytes} bytes",
                    entry,
                )
            if size > limits.ratio_floor_bytes and size > limits.max_ratio * max(compressed, 1):
                raise UnsafeArchive(
                    "ratio",
                    f"entry compression ratio exceeds {limits.max_ratio:g}:1",
                    entry,
                )
            digest.update(chunk)
            sink.write(chunk)
        return size, digest.hexdigest()

    def archive(self, handle: IO[bytes], *, prefix: str, depth: int) -> None:
        if depth > self.limits.max_depth:
            raise UnsafeArchive(
                "nesting", f"archives nest deeper than {self.limits.max_depth}", prefix or None
            )
        try:
            zf = zipfile.ZipFile(handle)
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            raise UnsafeArchive(
                "not_an_archive", f"not a ZIP archive: {exc}", prefix or None
            ) from exc
        with zf:
            infos = zf.infolist()
            self.entries += len(infos)
            if self.entries > self.limits.max_entries:
                raise UnsafeArchive(
                    "too_many_entries",
                    f"more than {self.limits.max_entries} entries",
                    prefix or None,
                )
            checked: list[tuple[zipfile.ZipInfo, str]] = []
            for info in infos:  # validate every name before writing anything
                name = decode_name(info)
                path = check_name(name)
                shown = f"{prefix}{path}"
                if info.flag_bits & 0x1 or info.flag_bits & 0x40:
                    raise UnsafeArchive("encrypted", "encrypted entries are refused", shown)
                if _is_symlink(info):
                    raise UnsafeArchive("symlink", "symlink entries are refused", shown)
                checked.append((info, path))
            for info, path in checked:
                if info.is_dir() or path.endswith("/"):
                    continue
                self._entry(zf, info, f"{prefix}{path}", depth)

    def _entry(self, zf: zipfile.ZipFile, info: zipfile.ZipInfo, path: str, depth: int) -> None:
        if path in self.seen:
            self.skipped.append({"path": path, "mime": mime_of(path), "reason": "duplicate"})
            return
        self.seen.add(path)
        index = len(self.files) + len(self.skipped)
        target = self.out / f"{index:05d}"
        try:
            with zf.open(info) as source, open(target, "wb") as sink:
                size, sha = self._inflate(
                    source, sink, entry=path, compressed=info.compress_size
                )
        except UnsafeArchive:
            target.unlink(missing_ok=True)
            raise
        except NotImplementedError as exc:
            target.unlink(missing_ok=True)
            raise UnsafeArchive("unsupported", f"unsupported entry: {exc}", path) from exc
        except (zipfile.BadZipFile, zlib.error, EOFError, OSError, RuntimeError) as exc:
            target.unlink(missing_ok=True)
            raise UnsafeArchive("corrupt", f"corrupt entry: {exc}", path) from exc
        with open(target, "rb") as probe:
            magic = probe.read(4)
        if magic == _ZIP_MAGIC:
            with open(target, "rb") as nested:
                self.archive(nested, prefix=f"{path}/", depth=depth + 1)
            target.unlink(missing_ok=True)
            return
        mime = mime_of(path)
        if not allowed(path, mime, self.allow):
            target.unlink(missing_ok=True)
            self.skipped.append({"path": path, "mime": mime, "reason": "not_allowed"})
            return
        self.files.append(
            {"path": path, "file": target.name, "mime": mime, "size": size, "sha256": sha}
        )


def extract(
    archive: str | Path,
    out_dir: str | Path,
    *,
    allow: Sequence[str] | None = None,
    limits: ExtractLimits | None = None,
) -> dict[str, Any]:
    """Extract ``archive`` into ``out_dir``; ``{"files": [...], "skipped": [...]}``.

    Raises:
        UnsafeArchive: the archive was rejected (nothing it wrote is kept).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    caps = limits or ExtractLimits()
    extractor = _Extractor(out, allow, caps)
    size = os.path.getsize(archive)
    try:
        with open(archive, "rb") as handle:
            extractor.archive(handle, prefix="", depth=1)
        if extractor.total > caps.ratio_floor_bytes and extractor.total > caps.max_ratio * max(
            size, 1
        ):
            raise UnsafeArchive("ratio", f"archive compression ratio exceeds {caps.max_ratio:g}:1")
    except UnsafeArchive:
        for child in out.iterdir():
            child.unlink(missing_ok=True)
        raise
    return {"files": extractor.files, "skipped": extractor.skipped}


def main(argv: Sequence[str] | None = None) -> int:
    """``python -I -m vdreamer._extract SPEC.json`` → writes ``SPEC.result.json``.

    The spec names ``archive``, ``out``, ``allow``, ``limits``; the result is
    ``{"ok": true, "files", "skipped"}`` or ``{"ok": false, "error"}``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m vdreamer._extract SPEC.json", file=sys.stderr)
        return 64
    spec_path = Path(args[0])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    limits = ExtractLimits(**spec.get("limits", {}))
    result_path = spec_path.with_suffix(".result.json")
    try:
        result: dict[str, Any] = {
            "ok": True,
            **extract(spec["archive"], spec["out"], allow=spec.get("allow"), limits=limits),
        }
        code = 0
    except UnsafeArchive as exc:
        result = {"ok": False, "error": exc.as_dict()}
        code = 2
    except MemoryError:
        result = {"ok": False, "error": {"code": "resources", "message": "out of memory"}}
        code = 3
    fd, tmp = tempfile.mkstemp(dir=result_path.parent, suffix=".tmp")
    with open(fd, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    os.replace(tmp, result_path)
    return code


def limits_dict(limits: ExtractLimits) -> dict[str, Any]:
    return dataclasses.asdict(limits)


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    sys.exit(main())
