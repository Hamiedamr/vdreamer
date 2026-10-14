"""Object storage for artifact bytes and inbound files (PRD §2.1, §3.4, §3.5, A9, A17).

One key space, three prefixes:

- ``artifacts/{thread_id}/{run_id}/{artifact_id}.{ext}`` — artifact bytes (D7);
- ``inputs/…`` — files upstream systems upload for runs (``FileRef``, A17);
  write-only for them, read-only for vdreamer;
- ``extracted/{thread_id}/{run_id}/…`` — what ``ctx.files.safe_extract``
  unpacked (A34).

Production is MinIO (:class:`MinioObjectStore`, the ``[minio]`` extra), with
a bucket lifecycle rule that deletes ``artifacts/`` and ``extracted/`` after
30 days (A9 — a missing object then reads back as ``None`` and the gateway
answers ``410``). ``vdreamer dev`` keeps objects on the local filesystem next
to its ``.db`` file (:class:`FilesystemObjectStore`, D11) and sweeps expired
ones at boot; tests use :class:`InMemoryObjectStore`.

The core methods are synchronous — both the MinIO client and the filesystem
are blocking APIs — and every one has an ``a``-prefixed twin that runs it on
a thread, so async callers never block their loop.
"""

from __future__ import annotations

import abc
import asyncio
import io
import os
import re
import shutil
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

#: The prefixes vdreamer reads and writes (see the module docstring).
ARTIFACTS_PREFIX: Final = "artifacts/"
INPUTS_PREFIX: Final = "inputs/"
EXTRACTED_PREFIX: Final = "extracted/"

#: A9: artifacts (and extracted inputs) are deleted after 30 days.
RETENTION_DAYS: Final = 30
EXPIRING_PREFIXES: Final[tuple[str, ...]] = (ARTIFACTS_PREFIX, EXTRACTED_PREFIX)

MAX_KEY_LENGTH: Final = 1024
_BAD_KEY_CHARS: Final = re.compile(r"[\x00-\x1f\x7f\\]")

#: MinIO settings — env only (A23); the keys are secrets, never YAML.
MINIO_ENDPOINT_ENV: Final = "VDREAMER_MINIO_ENDPOINT"
MINIO_ACCESS_KEY_ENV: Final = "VDREAMER_MINIO_ACCESS_KEY"
MINIO_SECRET_KEY_ENV: Final = "VDREAMER_MINIO_SECRET_KEY"
MINIO_BUCKET_ENV: Final = "VDREAMER_MINIO_BUCKET"
MINIO_SECURE_ENV: Final = "VDREAMER_MINIO_SECURE"
DEFAULT_BUCKET: Final = "vdreamer"


class ObjectKeyError(ValueError):
    """A key that is not a clean relative ``a/b/c`` path."""


def check_key(key: str) -> str:
    """``key`` if it is a clean relative object key, else :class:`ObjectKeyError`.

    No leading ``/``, no backslashes or control characters, no empty, ``.``
    or ``..`` segments — so a key can never climb out of its prefix, on
    MinIO or on the dev filesystem.
    """
    if not isinstance(key, str) or not key or len(key) > MAX_KEY_LENGTH:
        raise ObjectKeyError(f"invalid object key {key!r}")
    if key.startswith("/") or _BAD_KEY_CHARS.search(key):
        raise ObjectKeyError(f"invalid object key {key!r}")
    if any(part in ("", ".", "..") for part in key.split("/")):
        raise ObjectKeyError(f"invalid object key {key!r}")
    return key


class ObjectStore(abc.ABC):
    """Bytes by key. Missing objects read back as ``None`` / ``False``."""

    @abc.abstractmethod
    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        """Store ``data`` under ``key`` (overwrites)."""

    def put_file(self, key: str, path: str | Path, content_type: str) -> None:
        """Store the file at ``path`` under ``key``; stores override to stream it."""
        self.put_bytes(key, Path(path).read_bytes(), content_type)

    @abc.abstractmethod
    def get_bytes(self, key: str) -> bytes | None:
        """The object's bytes, or ``None`` when it does not exist (A9)."""

    def get_to_file(self, key: str, dest: str | Path) -> bool:
        """Write the object to ``dest``; ``False`` when it does not exist."""
        data = self.get_bytes(key)
        if data is None:
            return False
        Path(dest).write_bytes(data)
        return True

    @abc.abstractmethod
    def size(self, key: str) -> int | None:
        """The object's size in bytes, or ``None`` when it does not exist."""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Remove the object; a missing one is not an error."""

    def local_path(self, key: str) -> Path | None:
        """A readable local path for the object, when the store has one (dev)."""
        return None

    # -- async twins -------------------------------------------------------

    async def aput_bytes(self, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(self.put_bytes, key, data, content_type)

    async def aput_file(self, key: str, path: str | Path, content_type: str) -> None:
        await asyncio.to_thread(self.put_file, key, path, content_type)

    async def aget_bytes(self, key: str) -> bytes | None:
        return await asyncio.to_thread(self.get_bytes, key)

    async def aget_to_file(self, key: str, dest: str | Path) -> bool:
        return await asyncio.to_thread(self.get_to_file, key, dest)

    async def asize(self, key: str) -> int | None:
        return await asyncio.to_thread(self.size, key)

    async def adelete(self, key: str) -> None:
        await asyncio.to_thread(self.delete, key)


class InMemoryObjectStore(ObjectStore):
    """Dict-backed — tests and the ``dev --input`` smoke."""

    def __init__(self) -> None:
        self._objects: dict[str, tuple[bytes, str]] = {}
        self._lock = threading.Lock()

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        check_key(key)
        with self._lock:
            self._objects[key] = (bytes(data), content_type)

    def get_bytes(self, key: str) -> bytes | None:
        with self._lock:
            found = self._objects.get(key)
        return found[0] if found is not None else None

    def size(self, key: str) -> int | None:
        data = self.get_bytes(key)
        return len(data) if data is not None else None

    def delete(self, key: str) -> None:
        with self._lock:
            self._objects.pop(key, None)

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._objects)


class FilesystemObjectStore(ObjectStore):
    """Objects as files under ``root`` (``vdreamer dev``, D11): ``root/<key>``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        """Where ``key`` lives; refuses anything resolving outside ``root``."""
        check_key(key)
        path = self.root.joinpath(*key.split("/"))
        root = self.root.resolve()
        if not path.resolve().is_relative_to(root):
            raise ObjectKeyError(f"object key {key!r} escapes the store root")
        return path

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def put_file(self, key: str, path: str | Path, content_type: str) -> None:
        dest = self.path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        shutil.copyfile(path, tmp)
        os.replace(tmp, dest)

    def get_bytes(self, key: str) -> bytes | None:
        try:
            return self.path(key).read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            return None

    def get_to_file(self, key: str, dest: str | Path) -> bool:
        try:
            shutil.copyfile(self.path(key), dest)
        except (FileNotFoundError, IsADirectoryError):
            return False
        return True

    def size(self, key: str) -> int | None:
        try:
            path = self.path(key)
            return path.stat().st_size if path.is_file() else None
        except FileNotFoundError:
            return None

    def delete(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)

    def local_path(self, key: str) -> Path | None:
        path = self.path(key)
        return path if path.is_file() else None

    def sweep(
        self,
        prefixes: tuple[str, ...] = EXPIRING_PREFIXES,
        *,
        older_than_s: float = RETENTION_DAYS * 86400,
        now: float | None = None,
    ) -> int:
        """Delete objects under ``prefixes`` older than the retention (A9); returns the count.

        The dev twin of the MinIO bucket lifecycle rule; ``vdreamer dev`` runs
        it at boot.
        """
        cutoff = (time.time() if now is None else now) - older_than_s
        removed = 0
        for prefix in prefixes:
            base = self.root.joinpath(*prefix.strip("/").split("/"))
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
        return removed


class MinioConfigError(Exception):
    """The MinIO environment is incomplete, or the ``[minio]`` extra is missing."""


class MinioObjectStore(ObjectStore):
    """Objects in one MinIO bucket (PRD §2.1; the ``[minio]`` extra)."""

    def __init__(self, client: Any, bucket: str = DEFAULT_BUCKET) -> None:
        self._client = client
        self.bucket = bucket

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> MinioObjectStore | None:
        """The store ``VDREAMER_MINIO_*`` describes, or ``None`` when no endpoint is set.

        Raises:
            MinioConfigError: an endpoint without both keys, or no ``minio`` package.
        """
        env = os.environ if environ is None else environ
        endpoint = env.get(MINIO_ENDPOINT_ENV, "").strip()
        if not endpoint:
            return None
        access = env.get(MINIO_ACCESS_KEY_ENV, "")
        secret = env.get(MINIO_SECRET_KEY_ENV, "")
        if not access or not secret:
            raise MinioConfigError(
                f"{MINIO_ENDPOINT_ENV} is set but {MINIO_ACCESS_KEY_ENV} / "
                f"{MINIO_SECRET_KEY_ENV} are not"
            )
        secure = env.get(MINIO_SECURE_ENV, "true").strip().lower() not in ("0", "false", "no")
        try:
            from minio import Minio
        except ImportError as exc:  # pragma: no cover - the extra is installed in dev
            raise MinioConfigError(
                "MinIO storage needs the [minio] extra: pip install 'vdreamer[minio]'"
            ) from exc
        client = Minio(endpoint, access_key=access, secret_key=secret, secure=secure)
        return cls(client, env.get(MINIO_BUCKET_ENV, "").strip() or DEFAULT_BUCKET)

    def setup(self, retention_days: int = RETENTION_DAYS) -> None:
        """Create the bucket if missing and install the A9 lifecycle rule (``migrate``)."""
        from minio.commonconfig import ENABLED, Filter
        from minio.lifecycleconfig import Expiration, LifecycleConfig, Rule

        if not self._client.bucket_exists(self.bucket):
            self._client.make_bucket(self.bucket)
        rules = [
            Rule(
                ENABLED,
                rule_filter=Filter(prefix=prefix),
                rule_id=f"vdreamer-expire-{prefix.strip('/')}",
                expiration=Expiration(days=retention_days),
            )
            for prefix in EXPIRING_PREFIXES
        ]
        self._client.set_bucket_lifecycle(self.bucket, LifecycleConfig(rules))

    def probe(self) -> None:
        """Raise unless the bucket is reachable (readiness)."""
        if not self._client.bucket_exists(self.bucket):
            raise RuntimeError(f"MinIO bucket {self.bucket!r} does not exist (run migrate)")

    @staticmethod
    def _missing(exc: Exception) -> bool:
        return getattr(exc, "code", None) in ("NoSuchKey", "NoSuchObject", "NoSuchBucket")

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        check_key(key)
        self._client.put_object(
            self.bucket, key, io.BytesIO(data), len(data), content_type=content_type
        )

    def put_file(self, key: str, path: str | Path, content_type: str) -> None:
        check_key(key)
        self._client.fput_object(self.bucket, key, str(path), content_type=content_type)

    def get_bytes(self, key: str) -> bytes | None:
        from minio.error import S3Error

        check_key(key)
        try:
            response = self._client.get_object(self.bucket, key)
        except S3Error as exc:
            if self._missing(exc):
                return None
            raise
        try:
            return bytes(response.read())
        finally:
            response.close()
            response.release_conn()

    def get_to_file(self, key: str, dest: str | Path) -> bool:
        from minio.error import S3Error

        check_key(key)
        try:
            self._client.fget_object(self.bucket, key, str(dest))
        except S3Error as exc:
            if self._missing(exc):
                return False
            raise
        return True

    def size(self, key: str) -> int | None:
        from minio.error import S3Error

        check_key(key)
        try:
            return int(self._client.stat_object(self.bucket, key).size)
        except S3Error as exc:
            if self._missing(exc):
                return None
            raise

    def delete(self, key: str) -> None:
        check_key(key)
        self._client.remove_object(self.bucket, key)
