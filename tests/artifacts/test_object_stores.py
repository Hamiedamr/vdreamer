"""Object stores (§3.4, A9, A17): key hygiene, the dev filesystem, and live MinIO.

The MinIO test runs against a real server when ``VDREAMER_TEST_MINIO_ENDPOINT``
(plus ``..._ACCESS_KEY`` / ``..._SECRET_KEY``) is set, e.g. a local
``docker run minio/minio server /data``; otherwise it skips.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest

from vdreamer.context import FileRef
from vdreamer.objects import (
    FilesystemObjectStore,
    InMemoryObjectStore,
    MinioConfigError,
    MinioObjectStore,
    ObjectKeyError,
    check_key,
)


@pytest.mark.parametrize(
    "key", ["", "/abs", "a/../b", "a//b", "./a", "a\\b", "a/\x00", "..", "a/."]
)
def test_unclean_keys_are_refused(key: str) -> None:
    with pytest.raises(ObjectKeyError):
        check_key(key)
    with pytest.raises(ObjectKeyError):
        InMemoryObjectStore().put_bytes(key, b"x", "text/plain")


@pytest.mark.parametrize(
    "key", ["artifacts/t/r/a.json", "inputs/../artifacts/x", "/inputs/x", "inputs/", "inputs\\x"]
)
def test_file_refs_must_be_clean_keys_under_inputs(key: str) -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        FileRef(object_key=key)
    assert FileRef(object_key="inputs/site 7/pkg.zip").object_key == "inputs/site 7/pkg.zip"


def test_filesystem_store_roundtrip_and_sweep(tmp_path: Path) -> None:
    store = FilesystemObjectStore(tmp_path)
    store.put_bytes("artifacts/t/r/a.json", b"{}", "application/json")
    assert (tmp_path / "artifacts/t/r/a.json").read_bytes() == b"{}"
    assert store.get_bytes("artifacts/t/r/a.json") == b"{}"
    assert store.size("artifacts/t/r/a.json") == 2
    assert store.local_path("artifacts/t/r/a.json") == tmp_path / "artifacts/t/r/a.json"
    assert store.get_bytes("artifacts/t/r/missing") is None
    store.put_bytes("inputs/keep.zip", b"pk", "application/zip")
    old = time.time() - 31 * 86400
    os.utime(tmp_path / "artifacts/t/r/a.json", (old, old))
    os.utime(tmp_path / "inputs/keep.zip", (old, old))
    assert store.sweep() == 1  # artifacts/ and extracted/ expire; inputs/ is upstream's
    assert store.get_bytes("artifacts/t/r/a.json") is None
    assert store.get_bytes("inputs/keep.zip") == b"pk"


def test_minio_config_is_env_only_and_complete() -> None:
    assert MinioObjectStore.from_env({}) is None
    with pytest.raises(MinioConfigError, match="ACCESS_KEY"):
        MinioObjectStore.from_env({"VDREAMER_MINIO_ENDPOINT": "minio.internal:9000"})
    store = MinioObjectStore.from_env(
        {
            "VDREAMER_MINIO_ENDPOINT": "minio.internal:9000",
            "VDREAMER_MINIO_ACCESS_KEY": "ak",
            "VDREAMER_MINIO_SECRET_KEY": "sk",
            "VDREAMER_MINIO_BUCKET": "art",
            "VDREAMER_MINIO_SECURE": "false",
        }
    )
    assert store is not None and store.bucket == "art"


@pytest.fixture
def minio() -> MinioObjectStore:
    endpoint = os.environ.get("VDREAMER_TEST_MINIO_ENDPOINT")
    if not endpoint:
        pytest.skip("VDREAMER_TEST_MINIO_ENDPOINT not set")
    store = MinioObjectStore.from_env(
        {
            "VDREAMER_MINIO_ENDPOINT": endpoint,
            "VDREAMER_MINIO_ACCESS_KEY": os.environ.get("VDREAMER_TEST_MINIO_ACCESS_KEY", ""),
            "VDREAMER_MINIO_SECRET_KEY": os.environ.get("VDREAMER_TEST_MINIO_SECRET_KEY", ""),
            "VDREAMER_MINIO_BUCKET": f"vdreamer-test-{uuid.uuid4().hex[:8]}",
            "VDREAMER_MINIO_SECURE": "false",
        }
    )
    assert store is not None
    store.setup()
    return store


def test_live_minio_roundtrip_and_lifecycle(minio: MinioObjectStore, tmp_path: Path) -> None:
    minio.probe()
    key = "artifacts/t1/run_1/art_1.parquet"
    minio.put_bytes(key, b"PAR1", "application/vnd.apache.parquet")
    assert minio.get_bytes(key) == b"PAR1" and minio.size(key) == 4
    source = tmp_path / "big.bin"
    source.write_bytes(os.urandom(3 * 1024 * 1024))
    minio.put_file("artifacts/t1/run_1/art_2.bin", source, "application/octet-stream")
    dest = tmp_path / "back.bin"
    assert minio.get_to_file("artifacts/t1/run_1/art_2.bin", dest)
    assert dest.read_bytes() == source.read_bytes()
    minio.delete(key)
    assert minio.get_bytes(key) is None and minio.size(key) is None
    assert not minio.get_to_file(key, tmp_path / "gone")
    rules = minio._client.get_bucket_lifecycle(minio.bucket).rules
    assert {(r.rule_filter.prefix, r.expiration.days) for r in rules} == {
        ("artifacts/", 30),
        ("extracted/", 30),
    }
