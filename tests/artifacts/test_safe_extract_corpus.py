"""Milestone 6 exit: ``safe_extract`` defeats the traversal / zip-bomb corpus (A34).

Each hostile archive is built here, byte for byte, and must be *rejected*
with a stable code — with nothing left in the output directory. The benign
cases pin the manifest shape, the ``allow`` filter, nesting, and the
UTF-8 → CP437 → CP1256 name decoding.
"""

from __future__ import annotations

import io
import stat
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

from vdreamer._extract import ExtractLimits, UnsafeArchive, decode_name, extract, main

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")  # zipfile's duplicate-name warning


def _zip(
    entries: list[tuple[str | zipfile.ZipInfo, bytes]], method: int = zipfile.ZIP_DEFLATED
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", method) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buffer.getvalue()


def _run(tmp_path: Path, data: bytes, **kwargs: object) -> dict[str, object]:
    archive = tmp_path / "in.zip"
    archive.write_bytes(data)
    return extract(archive, tmp_path / "out", **kwargs)  # type: ignore[arg-type]


def _rejects(tmp_path: Path, data: bytes, code: str, **kwargs: object) -> UnsafeArchive:
    with pytest.raises(UnsafeArchive) as info:
        _run(tmp_path, data, **kwargs)
    assert info.value.code == code, info.value.message
    out = tmp_path / "out"
    assert not out.exists() or list(out.iterdir()) == [], "a rejected archive leaves nothing"
    return info.value


def _raw_name_zip(raw_name: bytes, data: bytes = b"x", flags: int = 0) -> bytes:
    """A stored one-entry ZIP whose name bytes are exactly ``raw_name`` (no UTF-8 flag)."""
    crc = zlib.crc32(data)
    local = struct.pack(
        "<IHHHHHIIIHH", 0x04034B50, 20, flags, 0, 0, 0, crc, len(data), len(data),
        len(raw_name), 0,
    )
    central = struct.pack(
        "<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, flags, 0, 0, 0, crc, len(data), len(data),
        len(raw_name), 0, 0, 0, 0, 0, 0,
    )
    body = local + raw_name + data
    directory = central + raw_name
    end = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(directory), len(body), 0)
    return body + directory + end


# -- hostile ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["../evil.sh", "a/../../evil.sh", "a\\..\\..\\evil.sh", "ok/../../../etc/passwd"],
)
def test_traversal_is_rejected(tmp_path: Path, name: str) -> None:
    error = _rejects(tmp_path, _zip([("fine.txt", b"ok"), (name, b"pwn")]), "traversal")
    assert error.entry in {name, name.replace("\\", "/")}  # Windows zipfile normalizes "\\"


@pytest.mark.parametrize("name", ["/etc/passwd", "\\windows\\system32\\x.dll", "C:/x.txt", "c:x"])
def test_absolute_paths_are_rejected(tmp_path: Path, name: str) -> None:
    _rejects(tmp_path, _zip([(name, b"pwn")]), "absolute_path")


def test_symlinks_are_rejected(tmp_path: Path) -> None:
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    _rejects(tmp_path, _zip([(info, b"/etc/passwd")]), "symlink")


def test_encryption_is_rejected(tmp_path: Path) -> None:
    _rejects(tmp_path, _raw_name_zip(b"secret.txt", flags=0x1), "encrypted")


def test_control_characters_in_names_are_rejected(tmp_path: Path) -> None:
    _rejects(tmp_path, _raw_name_zip(b"a\x00b.txt"), "invalid_name")


def test_a_high_ratio_bomb_is_rejected_while_inflating(tmp_path: Path) -> None:
    bomb = _zip([("zeros.bin", b"\0" * (50 * 1024 * 1024))])  # ~50 KB on the wire
    assert len(bomb) < 100 * 1024
    _rejects(tmp_path, bomb, "ratio")


def test_the_per_file_cap_counts_inflated_bytes(tmp_path: Path) -> None:
    limits = ExtractLimits(max_file_bytes=1024 * 1024, max_ratio=10_000)
    big = _zip([("big.bin", bytes(range(256)) * 12_000)])
    error = _rejects(tmp_path, big, "file_too_large", limits=limits)
    assert error.entry == "big.bin"


def test_lying_size_headers_are_rejected(tmp_path: Path) -> None:
    data = bytearray(_zip([("big.bin", b"\x01" * (3 * 1024 * 1024))], method=zipfile.ZIP_STORED))
    # Declare 10 bytes in both headers: inflation stops there and the CRC no longer matches.
    struct.pack_into("<I", data, data.find(b"PK\x03\x04") + 22, 10)
    struct.pack_into("<I", data, data.find(b"PK\x01\x02") + 24, 10)
    _rejects(tmp_path, bytes(data), "corrupt")


def test_the_total_cap(tmp_path: Path) -> None:
    entries = [(f"f{i}.bin", bytes([i]) * 400_000) for i in range(5)]
    limits = ExtractLimits(max_total_bytes=1_000_000, max_ratio=10_000)
    _rejects(tmp_path, _zip(entries), "total_too_large", limits=limits)


def test_overlapping_entries_bomb_hits_the_total_cap(tmp_path: Path) -> None:
    """Fifield-style: many central-directory entries share one local payload."""
    payload = b"\0" * (2 * 1024 * 1024)
    single = _zip([("k000", payload)])
    local_end = single.find(b"PK\x01\x02")
    central_end = single.find(b"PK\x05\x06")
    central = single[local_end:central_end]
    count = 200
    # Distinct central names (so the duplicate guard does not stop early), one payload.
    directory = b"".join(central[:46] + f"{i:04d}".encode() + central[50:] for i in range(count))
    end = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, count, count, len(directory), local_end, 0)
    limits = ExtractLimits(max_total_bytes=64 * 1024 * 1024, max_ratio=10_000)
    with pytest.raises(UnsafeArchive) as info:
        _run(tmp_path, single[:local_end] + directory + end, limits=limits)
    # Either the total cap trips, or zipfile notices the local/central name mismatch first.
    assert info.value.code in {"total_too_large", "corrupt"}


def test_too_many_entries(tmp_path: Path) -> None:
    entries = [(f"f{i}.txt", b"x") for i in range(30)]
    _rejects(tmp_path, _zip(entries), "too_many_entries", limits=ExtractLimits(max_entries=20))


def test_nesting_beyond_two_levels_is_rejected(tmp_path: Path) -> None:
    level3 = _zip([("deep.txt", b"deep")])
    level2 = _zip([("l3.zip", level3)])
    level1 = _zip([("l2.zip", level2)])
    error = _rejects(tmp_path, level1, "nesting")
    assert error.entry == "l2.zip/l3.zip/"


def test_a_recursive_quine_style_nest_stops_at_the_depth_cap(tmp_path: Path) -> None:
    inner = _zip([("x.txt", b"x")])
    for _ in range(10):
        inner = _zip([("again.zip", inner)])
    _rejects(tmp_path, inner, "nesting")


def test_not_a_zip(tmp_path: Path) -> None:
    _rejects(tmp_path, b"%PDF-1.7 not an archive", "not_an_archive")


def test_a_corrupt_member(tmp_path: Path) -> None:
    data = bytearray(_zip([("a.txt", b"hello world " * 100)]))
    start = data.find(b"a.txt") + 5
    data[start + 5 : start + 15] = b"\xff" * 10
    with pytest.raises(UnsafeArchive) as info:
        _run(tmp_path, bytes(data))
    assert info.value.code in {"corrupt", "unsupported"}


# -- benign -----------------------------------------------------------------


def test_manifest_allow_filter_and_one_nested_level(tmp_path: Path) -> None:
    inner = _zip([("scan.pdf", b"%PDF-1.7"), ("inner.exe", b"MZ")])
    result = _run(
        tmp_path,
        _zip([("site/a.jpg", b"\xff\xd8\xffjpeg"), ("site/docs.zip", inner), ("site/", b"")]),
        allow=["*.jpg", "application/pdf"],
    )
    files = result["files"]
    assert [f["path"] for f in files] == ["site/a.jpg", "site/docs.zip/scan.pdf"]  # type: ignore[index]
    first = files[0]  # type: ignore[index]
    assert first["mime"] == "image/jpeg" and first["size"] == 7 and len(first["sha256"]) == 64
    assert (tmp_path / "out" / first["file"]).read_bytes() == b"\xff\xd8\xffjpeg"
    assert [s["path"] for s in result["skipped"]] == ["site/docs.zip/inner.exe"]  # type: ignore[index]
    # Archive paths never reach the disk: outputs are numbered files only.
    assert all(p.name.isdigit() for p in (tmp_path / "out").iterdir())


def test_allow_none_keeps_everything(tmp_path: Path) -> None:
    result = _run(tmp_path, _zip([("a.bin", b"1"), ("b.txt", b"2")]))
    assert len(result["files"]) == 2 and result["skipped"] == []  # type: ignore[arg-type]


def test_name_decoding_utf8_cp437_cp1256() -> None:
    def info_for(raw: bytes, flags: int = 0) -> zipfile.ZipInfo:
        with zipfile.ZipFile(io.BytesIO(_raw_name_zip(raw, flags=flags))) as zf:
            return zf.infolist()[0]

    assert decode_name(info_for("تقرير.pdf".encode(), flags=0x800)) == "تقرير.pdf"
    assert decode_name(info_for("rapport-été.txt".encode())) == "rapport-été.txt"  # unflagged UTF-8
    assert decode_name(info_for("résumé.txt".encode("cp437"))) == "résumé.txt"
    assert decode_name(info_for("صورة-الموقع.jpg".encode("cp1256"))) == "صورة-الموقع.jpg"


def test_the_cli_entry_writes_a_result_file(tmp_path: Path) -> None:
    import json

    (tmp_path / "in.zip").write_bytes(_zip([("../x", b"x")]))
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"archive": str(tmp_path / "in.zip"), "out": str(tmp_path / "out")}),
        encoding="utf-8",
    )
    assert main([str(spec)]) == 2
    result = json.loads((tmp_path / "spec.result.json").read_text(encoding="utf-8"))
    assert result == {
        "ok": False,
        "error": {"code": "traversal", "message": result["error"]["message"], "entry": "../x"},
    }
