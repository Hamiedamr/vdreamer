"""The five artifact kinds (§3.4): accepted data, previews, and no auto-detection."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pydantic
import pytest

from vdreamer import tables
from vdreamer.artifact_data import JSON_PREVIEW_MAX_BYTES, encode, extension_for, image_info


def _png(width: int, height: int) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\0" * 4


def _jpeg(width: int, height: int) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\0" + b"\0" * 9
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x11\0"
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


def _table(kind_data: Any, tmp_path: Path) -> tuple[Any, pa.Table]:
    encoded = encode("table", kind_data, name="t")
    try:
        import pyarrow.parquet as pq

        return encoded, pq.read_table(encoded.path)
    finally:
        encoded.cleanup()


@pytest.mark.parametrize(
    "data",
    [
        pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}],
        ({"a": i, "b": c} for i, c in zip([1, 2, 3], "xyz", strict=True)),
    ],
    ids=["arrow", "pandas", "list", "generator"],
)
def test_tables_accept_arrow_pandas_and_row_iterators(data: Any, tmp_path: Path) -> None:
    encoded, table = _table(data, tmp_path)
    assert encoded.mime_type == tables.PARQUET_MIME and encoded.ext == "parquet"
    assert table.column("a").to_pylist() == [1, 2, 3]
    assert encoded.preview["row_count"] == 3
    assert encoded.preview["rows"][0] == {"a": 1, "b": "x"}
    assert {"name": "a", "type": "int64"} in encoded.preview["schema"]
    assert encoded.metadata == {"truncated": False}


def test_large_iterators_write_row_groups_and_truncate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow.parquet as pq

    monkeypatch.setattr(tables, "ROW_GROUP_ROWS", 1000)
    monkeypatch.setenv(tables.TABLE_MAX_ROWS_ENV, "4500")
    encoded = encode("table", ({"i": i} for i in range(10_000)), name="big")
    try:
        meta = pq.ParquetFile(encoded.path).metadata
        assert meta.num_rows == 4500 and meta.num_row_groups == 5
        assert encoded.metadata == {"truncated": True}
        assert encoded.preview["row_count"] == 4500 and len(encoded.preview["rows"]) == 10
    finally:
        encoded.cleanup()


def test_table_paging_and_json_safe_cells(tmp_path: Path) -> None:
    import datetime
    import decimal

    path = tmp_path / "t.parquet"
    rows = [
        {"d": datetime.date(2026, 1, i + 1), "n": decimal.Decimal(f"{i}.50"), "f": float(i)}
        for i in range(30)
    ]
    tables.write_parquet(rows, path)
    page = tables.page(path, offset=28, limit=5, sort=None)
    assert page["rows"] == [
        {"d": "2026-01-29", "n": "28.50", "f": 28.0},
        {"d": "2026-01-30", "n": "29.50", "f": 29.0},
    ]
    assert tables.page(path, offset=0, limit=1, sort="-f")["rows"][0]["f"] == 29.0
    with pytest.raises(tables.SortError):
        tables.page(path, offset=0, limit=1, sort="missing")


@pytest.mark.parametrize(
    "data",
    ["not rows", b"bytes", {"a": 1}, [1, 2, 3], 42],
    ids=["str", "bytes", "dict", "ints", "int"],
)
def test_unsupported_table_data_is_a_type_error(data: Any) -> None:
    with pytest.raises(TypeError):
        encode("table", data, name="t")


def test_images_png_jpeg_svg_with_dimensions(tmp_path: Path) -> None:
    assert image_info(_png(640, 480)) == ("image/png", 640, 480)
    assert image_info(_jpeg(320, 200)) == ("image/jpeg", 320, 200)
    svg = b'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 40">'
    assert image_info(svg) == ("image/svg+xml", 120, 40)
    sized = b'<svg width="300px" height="150" xmlns="http://www.w3.org/2000/svg"></svg>'
    assert image_info(sized) == ("image/svg+xml", 300, 150)
    path = tmp_path / "chart.png"
    path.write_bytes(_png(8, 9))
    encoded = encode("image", path, name="chart")
    assert (encoded.preview, encoded.filename, encoded.ext) == (
        {"width": 8, "height": 9},
        "chart.png",
        "png",
    )
    encoded = encode("image", str(path), name="chart")
    assert encoded.data == _png(8, 9)


@pytest.mark.parametrize("data", [b"GIF89a....", b"RIFF....WEBP", "not-a-path-or-bytes.png", 3])
def test_other_image_data_is_a_type_error(data: Any) -> None:
    with pytest.raises((TypeError, FileNotFoundError)):
        encode("image", data, name="x")


def test_html_preview_title_and_summary() -> None:
    page = (
        "<html><head><title> Site &amp; antenna\n report </title>"
        '<meta name="description" content="Tower 7 checks"></head></html>'
    )
    encoded = encode("html", page, name="report")
    assert encoded.preview == {"title": "Site & antenna report", "summary": "Tower 7 checks"}
    assert encoded.mime_type.startswith("text/html")
    assert encode("html", "<p>no title</p>", name="fallback").preview == {"title": "fallback"}
    with pytest.raises(TypeError):
        encode("html", b"<p>bytes</p>", name="x")


class Verdict(pydantic.BaseModel):
    ok: bool
    reasons: list[str]


def test_json_takes_models_and_dicts_with_size_rules() -> None:
    encoded = encode("json", Verdict(ok=True, reasons=["tilt"]), name="v")
    assert encoded.preview == {"body": {"ok": True, "reasons": ["tilt"]}}
    big = {"blob": "x" * (JSON_PREVIEW_MAX_BYTES + 10)}
    assert encode("json", big, name="v").preview == {"truncated": True}
    with pytest.raises(ValueError, match="at most"):
        encode("json", {"blob": "x" * (1024 * 1024 + 1)}, name="v")
    for bad in ([1, 2], "{}", {"x": object()}):
        with pytest.raises(TypeError):
            encode("json", bad, name="v")


def test_files_take_bytes_or_paths_with_a_mime(tmp_path: Path) -> None:
    encoded = encode("file", b"%PDF-1.7", name="scan", mime_type="application/pdf")
    assert encoded.preview == {"filename": "scan", "mime": "application/pdf"}
    assert encoded.ext == "pdf"
    path = tmp_path / "report final.pdf"
    path.write_bytes(b"%PDF")
    encoded = encode("file", path, name="report")
    assert encoded.preview == {"filename": "report final.pdf", "mime": "application/pdf"}
    assert encode("file", b"?", name="blob").mime_type == "application/octet-stream"
    with pytest.raises(TypeError):
        encode("file", {"not": "bytes"}, name="x")


def test_unknown_kinds_and_none() -> None:
    with pytest.raises(TypeError):
        encode("video", b"x", name="x")
    with pytest.raises(TypeError, match="must not be None"):
        encode("json", None, name="x")


def test_extensions_are_safe() -> None:
    assert extension_for("application/pdf") == "pdf"
    assert extension_for("text/html; charset=utf-8") == "html"
    assert extension_for("application/x-unknown-thing") == "bin"
    assert extension_for("text/plain", "../../evil.sh/x") == "txt"  # suffix-less → by mime
    assert extension_for("text/plain", "notes.TXT") == "txt"
