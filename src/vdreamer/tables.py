"""Parquet table artifacts: writing, previews, paging, and export (PRD §3.4, A24).

Table artifacts are stored as Parquet (A24), written incrementally in row
groups so a large row iterator never has to fit in memory, and capped at
:data:`TABLE_MAX_ROWS` (5M; ``VDREAMER_TABLE_MAX_ROWS`` overrides) — beyond
it the table is truncated and ``metadata.truncated`` says so.

The read side serves the gateway: :func:`page` for
``/preview?offset&limit&sort`` and :func:`iter_csv` / :func:`write_xlsx` for
``/download?format=csv|xlsx``. XLSX holds at most 1,048,575 data rows below
its header (:data:`XLSX_MAX_ROWS`); a larger table is the caller's ``422``.

pyarrow comes with the ``[sandbox]`` extra (and XLSX export with its
``xlsxwriter``); without it, table artifacts raise :class:`TablesUnavailable`.
"""

from __future__ import annotations

import base64
import contextlib
import datetime
import decimal
import io
import math
import os
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

TABLE_MAX_ROWS_ENV: Final = "VDREAMER_TABLE_MAX_ROWS"
#: §3.4: tables beyond 5M rows are truncated (``metadata.truncated``).
DEFAULT_TABLE_MAX_ROWS: Final = 5_000_000
#: Rows per Parquet row group (and per streamed export batch).
ROW_GROUP_ROWS: Final = 65_536
#: The SSE preview carries the schema, this many rows, and ``row_count``.
PREVIEW_ROWS: Final = 10
#: An XLSX sheet holds 1,048,576 rows: the header plus this many.
XLSX_MAX_ROWS: Final = 1_048_575
PARQUET_MIME: Final = "application/vnd.apache.parquet"
XLSX_MIME: Final = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class TablesUnavailable(RuntimeError):
    """pyarrow (the ``[sandbox]`` extra) is not installed."""


class SortError(ValueError):
    """A ``sort`` parameter naming an unknown column."""


class CorruptTable(ValueError):
    """The stored bytes are not readable Parquet."""


def _pa() -> Any:
    try:
        import pyarrow
    except ImportError as exc:
        raise TablesUnavailable(
            "table artifacts need pyarrow: pip install 'vdreamer[sandbox]'"
        ) from exc
    return pyarrow


def table_max_rows() -> int:
    raw = os.environ.get(TABLE_MAX_ROWS_ENV, "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return DEFAULT_TABLE_MAX_ROWS


def json_safe(value: Any) -> Any:
    """A cell (or row) as JSON: ISO dates, str decimals, base64 bytes, NaN → ``None``."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def _is_dataframe(data: Any) -> bool:
    kind = type(data)
    return kind.__name__ == "DataFrame" and kind.__module__.split(".", 1)[0] == "pandas"


@dataclass(frozen=True)
class WrittenTable:
    """What :func:`write_parquet` wrote: the preview block and the counts."""

    row_count: int
    truncated: bool
    preview: dict[str, Any]


def schema_of(schema: Any) -> list[dict[str, str]]:
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def _batches(data: Any, pa: Any) -> Iterator[Any]:
    """``data`` as pyarrow tables of ≤ :data:`ROW_GROUP_ROWS` rows (``TypeError`` otherwise)."""
    if isinstance(data, pa.Table):
        yield from (pa.Table.from_batches([b]) for b in data.to_batches(ROW_GROUP_ROWS))
        if data.num_rows == 0:
            yield data
        return
    if isinstance(data, pa.RecordBatch):
        yield from _batches(pa.Table.from_batches([data]), pa)
        return
    if _is_dataframe(data):
        yield from _batches(pa.Table.from_pandas(data), pa)
        return
    if isinstance(data, (str, bytes, bytearray, Mapping)) or not isinstance(data, Iterable):
        raise TypeError(
            "table data must be a pyarrow.Table, a pandas.DataFrame, or an iterable of "
            f"row mappings, got {type(data).__name__}"
        )
    schema = None
    chunk: list[Mapping[str, Any]] = []
    for row in data:
        if not isinstance(row, Mapping):
            raise TypeError(f"table rows must be mappings (dicts), got {type(row).__name__}")
        chunk.append(row)
        if len(chunk) >= ROW_GROUP_ROWS:
            table = pa.Table.from_pylist(chunk, schema=schema)
            schema = table.schema
            yield table
            chunk = []
    if chunk or schema is None:
        yield pa.Table.from_pylist(chunk, schema=schema)


def write_parquet(data: Any, dest: str | Path, *, max_rows: int | None = None) -> WrittenTable:
    """Write ``data`` to ``dest`` as Parquet, row group by row group (§3.4)."""
    pa = _pa()
    import pyarrow.parquet as pq

    cap = table_max_rows() if max_rows is None else max_rows
    writer = None
    rows = 0
    truncated = False
    head: list[dict[str, Any]] = []
    schema = None
    try:
        for batch in _batches(data, pa):
            if schema is not None and not batch.schema.equals(schema):
                try:
                    batch = batch.cast(schema)
                except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
                    raise TypeError(f"table rows do not share one schema: {exc}") from exc
            if rows + batch.num_rows > cap:
                batch = batch.slice(0, cap - rows)
                truncated = True
            if writer is None:
                schema = batch.schema
                writer = pq.ParquetWriter(str(dest), schema)
            if batch.num_rows or rows == 0:
                writer.write_table(batch, row_group_size=ROW_GROUP_ROWS)
            if len(head) < PREVIEW_ROWS:
                head.extend(batch.slice(0, PREVIEW_ROWS - len(head)).to_pylist())
            rows += batch.num_rows
            if truncated:
                break
    finally:
        if writer is not None:
            writer.close()
    assert schema is not None  # _batches always yields at least one table
    preview = {
        "schema": schema_of(schema),
        "rows": [json_safe(row) for row in head],
        "row_count": rows,
    }
    return WrittenTable(row_count=rows, truncated=truncated, preview=preview)


# ---------------------------------------------------------------------------
# Read side (the gateway)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _parquet(path: str | Path) -> Iterator[Any]:
    """An open ``ParquetFile`` whose handle closes with the block (Windows holds files)."""
    pa = _pa()
    import pyarrow.parquet as pq

    with open(path, "rb") as handle:
        try:
            pf = pq.ParquetFile(handle)
        except (pa.ArrowInvalid, OSError) as exc:
            raise CorruptTable(f"the stored table is not readable Parquet: {exc}") from exc
        try:
            yield pf
        finally:
            pf.close()


def row_count(path: str | Path) -> int:
    with _parquet(path) as pf:
        return int(pf.metadata.num_rows)


def parse_sort(sort: str | None, columns: Sequence[str]) -> list[tuple[str, str]]:
    """``"a,-b"`` → ``[("a", "ascending"), ("b", "descending")]``; unknown columns raise."""
    if not sort:
        return []
    keys: list[tuple[str, str]] = []
    for raw in sort.split(","):
        item = raw.strip()
        if not item:
            continue
        order = "descending" if item.startswith("-") else "ascending"
        name = item.lstrip("+-")
        if name not in columns:
            raise SortError(f"unknown sort column {name!r}")
        keys.append((name, order))
    return keys


def page(
    path: str | Path, *, offset: int, limit: int, sort: str | None = None
) -> dict[str, Any]:
    """One ``/preview`` page: ``{schema, rows, row_count, offset, limit, sort}``.

    Unsorted pages read only the row groups they span; a sorted page sorts
    the whole table first (nulls last).
    """
    with _parquet(path) as pf:
        return _page(pf, offset=offset, limit=limit, sort=sort)


def _page(pf: Any, *, offset: int, limit: int, sort: str | None) -> dict[str, Any]:
    schema = pf.schema_arrow
    total = int(pf.metadata.num_rows)
    keys = parse_sort(sort, schema.names)
    rows: list[dict[str, Any]] = []
    if keys:
        import pyarrow.compute as pc

        table = pf.read()
        indices = pc.sort_indices(table, sort_keys=[(n, o, "at_end") for n, o in keys])
        rows = table.take(indices.slice(offset, limit)).to_pylist()
    else:
        start = 0
        first_start = 0
        groups: list[int] = []
        for index in range(pf.num_row_groups):
            n = pf.metadata.row_group(index).num_rows
            if start + n > offset and start < offset + limit:
                if not groups:
                    first_start = start
                groups.append(index)
            start += n
        if groups:
            table = pf.read_row_groups(groups)
            rows = table.slice(offset - first_start, limit).to_pylist()
    return {
        "schema": schema_of(schema),
        "rows": [json_safe(row) for row in rows],
        "row_count": total,
        "offset": offset,
        "limit": limit,
        "sort": sort,
    }


def iter_csv(path: str | Path) -> Iterator[bytes]:
    """The table as CSV (UTF-8, header first), one row group's batch at a time."""
    _pa()
    import pyarrow.csv as pacsv

    with _parquet(path) as pf:
        header = True
        for batch in pf.iter_batches(batch_size=ROW_GROUP_ROWS):
            buffer = io.BytesIO()
            options = pacsv.WriteOptions(include_header=header)
            pacsv.write_csv(batch, buffer, write_options=options)
            header = False
            yield buffer.getvalue()
        if header:  # an empty table still gets its header row
            buffer = io.BytesIO()
            pacsv.write_csv(pf.schema_arrow.empty_table(), buffer)
            yield buffer.getvalue()


def _xlsx_cell(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime.datetime):
        return value.replace(tzinfo=None).isoformat(sep=" ")
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    safe = json_safe(value)
    return safe if isinstance(safe, str) else str(safe)


def write_xlsx(path: str | Path, dest: str | Path) -> None:
    """The table as one XLSX sheet at ``dest`` (xlsxwriter, constant memory).

    Raises:
        ValueError: more than :data:`XLSX_MAX_ROWS` rows (the route's ``422``).
    """
    _pa()
    try:
        import xlsxwriter
    except ImportError as exc:  # pragma: no cover - part of the [sandbox] extra
        raise TablesUnavailable(
            "XLSX export needs xlsxwriter: pip install 'vdreamer[sandbox]'"
        ) from exc
    with _parquet(path) as pf:
        _write_xlsx(pf, dest, xlsxwriter)


def _write_xlsx(pf: Any, dest: str | Path, xlsxwriter: Any) -> None:
    total = int(pf.metadata.num_rows)
    if total > XLSX_MAX_ROWS:
        raise ValueError(f"{total} rows exceed the XLSX limit of {XLSX_MAX_ROWS}")
    workbook = xlsxwriter.Workbook(
        str(dest), {"constant_memory": True, "strings_to_formulas": False, "strings_to_urls": False}
    )
    try:
        sheet = workbook.add_worksheet("data")
        for col, name in enumerate(pf.schema_arrow.names):
            sheet.write_string(0, col, name)
        row_index = 1
        for batch in pf.iter_batches(batch_size=ROW_GROUP_ROWS):
            for row in batch.to_pylist():
                for col, value in enumerate(row.values()):
                    cell = _xlsx_cell(value)
                    if cell is None:
                        continue
                    if isinstance(cell, str):
                        sheet.write_string(row_index, col, cell)
                    else:
                        sheet.write(row_index, col, cell)
                row_index += 1
    finally:
        workbook.close()
