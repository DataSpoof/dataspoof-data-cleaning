"""Files on local disk, HTTP(S), and cloud object storage (S3, GCS, Azure) in any common format.

Cloud paths (``s3://``, ``gs://``, ``az://``, ``abfss://`` …) are read by Polars' native
object-store client — no extra packages — with credentials from ``storage_options`` or the
standard environment variables (``AWS_ACCESS_KEY_ID``/``AWS_PROFILE``,
``GOOGLE_APPLICATION_CREDENTIALS``, ``AZURE_STORAGE_ACCOUNT_NAME``/``…_KEY`` …). Globs such as
``s3://bucket/sales/2024-*.parquet`` read many files as one table.
"""

from __future__ import annotations

import glob as _glob
import io
import urllib.request
from pathlib import Path

import polars as pl

from ._util import redact

FORMATS = {".csv": "csv", ".tsv": "tsv", ".txt": "csv", ".parquet": "parquet", ".pq": "parquet",
           ".json": "json", ".ndjson": "ndjson", ".jsonl": "ndjson", ".xlsx": "excel", ".xlsm": "excel",
           ".xls": "excel", ".xlsb": "excel", ".ods": "ods", ".arrow": "ipc", ".feather": "ipc", ".ipc": "ipc",
           ".avro": "avro"}
COMPRESSION = (".gz", ".gzip", ".zst", ".zstd", ".bz2", ".zip")
CLOUD_SCHEMES = ("s3://", "s3a://", "gs://", "gcs://", "az://", "azure://", "abfs://", "abfss://", "adl://", "hf://")
SCANNABLE = ("csv", "tsv", "parquet", "ndjson", "ipc")
MAX_HTTP_BYTES = 2 * 1024 ** 3


def is_cloud(path) -> bool:
    return str(path).lower().startswith(CLOUD_SCHEMES)


def is_http(path) -> bool:
    return str(path).lower().startswith(("http://", "https://"))


def detect_format(path) -> str | None:
    name = str(path).split("?", 1)[0].split("#", 1)[0].lower()
    for ext in COMPRESSION:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return FORMATS.get(Path(name).suffix)


def _http_bytes(url: str, headers: dict | None = None, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "automatedcleaning-ingest", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:     # noqa: S310 - http(s) only (checked)
        data = resp.read(MAX_HTTP_BYTES + 1)
    if len(data) > MAX_HTTP_BYTES:
        raise ValueError(f"{redact(url)} is larger than {MAX_HTTP_BYTES} bytes; download it first.")
    return data


def _eager(fmt: str, src, kw: dict) -> pl.DataFrame:
    if fmt in ("csv", "tsv"):
        return pl.read_csv(src, **({"separator": "\t"} if fmt == "tsv" else {}), **kw)
    if fmt == "parquet":
        return pl.read_parquet(src, **kw)
    if fmt == "ndjson":
        return pl.read_ndjson(src, **kw)
    if fmt == "json":
        return pl.read_json(src, **kw)
    if fmt == "ipc":
        return pl.read_ipc(src, **kw)
    if fmt == "avro":
        return pl.read_avro(src, **kw)
    if fmt == "excel":
        try:
            return pl.read_excel(src, **kw)
        except ModuleNotFoundError as exc:
            raise ImportError("Reading Excel needs fastexcel: pip install \"AutomatedCleaning[excel]\"") from exc
    if fmt == "ods":
        return pl.read_ods(src, **kw)
    raise ValueError(f"Unsupported format {fmt!r}")


def _scan(fmt: str, path: str, storage_options, kw: dict) -> pl.LazyFrame:
    opts = {"storage_options": storage_options} if storage_options else {}
    if fmt in ("csv", "tsv"):
        return pl.scan_csv(path, **({"separator": "\t"} if fmt == "tsv" else {}), **opts, **kw)
    if fmt == "parquet":
        return pl.scan_parquet(path, **opts, **kw)
    if fmt == "ndjson":
        return pl.scan_ndjson(path, **opts, **kw)
    return pl.scan_ipc(path, **opts, **kw)


def read_file(path, *, format: str | None = None, columns=None, n_rows: int | None = None, storage_options=None,
              lazy: bool = False, include_source: bool = False, headers: dict | None = None, **kw):
    """Read a file / glob / URL / cloud object into a Polars DataFrame (``lazy=True``: LazyFrame).

    ``format`` is detected from the extension (csv, tsv, parquet, json, ndjson, excel, ods,
    arrow/feather, avro; ``.gz``/``.zst`` compressed CSV). ``columns`` and ``n_rows`` are pushed
    down to the reader where possible. ``include_source=True`` adds a ``_source_file`` column.
    Extra keyword arguments go to the underlying Polars reader (e.g. ``sheet_name=``)."""
    path = str(path)
    fmt = format or detect_format(path)
    if fmt is None:
        raise ValueError(f"Can't tell the format of {redact(path)}; pass format='csv' / 'parquet' / ...")
    compressed = path.split("?", 1)[0].lower().endswith(COMPRESSION)
    if include_source and fmt in SCANNABLE and not compressed:
        kw = {**kw, "include_file_paths": "_source_file"}

    if is_http(path):
        frame = _eager(fmt, io.BytesIO(_http_bytes(path, headers)), kw)
        if include_source:
            frame = frame.with_columns(pl.lit(redact(path)).alias("_source_file"))
    elif fmt in SCANNABLE and not compressed:
        lf = _scan(fmt, path, storage_options, kw)
        if columns:
            lf = lf.select([*columns, *(["_source_file"] if include_source else [])])
        if n_rows is not None:
            lf = lf.head(n_rows)
        return lf if lazy else lf.collect()
    else:
        if is_cloud(path):
            raise ValueError(f"{fmt} files can't be read straight from cloud storage; use parquet/csv/ndjson, "
                             "or download the file first.")
        matches = sorted(_glob.glob(path)) if any(ch in path for ch in "*?[") else [path]
        if not matches:
            raise FileNotFoundError(path)
        frames = []
        for p in matches:
            f = _eager(fmt, p, kw)
            frames.append(f.with_columns(pl.lit(p).alias("_source_file")) if include_source else f)
        frame = frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
    if columns:
        frame = frame.select([*columns, *(["_source_file"] if include_source else [])])
    if n_rows is not None:
        frame = frame.head(n_rows)
    return frame.lazy() if lazy else frame


def list_files(pattern: str) -> list:
    """Local files matching a glob, sorted (used by incremental file loads)."""
    if is_cloud(pattern) or is_http(pattern):
        raise ValueError("Incremental file loads list local/mounted folders; for buckets, mount them "
                         "(e.g. s3fs/gcsfuse) or load by date-partitioned globs.")
    return sorted(p for p in _glob.glob(pattern, recursive=True) if Path(p).is_file())
