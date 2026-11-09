"""Scoring: predictions for a frame, and batch jobs that stream files of any size."""

from __future__ import annotations

from typing import Any

import time
from pathlib import Path

import numpy as np
import polars as pl


def resolve_model(model):
    """A fitted model object, or a path to a saved model directory / registry spec."""
    if isinstance(model, (str, Path)):
        from ..automl import load_model
        return load_model(model)
    return model


def expected_columns(model) -> list:
    """Raw input columns the model was trained on (before its own preprocessing)."""
    prep = getattr(model, "preprocess_", None)
    schema = getattr(prep, "input_schema_", None) if prep is not None else None
    cols = list(schema) if schema else list(getattr(model, "feature_kinds_", {}) or {})
    target = getattr(model, "target", None)
    leaky = set(getattr(model, "dropped_leaky_", []) or [])
    return [c for c in cols if c != target and c not in leaky]


def model_inputs(model, df: pl.DataFrame) -> pl.DataFrame:
    """Only the columns the model was trained on (extra ids/metadata columns are normal when
    scoring and shouldn't trigger schema warnings); missing ones are left to the model."""
    expected = set(expected_columns(model))
    if not expected:
        return df
    df = df.select([c for c in df.columns if c in expected])
    # JSON / CSV can't tell 22 from 22.0: give numeric columns their training dtype back.
    prep = getattr(model, "preprocess_", None)
    schema = getattr(prep, "input_schema_", None) or {}
    if schema:
        from ..core.pipeline import _parse_dtype
        casts = []
        for c, t in df.schema.items():
            want = _parse_dtype(schema[c]) if c in schema else None
            if want is not None and t != want and (t.is_numeric() or t == pl.Null) and want.is_numeric():
                casts.append(pl.col(c).cast(want, strict=False))
        if casts:
            df = df.with_columns(casts)
    return df


def score(model, df, *, proba: bool = True, keep: list | None = None, prefix: str = "") -> pl.DataFrame:
    """Predictions for every row of ``df``.

    Returns the ``keep`` columns of ``df`` (default: none) followed by ``prediction`` and, for
    classifiers, ``proba_<class>`` columns (clustering models return ``cluster``)."""
    model = resolve_model(model)
    df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    out = df.select(keep) if keep else pl.DataFrame()
    df = model_inputs(model, df)
    from ..automl import AutoCluster
    if isinstance(model, AutoCluster):
        cols = [pl.Series(f"{prefix}cluster", model.predict(df).to_numpy())]
    else:
        pred, p = model.predict_with_proba(df)
        cols = [pred.alias(f"{prefix}prediction")]
        if proba and p is not None:
            cols += [p[c].alias(f"{prefix}{c}") for c in p.columns]
    return out.hstack(cols) if out.width else pl.DataFrame(cols)


def _batches(path: Path, chunk_size: int):
    """Yield Polars frames of ~chunk_size rows from CSV / Parquet / NDJSON without loading it all."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_size):
            yield pl.from_arrow(batch)
    elif suffix in (".csv", ".txt", ".tsv"):
        import pyarrow.csv as pacsv
        parse = pacsv.ParseOptions(delimiter="\t" if suffix == ".tsv" else ",")
        reader = pacsv.open_csv(path, parse_options=parse,
                                read_options=pacsv.ReadOptions(block_size=1 << 22))
        pending, size = [], 0
        for batch in reader:
            pending.append(batch)
            size += batch.num_rows
            while size >= chunk_size:          # emit exact chunk_size slices; keep the remainder
                table = _table(pending)
                yield pl.from_arrow(table.slice(0, chunk_size))
                rest = table.slice(chunk_size)
                pending, size = (rest.to_batches() if rest.num_rows else []), rest.num_rows
        if size:
            yield pl.from_arrow(_table(pending))
    elif suffix in (".ndjson", ".jsonl"):
        frame = pl.read_ndjson(path)
        for start in range(0, frame.height, chunk_size):
            yield frame.slice(start, chunk_size)
    else:
        raise ValueError(f"Unsupported input format {suffix!r}: use .csv, .tsv, .parquet or .ndjson")


def _table(batches):
    import pyarrow as pa
    return pa.Table.from_batches(batches)


def batch_score(model, source, output, *, chunk_size: int = 100_000, keep: list | str | None = "all",
                proba: bool = True, log_every: int | None = None) -> dict:
    """Score a large file chunk by chunk and write the results (``.parquet`` or ``.csv``).

    ``keep``: ``"all"`` input columns (default), a list of id columns, or ``None``.
    Memory use is bounded by ``chunk_size``. Returns a summary (rows, seconds, rows/second,
    prediction distribution, columns the model expected but the file lacked).
    """
    model = resolve_model(model)
    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_kind = output.suffix.lower()
    if out_kind not in (".parquet", ".csv"):
        raise ValueError("output must be a .parquet or .csv file")
    tmp = output.with_name(output.name + ".partial")
    tmp.unlink(missing_ok=True)
    t0, rows, chunks = time.perf_counter(), 0, 0
    writer: Any = None
    missing: list | None = None
    dist: dict = {}
    try:
        for chunk in _batches(source, chunk_size):
            if missing is None:
                missing = [c for c in expected_columns(model) if c not in chunk.columns]
            cols = chunk.columns if keep == "all" else (keep if isinstance(keep, list) else [])
            scored = score(model, chunk, proba=proba, keep=cols)
            if out_kind == ".parquet":
                import pyarrow.parquet as pq
                table = scored.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema)
                writer.write_table(table.cast(writer.schema))
            else:
                with open(tmp, "ab") as fh:
                    scored.write_csv(fh, include_header=chunks == 0)
            label = "cluster" if "cluster" in scored.columns else "prediction"
            if scored[label].dtype.is_numeric() and scored[label].n_unique() > 50:
                dist.setdefault("_sum", 0.0)
                dist["_sum"] += float(scored[label].sum())
            else:
                for v, n in scored[label].value_counts().iter_rows():
                    dist[str(v)] = dist.get(str(v), 0) + n
            rows += chunk.height
            chunks += 1
            if log_every and chunks % log_every == 0:
                print(f"  scored {rows:,} rows ({rows / (time.perf_counter() - t0):,.0f}/s)", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if chunks == 0:
        raise ValueError(f"{source} has no rows to score")
    tmp.replace(output)                      # only a complete file appears at the output path
    seconds = time.perf_counter() - t0
    summary = {"input": str(source), "output": str(output), "rows": rows, "chunks": chunks,
               "seconds": round(seconds, 3), "rows_per_second": round(rows / seconds, 1) if seconds else None,
               "missing_columns": missing or []}
    if "_sum" in dist:
        summary["mean_prediction"] = dist["_sum"] / rows
    else:
        summary["prediction_counts"] = dist
    return summary


def to_records(df: pl.DataFrame) -> list:
    """JSON-safe list of row dicts (NaN → None)."""
    out = []
    for row in df.iter_rows(named=True):
        out.append({k: (None if isinstance(v, float) and v != v else
                        (v.item() if isinstance(v, np.generic) else v)) for k, v in row.items()})
    return out
