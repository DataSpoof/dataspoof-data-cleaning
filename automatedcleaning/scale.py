"""Scaling out: transform files of any size, score on Spark / Ray / Dask, and push cleaning
pipelines down into the warehouse as SQL.

* ``transform_file(pipeline, "big.csv", "clean.parquet")``  a fitted pipeline over a file in
  bounded-memory chunks (valid because transform only replays learned, row-wise rules).
* ``score_spark(model_path, spark_df)`` / ``score_ray(model_path, dataset)`` /
  ``score_dask(model_path, ddf)``  distributed scoring: each worker loads the saved model once
  (signature-verified if ``AC_MODEL_PUBLIC_KEY`` is set) and scores its partitions.
* ``to_sql(pipeline, "raw.orders")``  compile a fitted cleaning pipeline into one ``SELECT`` so
  the warehouse does the work — no data leaves it. Steps without an exact SQL equivalent raise
  ``NotImplementedError`` instead of silently producing different results.
"""

from __future__ import annotations

import functools
import math
import time
from pathlib import Path

import polars as pl

from .ingestion._util import quote_ident


# ── files of any size ───────────────────────────────────────────────────────

def transform_file(pipeline, source, output, *, chunk_size: int = 100_000) -> dict:
    """Apply a *fitted* pipeline to ``source`` (csv/tsv/parquet/ndjson) chunk by chunk and write
    ``output`` (.parquet or .csv). Steps that only run during fit (row filters) are skipped,
    exactly as in ``pipeline.transform``."""
    from .deploy.scoring import _batches
    source, output = Path(source), Path(output)
    if output.suffix.lower() not in (".parquet", ".csv"):
        raise ValueError("output must be a .parquet or .csv file")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".partial")
    tmp.unlink(missing_ok=True)
    writer, rows, chunks, t0 = None, 0, 0, time.perf_counter()
    try:
        for chunk in _batches(source, chunk_size):
            out = pipeline.transform(chunk)
            if output.suffix.lower() == ".parquet":
                import pyarrow.parquet as pq
                table = out.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema)
                writer.write_table(table.cast(writer.schema))
            else:
                with open(tmp, "ab") as fh:
                    out.write_csv(fh, include_header=chunks == 0)
            rows += chunk.height
            chunks += 1
    finally:
        if writer is not None:
            writer.close()
    if chunks == 0:
        raise ValueError(f"{source} has no rows")
    tmp.replace(output)
    return {"rows": rows, "chunks": chunks, "seconds": round(time.perf_counter() - t0, 3), "output": str(output)}


# ── distributed scoring ─────────────────────────────────────────────────────

@functools.lru_cache(maxsize=4)
def _cached_model(path: str):
    from .automl import load_model
    model = load_model(path)
    return model


def partition_scorer(model_path, *, keep=None, proba: bool = True):
    """``f(iterator of pandas DataFrames) -> iterator of pandas DataFrames`` that scores each
    partition with the saved model (loaded once per worker process). This is the function Spark's
    ``mapInPandas``, Ray's ``map_batches`` and Dask's ``map_partitions`` run."""
    path = str(model_path)

    def score_partition(batches):
        from .deploy.scoring import score
        model = _cached_model(path)
        for pdf in batches:
            frame = pl.from_pandas(pdf)
            yield score(model, frame, keep=keep, proba=proba).to_pandas()
    return score_partition


def _sample_output(model_path, sample_pdf, keep, proba):
    return next(partition_scorer(model_path, keep=keep, proba=proba)(iter([sample_pdf])))


def score_spark(model_path, sdf, *, keep=None, proba: bool = True):
    """Score a Spark DataFrame. The output schema is inferred by scoring a small sample."""
    sample = sdf.limit(100).toPandas()
    schema = sdf.sparkSession.createDataFrame(_sample_output(model_path, sample, keep, proba)).schema
    return sdf.mapInPandas(partition_scorer(model_path, keep=keep, proba=proba), schema=schema)


def score_ray(model_path, dataset, *, keep=None, proba: bool = True, batch_size: int = 4096):
    """Score a Ray Dataset (``ray.data``)."""
    fn = partition_scorer(model_path, keep=keep, proba=proba)
    return dataset.map_batches(lambda pdf: next(fn(iter([pdf]))), batch_format="pandas", batch_size=batch_size)


def score_dask(model_path, ddf, *, keep=None, proba: bool = True):
    """Score a Dask DataFrame (output columns inferred from the first partition's head)."""
    fn = partition_scorer(model_path, keep=keep, proba=proba)
    meta = _sample_output(model_path, ddf.head(100), keep, proba).iloc[:0]
    return ddf.map_partitions(lambda pdf: next(fn(iter([pdf]))), meta=meta)


# ── SQL pushdown ────────────────────────────────────────────────────────────

_FLOAT = {"sqlite": "REAL", "mysql": "DOUBLE", "mariadb": "DOUBLE", "bigquery": "FLOAT64", "snowflake": "DOUBLE",
          "mssql": "FLOAT"}
_TRUE = {"mssql": "1"}
_FALSE = {"mssql": "0"}


def _lit(v, dialect: str) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return _TRUE.get(dialect, "TRUE") if v else _FALSE.get(dialect, "FALSE")
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            raise NotImplementedError("NaN / infinite constants have no portable SQL form")
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


def _num_expr(e: str, dialect: str) -> str:
    return f"CAST({e} AS {_FLOAT.get(dialect, 'DOUBLE PRECISION')})"


def to_sql(pipeline, table: str, *, columns=None, schema: dict | None = None, dialect: str = "ansi") -> str:
    """Compile a fitted pipeline into ``SELECT ... FROM table`` (``dialect``: ansi, postgresql,
    sqlite, mysql, mssql, snowflake, bigquery). ``columns``/``schema`` default to the input
    schema the pipeline was fitted on.

    Supported steps: drop_columns, drop_collinear, convert_numeric_text, impute, clip_outliers,
    log_transform_skewed, scale, group_rare_categories, encode_categorical, and the fit-only
    row filters (drop_duplicates, remove_outliers), which don't apply at transform time."""
    q = {"mysql": "`", "mariadb": "`", "bigquery": "`", "mssql": "["}.get(dialect, '"')
    schema = dict(schema or getattr(pipeline, "input_schema_", None) or {})
    if columns is None:
        if not schema:
            raise ValueError("Pass columns=[...]: the pipeline has no recorded input schema")
        columns = list(schema)
    target = getattr(pipeline, "target", None)
    exprs: dict = {c: quote_ident(c, q) for c in columns if c != target}
    is_int = {c: str(schema.get(c, "")).startswith(("Int", "UInt")) for c in exprs}
    for step in pipeline.steps:
        if not step.is_fitted:
            raise ValueError("to_sql needs a fitted pipeline")
        st, name, p = step.state_, step.name, step.params
        if name in ("drop_duplicates", "remove_outliers"):
            continue                                     # fit-only row filters
        if name in ("drop_columns", "drop_collinear"):
            for c in st["drop"]:
                exprs.pop(c, None)
        elif name == "convert_numeric_text":
            for c in st["columns"]:
                if c in exprs:
                    raw = f"TRIM({exprs[c]})"
                    stripped = raw
                    for ch in ["$", "₹", "€", "£", "¥", ",", "%", "(", ")", " "]:
                        stripped = f"REPLACE({stripped}, {_lit(ch, dialect)}, '')"
                    safe = {"snowflake": "TRY_CAST", "bigquery": "SAFE_CAST", "mssql": "TRY_CAST",
                            "duckdb": "TRY_CAST"}.get(dialect)
                    if safe is None:     # plain CAST errors (Postgres) or returns 0 (SQLite) on bad text
                        raise NotImplementedError(f"convert_numeric_text needs a safe cast; dialect '{dialect}' "
                                                  "has none (supported: snowflake, bigquery, mssql, duckdb)")
                    num = f"{safe}(NULLIF({stripped}, '') AS {_FLOAT.get(dialect, 'DOUBLE')})"
                    exprs[c] = f"(CASE WHEN {raw} LIKE '(%)' THEN -{num} ELSE {num} END)"
                    is_int[c] = False
        elif name == "impute":
            for c in st["indicators"]:
                if c in exprs:
                    exprs[f"{c}_was_missing"] = (f"(CASE WHEN {exprs[c]} IS NULL THEN {_lit(True, dialect)} "
                                                 f"ELSE {_lit(False, dialect)} END)")
            for c, v in st["fill"].items():
                if c in exprs:
                    exprs[c] = f"COALESCE({exprs[c]}, {_lit(v, dialect)})"
        elif name == "clip_outliers":
            for c, (lo, hi) in st["bounds"].items():
                if c not in exprs:
                    continue
                if is_int.get(c):
                    lo, hi = math.ceil(lo), math.floor(hi)
                    if lo > hi:
                        continue
                e = exprs[c]
                exprs[c] = (f"(CASE WHEN {e} < {_lit(lo, dialect)} THEN {_lit(lo, dialect)} "
                            f"WHEN {e} > {_lit(hi, dialect)} THEN {_lit(hi, dialect)} ELSE {e} END)")
        elif name == "log_transform_skewed":
            for c in st["columns"]:
                if c in exprs:
                    e = _num_expr(exprs[c], dialect)
                    exprs[c] = f"LN(1 + (CASE WHEN {e} < 0 THEN 0 ELSE {e} END))"
                    is_int[c] = False
        elif name == "scale":
            for c, (center, scale) in st["params"].items():
                if c in exprs:
                    exprs[c] = f"(({_num_expr(exprs[c], dialect)} - {_lit(float(center), dialect)}) / {_lit(float(scale), dialect)})"
                    is_int[c] = False
        elif name == "group_rare_categories":
            other = p["other_label"]
            for c, kept in st["keep"].items():
                if c in exprs:
                    e = exprs[c]
                    known = f"{e} IN ({', '.join(_lit(k, dialect) for k in kept)})" if kept else "1 = 0"
                    exprs[c] = f"(CASE WHEN {e} IS NULL THEN NULL WHEN {known} THEN {e} ELSE {_lit(other, dialect)} END)"
        elif name == "encode_categorical":
            method = p["method"]
            for c, learned in st["categories"].items():
                if c not in exprs:
                    continue
                e = exprs[c]
                if method == "onehot":
                    for cat in learned:
                        exprs[f"{c}_{cat}"] = f"(CASE WHEN {e} = {_lit(cat, dialect)} THEN 1 ELSE 0 END)"
                    if p["drop_original"]:
                        exprs.pop(c)
                else:
                    default = -1 if method == "ordinal" else 0.0
                    cases = " ".join(f"WHEN {_lit(k, dialect)} THEN {_lit(v, dialect)}" for k, v in learned.items())
                    exprs[c] = f"(CASE WHEN {e} IS NULL THEN NULL ELSE (CASE {e} {cases} ELSE {_lit(default, dialect)} END) END)"
        else:
            raise NotImplementedError(f"Step '{name}' has no exact SQL equivalent; run it in Python "
                                      "(e.g. transform_file) or split the pipeline.")
    if not exprs:
        raise ValueError("The pipeline drops every column")
    select = ",\n  ".join(f"{e} AS {quote_ident(c, q)}" for c, e in exprs.items())
    return f"SELECT\n  {select}\nFROM {quote_ident(table, q)}"
