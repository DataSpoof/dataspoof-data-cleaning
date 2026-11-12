# Scale-out & SQL pushdown: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Scale-out & SQL pushdown guide](../guide/integrations.md).

**On this page:**

- [`mlpilot.scale`](#mlpilotscale): `transform_file`, `partition_scorer`, `score_spark`, `score_ray`, `score_dask`, `to_sql`

---

## `mlpilot.scale`

<sub>Source: [`mlpilot/scale.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/scale.py)</sub>

Scaling out: transform files of any size, score on Spark / Ray / Dask, and push cleaning
pipelines down into the warehouse as SQL.

* `transform_file(pipeline, "big.csv", "clean.parquet")`  a fitted pipeline over a file in
  bounded-memory chunks (valid because transform only replays learned, row-wise rules).
* `score_spark(model_path, spark_df)` / `score_ray(model_path, dataset)` /
  `score_dask(model_path, ddf)`  distributed scoring: each worker loads the saved model once
  (signature-verified if `AC_MODEL_PUBLIC_KEY` is set) and scores its partitions.
* `to_sql(pipeline, "raw.orders")`  compile a fitted cleaning pipeline into one `SELECT` so
  the warehouse does the work — no data leaves it. Steps without an exact SQL equivalent raise
  `NotImplementedError` instead of silently producing different results.

### `transform_file()`

```python
def transform_file(pipeline, source, output, *, chunk_size: int=100000) -> dict
```

Apply a *fitted* pipeline to `source` (csv/tsv/parquet/ndjson) chunk by chunk and write
`output` (.parquet or .csv). Steps that only run during fit (row filters) are skipped,
exactly as in `pipeline.transform`.

### `partition_scorer()`

```python
def partition_scorer(model_path, *, keep=None, proba: bool=True)
```

`f(iterator of pandas DataFrames) -> iterator of pandas DataFrames` that scores each
partition with the saved model (loaded once per worker process). This is the function Spark's
`mapInPandas`, Ray's `map_batches` and Dask's `map_partitions` run.

### `score_spark()`

```python
def score_spark(model_path, sdf, *, keep=None, proba: bool=True)
```

Score a Spark DataFrame. The output schema is inferred by scoring a small sample.

### `score_ray()`

```python
def score_ray(model_path, dataset, *, keep=None, proba: bool=True, batch_size: int=4096)
```

Score a Ray Dataset (`ray.data`).

### `score_dask()`

```python
def score_dask(model_path, ddf, *, keep=None, proba: bool=True)
```

Score a Dask DataFrame (output columns inferred from the first partition's head).

### `to_sql()`

```python
def to_sql(pipeline, table: str, *, columns=None, schema: dict | None=None, dialect: str='ansi') -> str
```

Compile a fitted pipeline into `SELECT ... FROM table` (`dialect`: ansi, postgresql,
sqlite, mysql, mssql, snowflake, bigquery). `columns`/`schema` default to the input
schema the pipeline was fitted on.

Supported steps: drop_columns, drop_collinear, convert_numeric_text, impute, clip_outliers,
log_transform_skewed, scale, group_rare_categories, encode_categorical, and the fit-only
row filters (drop_duplicates, remove_outliers), which don't apply at transform time.
