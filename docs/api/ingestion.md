# Ingestion & connectors: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Ingestion & connectors guide](../guide/ingestion.md).

**On this page:**

- [`mlpilot.ingestion`](#mlpilotingestion)
- [`mlpilot.ingestion.api`](#mlpilotingestionapi): `APIError`, `RESTSource`
- [`mlpilot.ingestion.core`](#mlpilotingestioncore): `read`, `IngestResult`, `ingest`
- [`mlpilot.ingestion.files`](#mlpilotingestionfiles): `is_cloud`, `is_http`, `detect_format`, `read_file`, `list_files`
- [`mlpilot.ingestion.incremental`](#mlpilotingestionincremental): `Batch`, `IncrementalLoader`
- [`mlpilot.ingestion.schema`](#mlpilotingestionschema): `InferBooleans`, `type_pipeline`, `infer_types`
- [`mlpilot.ingestion.sql`](#mlpilotingestionsql): `is_sql_url`, `SQLSource`
- [`mlpilot.ingestion.warehouse`](#mlpilotingestionwarehouse): `SnowflakeSource`, `BigQuerySource`

---

## `mlpilot.ingestion`

<sub>Source: [`mlpilot/ingestion/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/ingestion/__init__.py)</sub>

Ingestion & connectors (module 1 of the platform).

* `read(source, ...)`      files, globs, URLs, S3/GCS/Azure, databases, warehouses, APIs -> DataFrame
* `ingest(source, ...)`    + type inference, data-contract validation, quality score, lineage
* `SQLSource`              any SQLAlchemy URL or DB-API connection (bound parameters only)
* `SnowflakeSource` / `BigQuerySource`   native warehouse connectors (Arrow results)
* `RESTSource`             JSON APIs with auth, pagination, retries and rate limiting
* `IncrementalLoader`      only new rows (watermark) or new files, with safe commits
* `infer_types`            text -> numbers / dates / booleans, reusable across batches

---

## `mlpilot.ingestion.api`

<sub>Source: [`mlpilot/ingestion/api.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/ingestion/api.py)</sub>

REST/JSON APIs: authentication, pagination, retries with back-off, rate limiting.

Pagination styles: `"page"` (?page=1,2,…), `"offset"` (?offset=0&limit=100),
`"cursor"` (a cursor value from the body), `"next"` (a next-page URL in the body) and
`"link"` (RFC 8288 `Link: <…>; rel="next"` header). Next-page URLs must stay on the same
origin as `base_url` so a token is never sent to another host.

### class `APIError`

```python
class APIError(RuntimeError):
```

**Constructor**

```python
APIError(status: int, url: str, body: str)
```

### class `RESTSource`

```python
class RESTSource:
```

A JSON API.

```python
api = RESTSource("https://api.example.com/v1", token_env="EXAMPLE_TOKEN")
api.fetch("orders", records="data", pagination="cursor", cursor_param="after",
          cursor_path="meta.next_cursor", params={"status": "paid"})
```

**Constructor**

```python
RESTSource(base_url: str, *, headers: dict | None=None, token: str | None=None, token_env: str | None=None, auth_header: str='Authorization', auth_scheme: str | None='Bearer', params: dict | None=None, timeout: float=30.0, retries: int=4, backoff: float=1.0, rate_limit: float | None=None, verify_origin: bool=True)
```

#### `RESTSource.get()`

```python
get(endpoint: str='', params: dict | None=None)
```

One GET with retries (429 / 5xx / network errors, honouring `Retry-After`).
Returns `(json_body, headers)`.

#### `RESTSource.fetch()`

```python
fetch(endpoint: str='', *, params: dict | None=None, records: str | None=None, pagination: str | None=None, page_param: str='page', start_page: int=1, size_param: str | None=None, page_size: int | None=None, offset_param: str='offset', limit_param: str='limit', cursor_param: str='cursor', cursor_path: str='next_cursor', next_path: str='next', max_pages: int=10000, flatten: bool=True) -> pl.DataFrame
```

All records from an endpoint as a DataFrame (nested objects flattened to `a.b` columns).

`records`: dotted path to the list of records in each response (default: the body if
it's a list, else the first list-valued field).

#### `RESTSource.read()`

```python
read(*, endpoint: str='', **kw) -> pl.DataFrame
```

---

## `mlpilot.ingestion.core`

<sub>Source: [`mlpilot/ingestion/core.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/ingestion/core.py)</sub>

One entry point for every source: `read()` returns a DataFrame, `ingest()` adds types,
validation, a data-contract check, lineage and an audit record.

### `read()`

```python
def read(source, *, query: str | None=None, table: str | None=None, params: dict | None=None, **kw)
```

Read from anything:

* a path, glob, `http(s)://` file URL or cloud URI (`s3://`, `gs://`, `az://`, `abfss://`)
* a database URL (`postgresql://…`, `mysql+pymysql://…`, `sqlite:///file.db` …) with
  `query=` (`:name` params in `params`) or `table=`
* a source object (`SQLSource`, `SnowflakeSource`, `BigQuerySource`, `RESTSource` …)
* an existing Polars / pandas DataFrame (passed through)

### class `IngestResult`

```python
class IngestResult:
```

The loaded data plus everything you'd want in an audit log.

| Field | Type | Default |
|---|---|---|
| `data` | `pl.DataFrame` | required |
| `source` | `str` | required |
| `name` | `str \| None` | required |
| `rows` | `int` | required |
| `columns` | `int` | required |
| `seconds` | `float` | required |
| `schema` | `dict` | required |
| `casts` | `dict` | required |
| `fingerprint` | `dict` | required |
| `contract` | `Any` | `None` |
| `quality` | `dict \| None` | `None` |
| `types_pipeline` | `Any` | `None` |

#### `IngestResult.passed` *(property)*

```python
passed: bool
```

#### `IngestResult.summary()`

```python
summary() -> str
```

#### `IngestResult.to_dict()`

```python
to_dict() -> dict
```

### `ingest()`

```python
def ingest(source, *, name: str | None=None, infer: bool=True, types=None, contract=None, on_violation: str='raise', quality: bool=False, run=None, **read_kw) -> IngestResult
```

Read, infer types, validate and record a dataset.

* `infer`: convert text columns holding numbers / dates / yes-no to real types; pass a
  fitted `types` pipeline (`result.types_pipeline`) to reuse the same conversions.
* `contract`: a `Contract` (or path to its YAML) to validate against; `on_violation`
  `"raise"` (`ContractViolation`), `"warn"` or `"ignore"`.
* `quality=True` adds the 0-100 quality score.
* `run`: a tracking `Run` — the dataset fingerprint is logged to it.

---

## `mlpilot.ingestion.files`

<sub>Source: [`mlpilot/ingestion/files.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/ingestion/files.py)</sub>

Files on local disk, HTTP(S), and cloud object storage (S3, GCS, Azure) in any common format.

Cloud paths (`s3://`, `gs://`, `az://`, `abfss://` …) are read by Polars' native
object-store client — no extra packages — with credentials from `storage_options` or the
standard environment variables (`AWS_ACCESS_KEY_ID`/`AWS_PROFILE`,
`GOOGLE_APPLICATION_CREDENTIALS`, `AZURE_STORAGE_ACCOUNT_NAME`/`…_KEY` …). Globs such as
`s3://bucket/sales/2024-*.parquet` read many files as one table.

### `is_cloud()`

```python
def is_cloud(path) -> bool
```

### `is_http()`

```python
def is_http(path) -> bool
```

### `detect_format()`

```python
def detect_format(path) -> str | None
```

### `read_file()`

```python
def read_file(path, *, format: str | None=None, columns=None, n_rows: int | None=None, storage_options=None, lazy: bool=False, include_source: bool=False, headers: dict | None=None, **kw)
```

Read a file / glob / URL / cloud object into a Polars DataFrame (`lazy=True`: LazyFrame).

`format` is detected from the extension (csv, tsv, parquet, json, ndjson, excel, ods,
arrow/feather, avro; `.gz`/`.zst` compressed CSV). `columns` and `n_rows` are pushed
down to the reader where possible. `include_source=True` adds a `_source_file` column.
Extra keyword arguments go to the underlying Polars reader (e.g. `sheet_name=`).

### `list_files()`

```python
def list_files(pattern: str) -> list
```

Local files matching a glob, sorted (used by incremental file loads).

---

## `mlpilot.ingestion.incremental`

<sub>Source: [`mlpilot/ingestion/incremental.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/ingestion/incremental.py)</sub>

Incremental loads: only the rows (or files) that are new since the last successful load.

Modes:

* **cursor** — a database table (or any callable) with a monotonically increasing column such
  as `updated_at` or an auto-increment id. Each load asks for rows past the saved watermark
  (as a bound parameter). With `key=`, rows that share the exact watermark value are
  re-checked and de-duplicated, so late rows with the same timestamp aren't lost.
* **files** — a folder/glob: only files that are new or changed (size or mtime) are read.

State lives in a small JSON file and is committed only after you've processed the batch
(`with loader.batch() as df: …` or `loader.run(sink)`), so a crash mid-way re-delivers the
same rows instead of silently skipping them. Two runners can't commit over each other.

### class `Batch`

```python
class Batch:
```

| Field | Type | Default |
|---|---|---|
| `data` | `pl.DataFrame` | required |
| `new_state` | `dict` | required |
| `base_version` | `int` | required |
| `info` | `dict` | required |

#### `Batch.empty` *(property)*

```python
empty: bool
```

### class `IncrementalLoader`

```python
class IncrementalLoader:
```

```python
loader = IncrementalLoader(SQLSource(url), table="orders", cursor="updated_at",
                             key="order_id", state="state/orders.json")
with loader.batch() as new_rows:          # state is saved only if this block succeeds
    append_to_warehouse(new_rows)
```

```python
files = IncrementalLoader(files="landing/*.csv", state="state/landing.json")
files.run(lambda df: df.write_parquet(...))
```

**Constructor**

```python
IncrementalLoader(source=None, *, state, table: str | None=None, cursor: str | None=None, key=None, files: str | None=None, read_kw: dict | None=None, infer: bool=True, columns=None, batch_limit: int | None=None)
```

#### `IncrementalLoader.state()`

```python
state() -> dict
```

#### `IncrementalLoader.watermark` *(property)*

```python
watermark
```

#### `IncrementalLoader.reset()`

```python
reset()
```

Forget everything: the next load is a full load.

#### `IncrementalLoader.load()`

```python
load() -> Batch
```

Fetch the next batch without committing it (see `commit`).

#### `IncrementalLoader.commit()`

```python
commit(batch: Batch)
```

Save the batch's watermark / file list. Fails if another run committed in between.

#### `IncrementalLoader.batch()`

```python
batch()
```

`with loader.batch() as df:` — commits only if the block finishes without error.

#### `IncrementalLoader.run()`

```python
run(sink=None) -> dict
```

Load, hand the rows to `sink(df)` (if any rows), commit; return the batch info.

---

## `mlpilot.ingestion.schema`

<sub>Source: [`mlpilot/ingestion/schema.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/ingestion/schema.py)</sub>

Schema inference: turn text columns that really hold numbers, dates or booleans into real types.

The result is a fitted, JSON-serialisable pipeline, so every later batch (e.g. incremental
loads) gets exactly the same types instead of being re-guessed batch by batch.

### class `InferBooleans`

```python
class InferBooleans(Step):
```

Learn text columns whose values are all yes/no-like and convert them to Boolean.

**Constructor**

```python
InferBooleans(*, columns=None, fit_only=None)
```

### `type_pipeline()`

```python
def type_pipeline() -> Pipeline
```

### `infer_types()`

```python
def infer_types(df: pl.DataFrame, pipeline: Pipeline | None=None)
```

Return `(typed_df, fitted_pipeline, casts)`; `casts` maps column -> "String -> Float64".

Pass a previously fitted `pipeline` to apply the same conversions without re-learning.

---

## `mlpilot.ingestion.sql`

<sub>Source: [`mlpilot/ingestion/sql.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/ingestion/sql.py)</sub>

Relational databases: any SQLAlchemy URL (PostgreSQL, MySQL, SQL Server, Oracle, SQLite,
Redshift, DuckDB, Snowflake/BigQuery dialects …) or any DB-API 2.0 connection.

Values always travel as bound parameters (write `:name` placeholders; they are translated
for the driver), and table/column names are quoted, so inputs can't inject SQL. Passwords are
redacted from every repr and report.

### `is_sql_url()`

```python
def is_sql_url(text) -> bool
```

### class `SQLSource`

```python
class SQLSource:
```

A database to read from.

```python
db = SQLSource("postgresql+psycopg://user:pass@host/db")      # needs SQLAlchemy + driver
db.query("SELECT * FROM orders WHERE region = :r", {"r": "EU"})
db.read_table("orders", since=("updated_at", last_seen))
SQLSource(sqlite3.connect("local.db"))                          # any DB-API connection
```

**Constructor**

```python
SQLSource(connection, *, quote: str | None=None)
```

#### `SQLSource.iter_query()`

```python
iter_query(sql: str, params: dict | None=None, batch_size: int=50000)
```

Yield the result in DataFrames of `batch_size` rows (bounded memory).

#### `SQLSource.query()`

```python
query(sql: str, params: dict | None=None) -> pl.DataFrame
```

Run a SELECT; `:name` placeholders are bound from `params`.

#### `SQLSource.read_table()`

```python
read_table(table: str, *, columns=None, where: dict | None=None, since=None, order_by=None, limit: int | None=None) -> pl.DataFrame
```

Read a table. `since=("updated_at", value)` keeps newer rows (`(col, value, True)`
for >=); `where={"region": "EU"}` filters by equality.

#### `SQLSource.read()`

```python
read(*, query: str | None=None, table: str | None=None, params=None, **kw) -> pl.DataFrame
```

#### `SQLSource.insert()`

```python
insert(table: str, rows, *, create: bool=True) -> int
```

Append rows (a DataFrame or a list of dicts) with bound parameters. `create=True`
creates the table first if it doesn't exist (column types inferred from the values).

#### `SQLSource.execute()`

```python
execute(sql: str, params: dict | None=None) -> None
```

Run a statement that returns no rows (DDL, DML) and commit.

#### `SQLSource.tables()`

```python
tables() -> list
```

#### `SQLSource.columns()`

```python
columns(table: str) -> dict
```

`{column: polars dtype}` from a zero-row read (works on any database).

#### `SQLSource.close()`

```python
close()
```

---

## `mlpilot.ingestion.warehouse`

<sub>Source: [`mlpilot/ingestion/warehouse.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/ingestion/warehouse.py)</sub>

Cloud data warehouses with their native connectors (Arrow results — fast for large reads).

* `SnowflakeSource`  needs `snowflake-connector-python[pandas]` (pyarrow fetches)
* `BigQuerySource`   needs `google-cloud-bigquery` (+ `google-cloud-bigquery-storage` for speed)

Both accept the same `:name` placeholders as `SQLSource` and bind them as real query
parameters. Credentials come from the usual places (Snowflake: `connect()` arguments such as
`authenticator="externalbrowser"` or a key pair; BigQuery: Application Default Credentials).

### class `SnowflakeSource`

```python
class SnowflakeSource:
```

```python
sf = SnowflakeSource(account="xy12345", user="ME", authenticator="externalbrowser",
                        warehouse="WH", database="DB", schema="PUBLIC")
sf.query("SELECT * FROM orders WHERE region = :r", {"r": "EU"})
```

**Constructor**

```python
SnowflakeSource(*, connection=None, **connect_kwargs)
```

#### `SnowflakeSource.query()`

```python
query(sql: str, params: dict | None=None) -> pl.DataFrame
```

#### `SnowflakeSource.read_table()`

```python
read_table(table: str, **kw) -> pl.DataFrame
```

#### `SnowflakeSource.read()`

```python
read(*, query=None, table=None, params=None, **kw) -> pl.DataFrame
```

#### `SnowflakeSource.close()`

```python
close()
```

### class `BigQuerySource`

```python
class BigQuerySource:
```

```python
bq = BigQuerySource(project="my-project")
bq.query("SELECT * FROM `ds.orders` WHERE day >= :d", {"d": date(2024, 1, 1)})
```

**Constructor**

```python
BigQuerySource(project: str | None=None, *, client=None, location: str | None=None, credentials=None)
```

#### `BigQuerySource.query()`

```python
query(sql: str, params: dict | None=None) -> pl.DataFrame
```

#### `BigQuerySource.read_table()`

```python
read_table(table: str, **kw) -> pl.DataFrame
```

#### `BigQuerySource.read()`

```python
read(*, query=None, table=None, params=None, **kw) -> pl.DataFrame
```
