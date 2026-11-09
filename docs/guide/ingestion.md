# Ingestion & connectors


`ac.read()` reads from any supported source and returns a Polars DataFrame. `ac.ingest()` also
infers types, validates the data and records where it came from.

```python
ac.read("sales.csv")                                   # csv, tsv, parquet, json, ndjson, excel, arrow, avro, .csv.gz
ac.read("data/2024-*.parquet", include_source=True)    # globs -> one table (+ _source_file column)
ac.read("https://example.com/export.csv")
ac.read("s3://bucket/sales/*.parquet", storage_options={"aws_region": "eu-west-1"})   # also gs://, az://, abfss://
ac.read("postgresql+psycopg://user:pass@host/db", query="SELECT * FROM orders WHERE region = :r",
        params={"r": "EU"})
ac.read("sqlite:///shop.db", table="orders")

result = ac.ingest("sales.csv", name="sales", contract="sales.contract.yaml", quality=True)
result.data                 # typed DataFrame: "1,200" -> 1200.0, "2024-01-05" -> date, "yes" -> True
print(result)               # rows, inferred types, contract result, quality score
result.fingerprint          # shape + schema + content hash (also logged with ingest(..., run=run))
```

**Connectors:**
- **Cloud storage:** S3, GCS and Azure are read natively by Polars, with no extra packages.
  Credentials come from `storage_options` or the standard environment variables.
- **Databases:** `SQLSource` takes any SQLAlchemy URL (PostgreSQL, MySQL, SQL Server, Oracle,
  Redshift, DuckDB, …) or any DB-API connection. Install with
  `pip install "mlpilot[sql]"` plus your database's driver.
- **Warehouses:** `SnowflakeSource` (`[snowflake]`) and `BigQuerySource` (`[bigquery]`) use the
  native connectors and fetch results as Arrow.
- **APIs:** `RESTSource` handles token auth (pass the token, or read it from an environment
  variable), and every pagination style: page, offset, cursor, next-URL and `Link` header. It
  retries 429 and 5xx responses with back-off and `Retry-After`, supports rate limiting, and
  flattens nested JSON into columns.

**Safety:**
- Values are always bound as query parameters (write `:name`), and table and column names are
  quoted, so inputs can't inject SQL.
- Passwords and tokens are redacted from every repr and report.
- An API never follows a next-page link to another host, so your token can't leak there.

```python
api = ac.RESTSource("https://api.example.com/v1", token_env="EXAMPLE_TOKEN", rate_limit=5)
orders = api.fetch("orders", records="data", pagination="cursor",
                   cursor_param="after", cursor_path="meta.next_cursor")
```

**Incremental loads** read only what is new since the last successful load. The state is saved
only after your code has processed the batch, so a crash re-delivers the rows instead of
dropping them. If two runners try to commit, the second gets an error rather than overwriting.

```python
loader = ac.IncrementalLoader(ac.SQLSource(url), table="orders", cursor="updated_at",
                              key="order_id", state="state/orders.json")
with loader.batch() as new_rows:          # rows past the saved watermark; late rows with the
    append_to_warehouse(new_rows)         # same timestamp are caught and de-duplicated by key

files = ac.IncrementalLoader(files="landing/**/*.csv", state="state/landing.json")
files.run(lambda df: df.write_parquet("bronze/batch.parquet"))   # only new or changed files
```

Types are learned from the first batch and then reused, so every later batch has the same
schema.
