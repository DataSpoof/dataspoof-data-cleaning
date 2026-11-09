"""AWS Glue: the Data Catalog and Glue jobs (needs ``boto3``; the ``[aws]`` extra).

>>> from mlpilot.integrations import aws_glue as glue
>>> glue.write_table(clean_df, "analytics", "customers", "s3://lake/analytics/customers/",
...                  partition_by=["country"], description="Cleaned customer master",
...                  parameters={"quality_score": "97.5"})        # Parquet on S3 + a catalog table
>>> glue.register_table(df, "analytics", "orders", "s3://lake/orders/")   # catalog only (data already there)
>>> glue.table_schema("analytics", "customers")                 # -> Polars schema from the catalog
>>> df = glue.read_table("analytics", "customers")              # location + format from the catalog
>>> glue.run_job("nightly-orders-etl", {"--date": "2026-11-08"})  # start a Glue job and wait

Athena, Redshift Spectrum, EMR and Lake Formation all read the same catalog. Every function takes
``glue=`` / ``storage_options=`` for a custom client or endpoint.
"""

from __future__ import annotations

import time

import polars as pl

try:
    import boto3
except ImportError as exc:                                   # pragma: no cover
    raise ImportError('The AWS Glue integration needs boto3: pip install "dataspoof-mlpilot[aws]"') from exc

_FORMATS = {
    "parquet": {"InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                "SerdeInfo": {"SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"}},
    "csv": {"InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
            "SerdeInfo": {"SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
                          "Parameters": {"field.delim": ",", "skip.header.line.count": "1"}}},
    "json": {"InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
             "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
             "SerdeInfo": {"SerializationLibrary": "org.openx.data.jsonserde.JsonSerDe"}},
}


def hive_type(dtype) -> str:
    """Polars dtype -> Glue / Hive type string."""
    if dtype == pl.Int64 or dtype == pl.UInt32:
        return "bigint"
    if dtype in (pl.Int32, pl.UInt16):
        return "int"
    if dtype in (pl.Int16, pl.UInt8):
        return "smallint"
    if dtype == pl.Int8:
        return "tinyint"
    if dtype == pl.UInt64:
        return "decimal(20,0)"
    if dtype == pl.Float64:
        return "double"
    if dtype == pl.Float32:
        return "float"
    if dtype == pl.Boolean:
        return "boolean"
    if dtype == pl.Date:
        return "date"
    if isinstance(dtype, pl.Datetime):
        return "timestamp"
    if isinstance(dtype, pl.Decimal):
        return f"decimal({dtype.precision or 38},{dtype.scale or 0})"
    if isinstance(dtype, pl.List):
        return f"array<{hive_type(dtype.inner)}>"
    if isinstance(dtype, pl.Struct):
        return "struct<" + ",".join(f"{f.name}:{hive_type(f.dtype)}" for f in dtype.fields) + ">"
    if dtype == pl.Binary:
        return "binary"
    return "string"


def polars_type(hive: str):
    """Glue / Hive type string -> Polars dtype (complex types become String)."""
    h = hive.strip().lower()
    simple = {"bigint": pl.Int64, "int": pl.Int32, "integer": pl.Int32, "smallint": pl.Int16, "tinyint": pl.Int8,
              "double": pl.Float64, "float": pl.Float32, "boolean": pl.Boolean, "date": pl.Date,
              "timestamp": pl.Datetime("us"), "string": pl.String, "binary": pl.Binary}
    if h in simple:
        return simple[h]
    if h.startswith(("varchar", "char")):
        return pl.String
    if h.startswith("decimal"):
        p, _, s = h[h.index("(") + 1:h.index(")")].partition(",")
        return pl.Decimal(int(p), int(s or 0))
    if h.startswith("array<"):
        return pl.List(polars_type(h[6:-1]))
    return pl.String


def _client(glue):
    return glue or boto3.client("glue")


def register_table(df_or_schema, database: str, table: str, location: str, *, fmt: str = "parquet",
                   partition_by: list | None = None, description: str | None = None, parameters: dict | None = None,
                   column_comments: dict | None = None, create_database: bool = True, glue=None) -> dict:
    """Create or update a catalog table describing data at ``location``. ``df_or_schema``: a
    DataFrame or a ``{column: dtype}`` schema. Returns the Glue ``TableInput`` that was written."""
    if fmt not in _FORMATS:
        raise ValueError(f"fmt must be one of {sorted(_FORMATS)}")
    c = _client(glue)
    schema = df_or_schema.schema if isinstance(df_or_schema, pl.DataFrame) else dict(df_or_schema)
    parts = list(partition_by or [])
    comments = column_comments or {}

    def col(name, dtype):
        out = {"Name": name, "Type": hive_type(dtype)}
        if comments.get(name):
            out["Comment"] = str(comments[name])[:255]
        return out
    table_input = {
        "Name": table, "TableType": "EXTERNAL_TABLE",
        "Parameters": {"classification": fmt, "EXTERNAL": "TRUE", "created_by": "mlpilot",
                       **({"skip.header.line.count": "1"} if fmt == "csv" else {}),
                       **{str(k): str(v) for k, v in (parameters or {}).items()}},
        "StorageDescriptor": {"Columns": [col(n, t) for n, t in schema.items() if n not in parts],
                              "Location": location.rstrip("/") + "/", **_FORMATS[fmt]},
        "PartitionKeys": [col(n, schema[n]) for n in parts],
    }
    if description:
        table_input["Description"] = description[:2048]
    if create_database:
        try:
            c.create_database(DatabaseInput={"Name": database})
        except c.exceptions.AlreadyExistsException:
            pass
    try:
        c.create_table(DatabaseName=database, TableInput=table_input)
    except c.exceptions.AlreadyExistsException:
        c.update_table(DatabaseName=database, TableInput=table_input)
    return table_input


def write_table(df: pl.DataFrame, database: str, table: str, location: str, *, partition_by: list | None = None,
                storage_options: dict | None = None, glue=None, add_partitions: bool = True, **register_kw) -> dict:
    """Write ``df`` as Parquet under ``location`` (Hive-style ``key=value`` folders when
    ``partition_by``), register the table, and add its partitions to the catalog."""
    base = location.rstrip("/")
    parts = list(partition_by or [])
    if parts:
        for keys, group in df.group_by(parts, maintain_order=True):
            sub = "/".join(f"{k}={v}" for k, v in zip(parts, keys))
            group.drop(parts).write_parquet(f"{base}/{sub}/part-0.parquet", storage_options=storage_options)
    else:
        df.write_parquet(f"{base}/part-0.parquet", storage_options=storage_options)
    out = register_table(df, database, table, location, partition_by=parts, glue=glue, **register_kw)
    if parts and add_partitions:
        c = _client(glue)
        sd = out["StorageDescriptor"]
        inputs = [{"Values": [str(v) for v in keys],
                   "StorageDescriptor": {**sd, "Location": f"{base}/" + "/".join(f"{k}={v}" for k, v in zip(parts, keys)) + "/"}}
                  for keys in df.select(parts).unique(maintain_order=True).iter_rows()]
        for i in range(0, len(inputs), 100):
            c.batch_create_partition(DatabaseName=database, TableName=table, PartitionInputList=inputs[i:i + 100])
    return out


def describe_table(database: str, table: str, *, glue=None) -> dict:
    return _client(glue).get_table(DatabaseName=database, Name=table)["Table"]


def table_schema(database: str, table: str, *, glue=None) -> dict:
    """``{column: polars dtype}`` from the catalog (partition keys included, last)."""
    t = describe_table(database, table, glue=glue)
    cols = t["StorageDescriptor"]["Columns"] + t.get("PartitionKeys", [])
    return {c["Name"]: polars_type(c["Type"]) for c in cols}


def read_table(database: str, table: str, *, storage_options: dict | None = None, lazy: bool = False, glue=None):
    """Read a catalog table from its S3 location in the format the catalog records."""
    t = describe_table(database, table, glue=glue)
    loc = t["StorageDescriptor"]["Location"].rstrip("/")
    fmt = (t.get("Parameters") or {}).get("classification", "parquet").lower()
    schema = table_schema(database, table, glue=glue)
    hive = bool(t.get("PartitionKeys"))
    if fmt == "parquet":
        lf = pl.scan_parquet(f"{loc}/**/*.parquet", storage_options=storage_options, hive_partitioning=hive)
    elif fmt == "csv":
        lf = pl.scan_csv(f"{loc}/**/*.csv", storage_options=storage_options)
    elif fmt == "json":
        lf = pl.scan_ndjson(f"{loc}/**/*.json", storage_options=storage_options)
    else:
        raise ValueError(f"Table format {fmt!r} isn't supported for reading (parquet, csv, json)")
    lf = lf.with_columns([pl.col(c).cast(t_, strict=False) for c, t_ in schema.items()
                          if c in lf.collect_schema().names()])
    return lf if lazy else lf.collect()


def run_job(name: str, arguments: dict | None = None, *, wait: bool = True, poll: float = 15.0,
            timeout: float = 6 * 3600, glue=None) -> dict:
    """Start a Glue job (``arguments`` like ``{"--date": "2026-11-08"}``) and, with ``wait``, poll
    until it finishes. Raises ``RuntimeError`` if it fails, stops or times out."""
    c = _client(glue)
    run_id = c.start_job_run(JobName=name, Arguments={str(k): str(v) for k, v in (arguments or {}).items()})["JobRunId"]
    out = {"job": name, "run_id": run_id, "state": "STARTING"}
    if not wait:
        return out
    t0 = time.time()
    while True:
        jr = c.get_job_run(JobName=name, RunId=run_id)["JobRun"]
        state = jr.get("JobRunState", "UNKNOWN")
        out.update(state=state, seconds=jr.get("ExecutionTime"), error=jr.get("ErrorMessage"))
        if state == "SUCCEEDED":
            return out
        if state in ("FAILED", "ERROR", "TIMEOUT", "STOPPED", "EXPIRED"):
            raise RuntimeError(f"Glue job {name} run {run_id} ended {state}: {jr.get('ErrorMessage') or ''}".strip())
        if time.time() - t0 > timeout:
            raise RuntimeError(f"Glue job {name} run {run_id} still {state} after {timeout:.0f}s")
        time.sleep(poll)
