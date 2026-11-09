"""Cloud data warehouses with their native connectors (Arrow results — fast for large reads).

* ``SnowflakeSource``  needs ``snowflake-connector-python[pandas]`` (pyarrow fetches)
* ``BigQuerySource``   needs ``google-cloud-bigquery`` (+ ``google-cloud-bigquery-storage`` for speed)

Both accept the same ``:name`` placeholders as ``SQLSource`` and bind them as real query
parameters. Credentials come from the usual places (Snowflake: ``connect()`` arguments such as
``authenticator="externalbrowser"`` or a key pair; BigQuery: Application Default Credentials).
"""

from __future__ import annotations

import datetime as _dt
import re
from decimal import Decimal

from typing import cast

import polars as pl

from ._util import build_select

_NAMED = re.compile(r"(?<![:\w]):([A-Za-z_]\w*)")


def _need(module: str, extra: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:
        raise ImportError(f'This connector needs {module}: pip install "dataspoof-mlpilot[{extra}]"') from exc


class SnowflakeSource:
    """>>> sf = SnowflakeSource(account="xy12345", user="ME", authenticator="externalbrowser",
    ...                         warehouse="WH", database="DB", schema="PUBLIC")
    >>> sf.query("SELECT * FROM orders WHERE region = :r", {"r": "EU"})"""

    quote = '"'
    dialect = "snowflake"

    def __init__(self, *, connection=None, **connect_kwargs):
        if connection is None:
            connector = _need("snowflake.connector", "snowflake")
            from ..secretstore import resolve
            connection = connector.connect(**{k: resolve(v) for k, v in connect_kwargs.items()})
        self._conn = connection
        self._account = connect_kwargs.get("account")

    def query(self, sql: str, params: dict | None = None) -> pl.DataFrame:
        sql = _NAMED.sub(lambda m: f"%({m[1]})s", sql) if params else sql   # snowflake: pyformat binding
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params or None)
            table = cur.fetch_arrow_all()
            if table is None:                                      # no rows
                return pl.DataFrame({d[0]: [] for d in cur.description or []})
            return cast(pl.DataFrame, pl.from_arrow(table))
        finally:
            cur.close()

    def read_table(self, table: str, **kw) -> pl.DataFrame:
        return self.query(*build_select(table, quote=self.quote, **kw))

    def read(self, *, query=None, table=None, params=None, **kw) -> pl.DataFrame:
        if query is not None:
            return self.query(query, params)
        if table is not None:
            return self.read_table(table, **kw)
        raise ValueError("Pass query='SELECT …' or table='name'.")

    def close(self):
        self._conn.close()

    def __repr__(self) -> str:
        return f"SnowflakeSource(account={self._account!r})"


def _bq_param(bigquery, name, value):
    if isinstance(value, bool):
        kind = "BOOL"
    elif isinstance(value, int):
        kind = "INT64"
    elif isinstance(value, float):
        kind = "FLOAT64"
    elif isinstance(value, Decimal):
        kind = "NUMERIC"
    elif isinstance(value, _dt.datetime):
        kind = "TIMESTAMP" if value.tzinfo is not None else "DATETIME"
    elif isinstance(value, _dt.date):
        kind = "DATE"
    elif isinstance(value, bytes):
        kind = "BYTES"
    elif isinstance(value, (list, tuple)):
        inner = _bq_param(bigquery, name, value[0]).type_ if value else "STRING"
        return bigquery.ArrayQueryParameter(name, inner, list(value))
    else:
        kind, value = "STRING", str(value)
    return bigquery.ScalarQueryParameter(name, kind, value)


class BigQuerySource:
    """>>> bq = BigQuerySource(project="my-project")
    >>> bq.query("SELECT * FROM `ds.orders` WHERE day >= :d", {"d": date(2024, 1, 1)})"""

    quote = "`"
    dialect = "bigquery"

    def __init__(self, project: str | None = None, *, client=None, location: str | None = None, credentials=None):
        if client is None:
            bigquery = _need("google.cloud.bigquery", "bigquery")
            client = bigquery.Client(project=project, credentials=credentials, location=location)
        self._client, self.project, self.location = client, project, location

    def query(self, sql: str, params: dict | None = None) -> pl.DataFrame:
        bigquery = _need("google.cloud.bigquery", "bigquery")
        config = None
        if params:
            sql = _NAMED.sub(lambda m: f"@{m[1]}", sql)          # bigquery: named @params
            config = bigquery.QueryJobConfig(query_parameters=[_bq_param(bigquery, k, v) for k, v in params.items()])
        job = self._client.query(sql, job_config=config, location=self.location)
        return cast(pl.DataFrame, pl.from_arrow(job.to_arrow()))

    def read_table(self, table: str, **kw) -> pl.DataFrame:
        return self.query(*build_select(table, quote=self.quote, **kw))

    def read(self, *, query=None, table=None, params=None, **kw) -> pl.DataFrame:
        if query is not None:
            return self.query(query, params)
        if table is not None:
            return self.read_table(table, **kw)
        raise ValueError("Pass query='SELECT …' or table='name'.")

    def __repr__(self) -> str:
        return f"BigQuerySource(project={self.project!r})"
