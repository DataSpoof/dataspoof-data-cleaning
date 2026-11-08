"""Relational databases: any SQLAlchemy URL (PostgreSQL, MySQL, SQL Server, Oracle, SQLite,
Redshift, DuckDB, Snowflake/BigQuery dialects …) or any DB-API 2.0 connection.

Values always travel as bound parameters (write ``:name`` placeholders; they are translated
for the driver), and table/column names are quoted, so inputs can't inject SQL. Passwords are
redacted from every repr and report.
"""

from __future__ import annotations

import importlib

import polars as pl

from ._util import build_select, quote_ident, redact, translate_params

SQL_SCHEMES = ("sqlite", "postgresql", "postgres", "mysql", "mariadb", "mssql", "oracle", "redshift", "duckdb",
               "snowflake", "bigquery", "trino", "presto", "clickhouse", "db2", "ibm_db_sa", "hana", "teradata",
               "databricks", "cockroachdb", "awsathena", "vertica", "firebird", "sybase", "access")


def is_sql_url(text) -> bool:
    s = str(text)
    if "://" not in s:
        return False
    scheme = s.split("://", 1)[0].lower().split("+", 1)[0]
    return scheme in SQL_SCHEMES


def _frame(rows, cols) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame({c: [] for c in cols})
    return pl.DataFrame([tuple(r) for r in rows], schema=list(cols), orient="row", infer_schema_length=None)


class SQLSource:
    """A database to read from.

    >>> db = SQLSource("postgresql+psycopg://user:pass@host/db")      # needs SQLAlchemy + driver
    >>> db.query("SELECT * FROM orders WHERE region = :r", {"r": "EU"})
    >>> db.read_table("orders", since=("updated_at", last_seen))
    >>> SQLSource(sqlite3.connect("local.db"))                          # any DB-API connection
    """

    quote = '"'

    def __init__(self, connection, *, quote: str | None = None):
        self._url = None
        if isinstance(connection, str):
            self._url = connection
            try:
                import sqlalchemy
            except ImportError:
                if connection.startswith("sqlite:///"):
                    import sqlite3
                    connection = sqlite3.connect(connection[len("sqlite:///"):])
                else:
                    raise ImportError('Database URLs need SQLAlchemy and a driver: pip install '
                                      '"AutomatedCleaning[sql]" (+ e.g. psycopg, pymysql)') from None
            else:
                connection = sqlalchemy.create_engine(connection)
        self._conn = connection
        self._sa = type(connection).__module__.startswith("sqlalchemy")
        if self._sa:
            self.dialect = connection.dialect.name
        else:
            module = importlib.import_module(type(connection).__module__.split(".")[0])
            self.dialect = {"sqlite3": "sqlite", "psycopg2": "postgresql", "psycopg": "postgresql"}.get(
                module.__name__, module.__name__)
            self._paramstyle = getattr(module, "paramstyle", "qmark")
        if quote is not None:
            self.quote = quote
        elif self.dialect in ("mysql", "mariadb", "bigquery", "databricks", "clickhouse"):
            self.quote = "`"
        elif self.dialect == "mssql":
            self.quote = "["

    # ── reading ─────────────────────────────────────────────────────────────
    def iter_query(self, sql: str, params: dict | None = None, batch_size: int = 50_000):
        """Yield the result in DataFrames of ``batch_size`` rows (bounded memory)."""
        if self._sa:
            from sqlalchemy import text
            with self._conn.connect() as conn:
                result = conn.execution_options(stream_results=True).execute(text(sql), params or {})
                cols = list(result.keys())
                yielded = False
                while True:
                    rows = result.fetchmany(batch_size)
                    if not rows:
                        break
                    yielded = True
                    yield _frame(rows, cols)
                if not yielded:
                    yield _frame([], cols)
            return
        q, p = translate_params(sql, params, self._paramstyle)
        cur = self._conn.cursor()
        try:
            cur.execute(q, p) if p else cur.execute(q)
            cols = [d[0] for d in cur.description or []]
            yielded = False
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                yielded = True
                yield _frame(rows, cols)
            if not yielded:
                yield _frame([], cols)
        finally:
            cur.close()

    def query(self, sql: str, params: dict | None = None) -> pl.DataFrame:
        """Run a SELECT; ``:name`` placeholders are bound from ``params``."""
        parts = list(self.iter_query(sql, params))
        return parts[0] if len(parts) == 1 else pl.concat(parts, how="vertical_relaxed")

    def read_table(self, table: str, *, columns=None, where: dict | None = None, since=None, order_by=None,
                   limit: int | None = None) -> pl.DataFrame:
        """Read a table. ``since=("updated_at", value)`` keeps newer rows (``(col, value, True)``
        for >=); ``where={"region": "EU"}`` filters by equality."""
        sql, params = build_select(table, columns=columns, where=where, since=since, order_by=order_by,
                                   limit=limit, quote=self.quote)
        if self.dialect == "mssql" and limit is not None:      # SQL Server has TOP, not LIMIT
            sql = sql.replace(f" LIMIT {int(limit)}", "").replace("SELECT ", f"SELECT TOP {int(limit)} ", 1)
        return self.query(sql, params)

    def read(self, *, query: str | None = None, table: str | None = None, params=None, **kw) -> pl.DataFrame:
        if query is not None:
            return self.query(query, params)
        if table is not None:
            return self.read_table(table, **kw)
        raise ValueError("Pass query='SELECT …' or table='name'.")

    # ── metadata ────────────────────────────────────────────────────────────
    def tables(self) -> list:
        if self._sa:
            from sqlalchemy import inspect
            return sorted(inspect(self._conn).get_table_names())
        if self.dialect == "sqlite":
            return self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")["name"].to_list()
        return self.query("SELECT table_name FROM information_schema.tables")["table_name"].to_list()

    def columns(self, table: str) -> dict:
        """``{column: polars dtype}`` from a zero-row read (works on any database)."""
        return {c: str(t) for c, t in self.read_table(table, limit=1).schema.items()}

    def close(self):
        if self._sa:
            self._conn.dispose()
        else:
            self._conn.close()

    def __repr__(self) -> str:
        where = redact(self._url) if self._url else type(self._conn).__name__
        return f"SQLSource({where}, dialect={self.dialect})"


__all__ = ["SQLSource", "is_sql_url", "quote_ident"]
