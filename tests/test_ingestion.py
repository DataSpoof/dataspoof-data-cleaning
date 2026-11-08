"""Tests for ingestion & connectors."""

import datetime as dt
import gzip
import json
import sqlite3
import sys
import threading
import types
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import polars as pl
import pytest

import automatedcleaning as ac
from automatedcleaning.ingestion import APIError, IncrementalLoader, RESTSource, SQLSource, read_file
from automatedcleaning.ingestion._util import build_select, quote_ident, redact, translate_params

SAMPLE = pl.DataFrame({"id": [1, 2, 3], "city": ["Pune", "Delhi", None], "amount": [1.5, 2.0, 3.25]})


# ── files ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ext", ["csv", "tsv", "parquet", "ndjson", "json", "arrow", "xlsx"])
def test_read_file_formats(tmp_path, ext):
    path = tmp_path / f"data.{ext}"
    writer = {"csv": SAMPLE.write_csv, "parquet": SAMPLE.write_parquet, "ndjson": SAMPLE.write_ndjson,
              "json": SAMPLE.write_json, "arrow": SAMPLE.write_ipc}
    if ext == "tsv":
        SAMPLE.write_csv(path, separator="\t")
    elif ext == "xlsx":
        pytest.importorskip("fastexcel")
        pytest.importorskip("openpyxl")
        SAMPLE.to_pandas().to_excel(path, index=False)
    else:
        writer[ext](path)
    out = ac.read(path)
    assert out.height == 3 and out["city"].to_list() == ["Pune", "Delhi", None]
    assert out["amount"].to_list() == [1.5, 2.0, 3.25]


def test_read_file_options(tmp_path):
    for i in range(3):
        SAMPLE.with_columns(pl.lit(i).alias("part")).write_parquet(tmp_path / f"part-{i}.parquet")
    many = read_file(str(tmp_path / "part-*.parquet"), include_source=True)
    assert many.height == 9 and many["_source_file"].n_unique() == 3
    assert read_file(str(tmp_path / "part-*.parquet"), columns=["id"], n_rows=2).shape == (2, 1)
    assert isinstance(read_file(tmp_path / "part-0.parquet", lazy=True), pl.LazyFrame)
    gz = tmp_path / "data.csv.gz"
    gz.write_bytes(gzip.compress(SAMPLE.write_csv().encode()))
    assert ac.read(gz).height == 3
    (tmp_path / "mystery.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="format"):
        ac.read(tmp_path / "mystery.bin")
    with pytest.raises(ValueError, match="cloud storage"):
        read_file("s3://bucket/book.xlsx")


def test_cloud_paths_use_polars_object_store(monkeypatch):
    seen = {}

    def fake_scan(path, **kw):
        seen.update(path=path, **kw)
        return SAMPLE.lazy()
    monkeypatch.setattr(pl, "scan_parquet", fake_scan)
    out = ac.read("s3://bucket/sales/2024-*.parquet", storage_options={"aws_region": "eu-west-1"})
    assert out.height == 3 and seen["path"].startswith("s3://") and seen["storage_options"]["aws_region"] == "eu-west-1"


# ── HTTP: files and APIs ────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    hits = {}

    def log_message(self, *a):
        pass

    def _json(self, obj, status=200, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        url = urllib.parse.urlsplit(self.path)
        q = dict(urllib.parse.parse_qsl(url.query))
        _Handler.hits[url.path] = _Handler.hits.get(url.path, 0) + 1
        items = [{"id": i, "user": {"name": f"u{i}", "tier": "gold" if i % 2 else "basic"}} for i in range(1, 8)]
        if url.path == "/data.csv":
            body = SAMPLE.write_csv().encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/secure":
            if self.headers.get("Authorization") != "Bearer t0k":
                return self._json({"error": "no"}, 401)
            self._json({"data": items[:2]})
        elif url.path == "/page":
            page, size = int(q["page"]), int(q["size"])
            self._json({"data": items[(page - 1) * size: page * size]})
        elif url.path == "/offset":
            off, lim = int(q["offset"]), int(q["limit"])
            self._json(items[off: off + lim])
        elif url.path == "/cursor":
            start = int(q.get("after", 0))
            chunk = items[start: start + 3]
            self._json({"results": chunk, "meta": {"next": str(start + 3) if start + 3 < len(items) else None}})
        elif url.path == "/next":
            start = int(q.get("from", 0))
            nxt = f"/next?from={start + 4}" if start + 4 < len(items) else None
            self._json({"items": items[start: start + 4], "links": {"next": nxt}})
        elif url.path == "/link":
            start = int(q.get("from", 0))
            hdr = {"Link": f'<{self._base}/link?from={start + 5}>; rel="next"'} if start + 5 < len(items) else {}
            self._json(items[start: start + 5], headers=hdr)
        elif url.path == "/flaky":
            if _Handler.hits[url.path] < 3:
                return self._json({"error": "busy"}, 429 if _Handler.hits[url.path] == 1 else 503,
                                  {"Retry-After": "0"})
            self._json({"data": items[:1]})
        elif url.path == "/evil":
            self._json({"items": items[:1], "links": {"next": "http://attacker.invalid/steal"}})
        else:
            self._json({"error": "missing"}, 404)


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    base = f"http://127.0.0.1:{httpd.server_port}"
    _Handler._base = base
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield base
    httpd.shutdown()


def test_http_file(server):
    assert ac.read(f"{server}/data.csv")["amount"].sum() == pytest.approx(6.75)


def test_rest_pagination_styles(server):
    api = RESTSource(server, backoff=0)
    assert api.fetch("page", records="data", pagination="page", size_param="size", page_size=3).height == 7
    assert api.fetch("offset", pagination="offset", page_size=3).height == 7
    cur = api.fetch("cursor", records="results", pagination="cursor", cursor_param="after", cursor_path="meta.next")
    assert cur["id"].to_list() == list(range(1, 8))
    assert api.fetch("next", records="items", pagination="next", next_path="links.next").height == 7
    link = api.fetch("link", pagination="link")
    assert link.height == 7 and {"user.name", "user.tier"} <= set(link.columns)      # flattened


def test_rest_auth_retries_and_safety(server, monkeypatch):
    with pytest.raises(APIError) as err:
        RESTSource(server, backoff=0).fetch("secure", records="data")
    assert err.value.status == 401
    monkeypatch.setenv("MY_TOKEN", "t0k")
    api = RESTSource(server, token_env="MY_TOKEN", backoff=0)
    assert api.fetch("secure", records="data").height == 2
    assert "t0k" not in repr(api)
    flaky = RESTSource(server, backoff=0)
    assert flaky.fetch("flaky", records="data").height == 1 and flaky.requests_made == 3
    with pytest.raises(ValueError, match="Refusing"):
        RESTSource(server, token="secret", backoff=0).fetch("evil", records="items", pagination="next",
                                                            next_path="links.next")
    with pytest.raises(ValueError, match="not set"):
        RESTSource(server, token_env="NOT_SET_ANYWHERE_123")
    with pytest.raises(ValueError, match="http"):
        RESTSource("ftp://example.com")


# ── SQL ─────────────────────────────────────────────────────────────────────

def _orders_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE orders (order_id INTEGER, region TEXT, amount TEXT, updated_at TEXT)")
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", [
        (1, "EU", "1,200.50", "2024-01-01 10:00:00"), (2, "US", "300", "2024-01-02 09:00:00"),
        (3, "EU", "45.25", "2024-01-03 08:00:00")])
    con.commit()
    return con


def test_identifier_quoting_and_params():
    assert quote_ident("sales.orders") == '"sales"."orders"'
    assert quote_ident('a"b') == '"a""b"'
    assert quote_ident("x]", "[") == "[x]]]"
    for bad in ("", "a..b", "x\x00"):
        with pytest.raises(ValueError):
            quote_ident(bad)
    sql, params = build_select("t", columns=["a"], where={"r": "EU", "z": None}, since=("u", 5), order_by="u",
                               limit=10)
    assert sql == 'SELECT "a" FROM "t" WHERE "r" = :w0 AND "z" IS NULL AND "u" > :since ORDER BY "u" LIMIT 10'
    assert params == {"w0": "EU", "since": 5}
    assert translate_params("a = :x AND b = :y", {"x": 1, "y": 2}, "qmark") == ("a = ? AND b = ?", [1, 2])
    assert translate_params("a = :x", {"x": 1}, "pyformat")[0] == "a = %(x)s"
    assert translate_params("v::int = :x", {"x": 1}, "qmark")[0] == "v::int = ?"
    assert redact("postgresql://bob:hunter2@db:5432/x") == "postgresql://bob:***@db:5432/x"
    assert "s3cr3t" not in redact("https://h/api?token=s3cr3t&page=2")


@pytest.mark.parametrize("kind", ["dbapi", "sqlalchemy"])
def test_sql_source(tmp_path, kind):
    path = tmp_path / "shop.db"
    _orders_db(path).close()
    if kind == "sqlalchemy":
        pytest.importorskip("sqlalchemy")
        db = SQLSource(f"sqlite:///{path}")
    else:
        db = SQLSource(sqlite3.connect(path))
    assert db.dialect == "sqlite" and db.tables() == ["orders"]
    assert db.query("SELECT * FROM orders WHERE region = :r", {"r": "EU"})["order_id"].to_list() == [1, 3]
    t = db.read_table("orders", columns=["order_id", "region"], since=("updated_at", "2024-01-01 10:00:00"),
                      order_by="order_id", limit=5)
    assert t.columns == ["order_id", "region"] and t["order_id"].to_list() == [2, 3]
    assert db.read_table("orders", where={"region": "US"}).height == 1
    evil = db.query("SELECT * FROM orders WHERE region = :r", {"r": "EU' OR '1'='1"})
    assert evil.height == 0                                         # bound, not inlined
    with pytest.raises(Exception):
        db.read_table('orders" ; DROP TABLE orders; --')            # quoted as one (missing) name
    assert db.tables() == ["orders"]
    assert db.query("SELECT * FROM orders WHERE 1 = 0").columns == ["order_id", "region", "amount", "updated_at"]
    assert sum(b.height for b in db.iter_query("SELECT * FROM orders", batch_size=2)) == 3
    db.close()


def test_read_dispatch_sql_url(tmp_path):
    pytest.importorskip("sqlalchemy")
    path = tmp_path / "shop.db"
    _orders_db(path).close()
    assert ac.read(f"sqlite:///{path}", table="orders").height == 3
    assert ac.read(f"sqlite:///{path}", query="SELECT COUNT(*) AS n FROM orders")["n"][0] == 3


# ── warehouses (fake connectors: the real ones need accounts) ───────────────

def test_snowflake_source_binds_and_fetches_arrow():
    import pyarrow as pa
    calls = []

    class Cur:
        description = [("A",)]

        def execute(self, sql, params):
            calls.append((sql, params))

        def fetch_arrow_all(self):
            return pa.table({"A": [1, 2]}) if "WHERE" in calls[-1][0] else None

        def close(self):
            pass
    conn = types.SimpleNamespace(cursor=Cur, close=lambda: None)
    sf = ac.SnowflakeSource(connection=conn)
    out = sf.read_table("DB.PUBLIC.T", where={"REGION": "EU"})
    assert out["A"].to_list() == [1, 2]
    assert calls[-1] == ('SELECT * FROM "DB"."PUBLIC"."T" WHERE "REGION" = %(w0)s', {"w0": "EU"})
    assert sf.query("SELECT A FROM T").columns == ["A"]              # empty result keeps columns


def test_bigquery_source_uses_query_parameters(monkeypatch):
    import pyarrow as pa
    made = []
    fake = types.ModuleType("google.cloud.bigquery")
    fake.ScalarQueryParameter = lambda n, t, v: types.SimpleNamespace(name=n, type_=t, value=v)
    fake.ArrayQueryParameter = lambda n, t, v: types.SimpleNamespace(name=n, type_=f"ARRAY<{t}>", value=v)
    fake.QueryJobConfig = lambda query_parameters: types.SimpleNamespace(params=query_parameters)
    google, cloud = types.ModuleType("google"), types.ModuleType("google.cloud")
    google.cloud, cloud.bigquery = cloud, fake
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake)

    class Client:
        def query(self, sql, job_config=None, location=None):
            made.append((sql, job_config))
            return types.SimpleNamespace(to_arrow=lambda: pa.table({"n": [7]}))
    bq = ac.BigQuerySource(client=Client())
    out = bq.query("SELECT n FROM `p.d.t` WHERE day >= :d AND ok = :b AND id IN UNNEST(:ids)",
                   {"d": dt.date(2024, 1, 1), "b": True, "ids": [1, 2]})
    assert out["n"][0] == 7
    sql, cfg = made[0]
    assert "@d" in sql and "@b" in sql and ":d" not in sql
    assert [(p.name, p.type_) for p in cfg.params] == [("d", "DATE"), ("b", "BOOL"), ("ids", "ARRAY<INT64>")]
    assert bq.read_table("p.d.t", limit=1) is not None and made[-1][0] == "SELECT * FROM `p`.`d`.`t` LIMIT 1"


# ── schema inference & ingest ───────────────────────────────────────────────

def test_infer_types_is_reusable():
    raw = pl.DataFrame({"price": ["$1,200", "300", "(45)", "12.5", "7"], "when": ["2024-01-05"] * 5,
                        "active": ["yes", "no", "Y", "N", None], "zip": ["01234", "02134", "11001", "20001", "30301"],
                        "name": ["a", "b", "c", "d", "e"]})
    typed, pipe, casts = ac.infer_types(raw)
    assert typed["price"].to_list() == [1200.0, 300.0, -45.0, 12.5, 7.0]
    assert typed.schema["when"] in (pl.Date, pl.Datetime) and typed["active"].to_list() == [True, False, True, False, None]
    assert typed.schema["zip"] == pl.String and set(casts) == {"price", "when", "active"}
    later, _, _ = ac.infer_types(pl.DataFrame({"price": ["9"], "when": ["2024-02-01"], "active": ["no"],
                                               "zip": ["99999"], "name": ["z"]}), pipe)
    assert later.schema == typed.schema


def test_ingest_contract_quality_and_lineage(tmp_path):
    from automatedcleaning.quality import Contract, ContractViolation
    src = tmp_path / "s.csv"
    pl.DataFrame({"id": [1, 2, 3], "amount": ["10", "20", "x30"]}).write_csv(src)
    res = ac.ingest(src, name="sales", quality=True)
    assert res.rows == 3 and res.fingerprint["rows"] == 3 and res.quality["grade"] in "ABCDF"
    contract = Contract.from_dict({"name": "sales", "columns": {"id": {"type": "integer", "unique": True},
                                                                 "amount": {"type": "number"}}})
    with pytest.raises(ContractViolation):
        ac.ingest(src, contract=contract)
    with pytest.warns(ac.AutomatedCleaningWarning, match="contract"):
        assert not ac.ingest(src, contract=contract, on_violation="warn").passed
    tr = ac.Tracker(tmp_path / "runs")
    with tr.start_run() as run:
        ac.ingest(src, name="sales", run=run)
    assert run.datasets[0]["name"] == "sales"
    assert json.loads(json.dumps(res.to_dict()))["rows"] == 3
    import pandas as pd
    assert ac.read(pd.DataFrame({"a": [1]})).height == 1


# ── incremental loads ───────────────────────────────────────────────────────

def test_incremental_sql_watermark_late_rows_and_safe_commit(tmp_path):
    con = _orders_db(tmp_path / "shop.db")
    loader = IncrementalLoader(SQLSource(con), table="orders", cursor="updated_at", key="order_id",
                               state=tmp_path / "state" / "orders.json")
    first_batch = []
    first = loader.run(first_batch.append)
    assert first["rows"] == 3 and loader.watermark == "2024-01-03 08:00:00"
    assert loader.run()["rows"] == 0                                  # nothing new

    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", [
        (4, "US", "10", "2024-01-03 08:00:00"),                       # late row, same timestamp as watermark
        (5, "EU", "20", "2024-01-04 12:00:00")])
    con.commit()
    with pytest.raises(RuntimeError):                                 # failure inside the block: no commit
        with loader.batch() as df:
            assert sorted(df["order_id"].to_list()) == [4, 5]
            raise RuntimeError("sink failed")
    with loader.batch() as df:                                        # re-delivered after the failure
        assert sorted(df["order_id"].to_list()) == [4, 5]
        assert df.schema == first_batch[0].schema                     # same types as batch 1, not re-guessed
    assert loader.run()["rows"] == 0

    stale = loader.load()
    con.execute("INSERT INTO orders VALUES (6, 'EU', '1', '2024-02-01 00:00:00')")
    con.commit()
    loader.commit(loader.load())
    with pytest.raises(RuntimeError, match="changed"):
        loader.commit(stale)
    loader.reset()
    assert loader.run()["rows"] == 6


def test_incremental_files_and_callable(tmp_path):
    land = tmp_path / "landing"
    land.mkdir()
    SAMPLE.write_csv(land / "a.csv")
    loader = IncrementalLoader(files=str(land / "*.csv"), state=tmp_path / "files.json")
    got = []
    assert loader.run(got.append)["files"] == [str(land / "a.csv")] and got[0].height == 3
    assert loader.run(got.append)["rows"] == 0 and len(got) == 1    # sink not called for empty batches
    SAMPLE.write_csv(land / "b.csv")
    (land / "a.csv").write_text(SAMPLE.head(1).write_csv())          # changed file is re-read
    info = loader.run(got.append)
    assert sorted(info["files"]) == [str(land / "a.csv"), str(land / "b.csv")] and got[-1].height == 4
    assert "_source_file" in got[-1].columns

    events = pl.DataFrame({"ts": [1, 2, 3, 4], "v": ["a", "b", "c", "d"]})
    calls = []

    def fetch(watermark):
        calls.append(watermark)
        return events
    cb = IncrementalLoader(fetch, cursor="ts", state=tmp_path / "cb.json")
    assert cb.run()["rows"] == 4 and cb.run()["rows"] == 0 and calls == [None, 4]
    with pytest.raises(ValueError):
        IncrementalLoader(SQLSource(sqlite3.connect(":memory:")), cursor="ts", state=tmp_path / "x.json")
