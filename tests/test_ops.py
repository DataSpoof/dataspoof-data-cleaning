"""Production hardening: serving (metrics, rollouts, cloud routes, ASGI, latency), signing,
skops, monitoring operations, SQL writes, scale-out helpers and SQL pushdown."""

import asyncio
import io
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.datasets import make_classification

import mlpilot as ac
from mlpilot.deploy.cli import container_command
from mlpilot.ingestion import SQLSource
from mlpilot.monitoring import Monitor, join_labels, opsgenie_alert, pagerduty_alert, sql_sink

FAST = dict(time_budget=30, cv=3, verbose=False, max_trials=2, include=["logistic_regression", "random_forest"])


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=400, n_features=4, n_informative=3, n_redundant=0, random_state=0)
    df = pl.DataFrame(X, schema=list("abcd")).with_columns(pl.Series("id", np.arange(400)), pl.Series("y", y))
    return df.head(300), df.tail(100)


@pytest.fixture(scope="module")
def model(data):
    return ac.AutoML("y", **FAST).fit(data[0].drop("id"))


@pytest.fixture(scope="module")
def other_model(data):
    return ac.AutoML("y", **{**FAST, "include": ["logistic_regression"], "ensemble": False}).fit(data[0].drop("id"))


def call(app, method, path, body=None, key=None, headers=None):
    raw = json.dumps(body).encode() if body is not None else b""
    env = {"REQUEST_METHOD": method, "PATH_INFO": path, "CONTENT_LENGTH": str(len(raw)),
           "CONTENT_TYPE": "application/json", "wsgi.input": io.BytesIO(raw)}
    if key:
        env["HTTP_X_API_KEY"] = key
    for k, v in (headers or {}).items():
        env["HTTP_" + k.upper().replace("-", "_")] = v
    out = {}
    payload = b"".join(app(env, lambda s, h: out.update(status=int(s.split()[0]), headers=dict(h))))
    text = payload.decode()
    return out["status"], (json.loads(text) if out["headers"]["Content-Type"] == "application/json" else text)


def records(df, n=3):
    return [{k: v for k, v in r.items() if k not in ("y",)} for r in df.head(n).to_dicts()]


# ── serving ─────────────────────────────────────────────────────────────────

def test_predict_with_proba_and_inference_threads(data, model):
    _, test = data
    labels, proba = model.predict_with_proba(test)
    assert labels.to_list() == model.predict(test).to_list()
    assert np.allclose(proba.to_numpy(), model.predict_proba(test).to_numpy())
    jobs = [getattr(e, "n_jobs", None) for _, e in getattr(model.model_, "named_estimators_", {}).items()]
    assert all(j in (None, 1) for j in jobs)


def test_metrics_cloud_routes_and_payload_shapes(data, model, monkeypatch):
    _, test = data
    monkeypatch.setenv("AIP_PREDICT_ROUTE", "/v1/models/m:predict")
    monkeypatch.setenv("AIP_HEALTH_ROUTE", "/v1/models/m")
    app = ac.make_app(model, api_key="k", name="churn", version="2")
    rec = records(test)
    assert call(app, "GET", "/ping")[0] == 200 and call(app, "GET", "/v1/models/m")[0] == 200
    for path, body in [("/invocations", {"instances": rec}), ("/score", {"data": rec}),
                       ("/v1/models/m:predict", {"instances": rec}), ("/predict", rec)]:
        status, res = call(app, "POST", path, body, key="k")
        assert status == 200 and len(res["predictions"]) == 3 and res["variant"] == "primary", path
    call(app, "POST", "/predict", {"records": []}, key="k")                       # a 422
    status, text = call(app, "GET", "/metrics")                                   # public by default
    assert status == 200 and 'ac_requests_total{model="churn",version="2",route="predict",status="200",variant="primary"} 4' in text
    assert 'ac_rows_scored_total{model="churn",version="2",variant="primary"} 12' in text
    assert 'ac_predict_duration_seconds_count{model="churn",version="2"} 4' in text and 'status="422"' in text
    private = ac.make_app(model, api_key="k", public_metrics=False)
    assert call(private, "GET", "/metrics")[0] == 401 and call(private, "GET", "/metrics", key="k")[0] == 200


def test_canary_and_shadow(tmp_path, data, model, other_model):
    _, test = data
    log = tmp_path / "pred.jsonl"
    app = ac.make_app(model, canary=(other_model, 0.3), shadow=other_model, prediction_log=log)
    seen = {"primary": 0, "canary": 0}
    for i in range(60):
        status, res = call(app, "POST", "/predict", {"records": records(test, 1)}, headers={"X-Request-ID": f"r{i}"})
        seen[res["variant"]] += 1
    assert 8 <= seen["canary"] <= 30 and seen["primary"] > seen["canary"]
    # the same request id always goes to the same variant (stable retries)
    v1 = call(app, "POST", "/predict", {"records": records(test, 1)}, headers={"X-Request-ID": "fixed"})[1]["variant"]
    v2 = call(app, "POST", "/predict", {"records": records(test, 1)}, headers={"X-Request-ID": "fixed"})[1]["variant"]
    assert v1 == v2
    app.shadow_pool.shutdown(wait=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 62 and all("shadow_prediction" in r for r in rows)
    text = call(app, "GET", "/metrics")[1]
    assert "ac_shadow_predictions_total" in text and 'variant="canary"' in text
    with pytest.raises(ValueError):
        ac.make_app(model, canary=(other_model, 1.5))


def test_app_from_registry(tmp_path, model, other_model, data):
    reg = ac.ModelRegistry(tmp_path / "reg")
    reg.register(model, "m", stage="production")
    reg.register(other_model, "m", stage="staging")
    app = ac.deploy.app_from_registry(reg, "m", canary_weight=0.5)
    variants = {call(app, "POST", "/predict", {"records": records(data[1], 1)},
                     headers={"X-Request-ID": str(i)})[1]["variant"] for i in range(30)}
    assert variants == {"primary", "canary"} and app.metadata["version"] == 1


def test_asgi_app(data, model):
    app = ac.deploy.make_asgi_app(model)
    body = json.dumps({"records": records(data[1], 2)}).encode()
    sent = []

    async def run(path, method, payload=b""):
        msgs = [{"type": "http.request", "body": payload, "more_body": False}]

        async def receive():
            return msgs.pop(0)

        async def send(m):
            sent.append(m)
        await app({"type": "http", "method": method, "path": path, "headers": [(b"content-type", b"application/json")]},
                  receive, send)
    asyncio.run(run("/predict", "POST", body))
    assert sent[0]["status"] == 200 and len(json.loads(sent[1]["body"])["predictions"]) == 2
    sent.clear()
    asyncio.run(run("/health", "GET"))
    assert sent[0]["status"] == 200


def test_container_command_and_docker_targets(tmp_path, model, monkeypatch):
    monkeypatch.setenv("AC_PORT", "8000")
    assert "0.0.0.0:8000" in container_command()
    monkeypatch.setenv("AIP_HTTP_PORT", "7080")                      # Vertex AI wins
    assert "0.0.0.0:7080" in container_command()
    out = ac.build_docker(model, tmp_path / "sm", target="sagemaker")
    assert "AC_PORT=8080" in (out / "Dockerfile").read_text()
    with pytest.raises(ValueError):
        ac.build_docker(model, tmp_path / "x", target="heroku")


# ── signing & skops ─────────────────────────────────────────────────────────

def test_signed_models(tmp_path, model, monkeypatch, data):
    pytest.importorskip("cryptography")
    from mlpilot.security import SignatureError, generate_keypair, sign_model, verify_model
    priv, pub = generate_keypair(tmp_path / "k.pem", tmp_path / "k.pub")
    other_priv, other_pub = generate_keypair(tmp_path / "o.pem", tmp_path / "o.pub", password="pw")
    path = model.save(tmp_path / "m")
    with pytest.raises(SignatureError, match="not signed"):
        verify_model(path, pub)
    sign_model(path, priv, signer="ci")
    assert verify_model(path, pub)["signer"] == "ci"
    with pytest.raises(SignatureError, match="different key"):
        verify_model(path, other_pub)
    monkeypatch.setenv("AC_MODEL_PUBLIC_KEY", str(pub))
    assert ac.load_model(path).predict(data[1]).len() == 100            # verified on load
    (path / "preprocess.json").write_text((path / "preprocess.json").read_text() + " ")
    with pytest.raises(SignatureError, match="modified"):
        ac.load_model(path)
    monkeypatch.delenv("AC_MODEL_PUBLIC_KEY")
    reg = ac.ModelRegistry(tmp_path / "reg")
    v = reg.register(model, "m", sign_with=other_priv, sign_password="pw")
    assert verify_model(v.model_path, other_pub)["key_id"]


def test_skops_export(tmp_path, data, model):
    pytest.importorskip("skops")
    from mlpilot.deploy import load_skops
    out = ac.export_model(model, tmp_path / "sk", format="skops")
    assert (out / "model.skops").exists() and not (out / "model.joblib").exists()
    loaded = load_skops(out)                                    # sklearn types are trusted by default
    assert loaded.predict(data[1]).to_list() == model.predict(data[1]).to_list()
    (out / "model.skops").write_bytes((out / "model.skops").read_bytes()[:-10])
    with pytest.raises(Exception):
        load_skops(out)


# ── monitoring operations ───────────────────────────────────────────────────

class _Capture(BaseHTTPRequestHandler):
    got = []

    def log_message(self, *a):
        pass

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _Capture.got.append((self.path, dict(self.headers), json.loads(body)))
        self.send_response(202)
        self.end_headers()


@pytest.fixture()
def capture():
    _Capture.got = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Capture)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", _Capture.got
    httpd.shutdown()


def test_incident_alerts(capture, data, model):
    url, got = capture
    _, test = data
    mon = Monitor(model, test, name="churn")
    mon.on_alert(pagerduty_alert("rk-123", url=url + "/v2/enqueue"))
    mon.on_alert(opsgenie_alert("og-key"))           # real endpoint unreachable here -> warning only
    shifted = test.with_columns((pl.col("a") * 5 + 3).alias("a"), (pl.col("b") - 4).alias("b"), (1 - pl.col("y")).alias("y"))
    with pytest.warns(ac.MLPilotWarning):
        report = mon.check(shifted)
    assert report.retrain_recommended and report.name == "churn"
    pd_events = [b for p, h, b in got if p == "/v2/enqueue"]
    assert pd_events and all(e["routing_key"] == "rk-123" and e["event_action"] == "trigger" for e in pd_events)
    assert {e["dedup_key"] for e in pd_events} >= {"churn:performance", "churn:data_drift"}
    assert any(e["payload"]["severity"] == "critical" for e in pd_events)


def test_delayed_labels_and_sql_sink(tmp_path, data, model):
    _, test = data
    log = tmp_path / "pred.jsonl"
    app = ac.make_app(model, prediction_log=log)
    call(app, "POST", "/predict", {"records": records(test, 100)})
    labels = test.select("id", pl.col("y").alias("outcome"))
    joined = join_labels(log, labels.head(80), on="id")
    assert joined.height == 80 and {"prediction", "label", "variant"} <= set(joined.columns)
    db = SQLSource(sqlite3.connect(tmp_path / "mon.db"))
    mon = Monitor(model, test).on_check(sql_sink(db, "model_monitoring"))
    live = mon.check_labels(joined)
    assert live["metric"] == "accuracy" and live["rows"] == 80 and not live["alerts"]
    wrong = joined.with_columns((1 - pl.col("label")).alias("label"))
    alerts = []
    mon.on_alert(lambda r, a: alerts.extend(a))
    assert mon.check_labels(wrong)["alerts"] and alerts[0].kind == "performance"
    mon.check(test)
    mon.check(test.head(50))
    stored = db.query("SELECT checked_at, rows, retrain_recommended FROM model_monitoring")
    assert stored.height == 2 and stored["rows"].to_list() == [100, 50]


def test_sql_insert_both_drivers(tmp_path):
    rows = [{"id": 1, "name": "a'b", "score": 0.5, "ok": True}, {"id": 2, "name": None, "score": None, "ok": False}]
    db = SQLSource(sqlite3.connect(tmp_path / "x.db"))
    assert db.insert("t", rows) == 2 and db.insert("t", pl.DataFrame(rows)) == 2
    assert db.query("SELECT COUNT(*) AS n FROM t WHERE name = :n", {"n": "a'b"})["n"][0] == 2
    pytest.importorskip("sqlalchemy")
    sa = SQLSource(f"sqlite:///{tmp_path / 'y.db'}")
    sa.insert("events", rows)
    assert sa.read_table("events")["name"].to_list() == ["a'b", None]


# ── scale ───────────────────────────────────────────────────────────────────

def test_transform_file_and_partition_scorer(tmp_path, data, model):
    train, test = data
    pipe = ac.Pipeline([ac.Impute(), ac.Scale()]).fit(train.drop("y"))
    test.write_csv(tmp_path / "in.csv")
    info = ac.transform_file(pipe, tmp_path / "in.csv", tmp_path / "out.parquet", chunk_size=30)
    assert info["rows"] == 100 and info["chunks"] == 4
    assert np.allclose(pl.read_parquet(tmp_path / "out.parquet")["a"].to_numpy(),
                       pipe.transform(test.drop("y"))["a"].to_numpy())
    path = model.save(tmp_path / "m")
    fn = ac.scale.partition_scorer(path, keep=["id"])
    parts = [test.to_pandas().iloc[:40], test.to_pandas().iloc[40:]]
    out = pd.concat(list(fn(iter(parts))))
    assert list(out.columns[:2]) == ["id", "prediction"] and len(out) == 100
    assert out["prediction"].tolist() == model.predict(test).to_list()


def test_sql_pushdown_matches_python(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    df = pl.DataFrame({"age": rng.normal(40, 12, n).round(), "income": rng.lognormal(10, 1, n),
                       "city": rng.choice(["pune", "delhi", "goa", "x1", "x2"], n, p=[.4, .3, .27, .02, .01]),
                       "plan": rng.choice(["a", "b", "c"], n), "junk": [None] * n})
    df = df.with_columns(pl.when(pl.Series(rng.random(n) < 0.1)).then(None).otherwise(pl.col("income")).alias("income"),
                         pl.when(pl.Series(rng.random(n) < 0.2)).then(None).otherwise(pl.col("plan")).alias("plan"))
    pipe = ac.Pipeline([ac.DropColumns(), ac.Impute(numeric="median", indicator_min_missing=0.05), ac.ClipOutliers(k=1.5),
                        ac.GroupRareCategories(min_frequency=0.05), ac.EncodeCategorical(method="onehot"),
                        ac.LogTransformSkewed(), ac.Scale(method="standard")])
    expected = pipe.fit_transform(df)
    sql = ac.to_sql(pipe, "raw_t", dialect="sqlite")
    con = sqlite3.connect(":memory:")
    df.to_pandas().to_sql("raw_t", con, index=False)
    try:
        cur = con.execute(sql)
    except sqlite3.OperationalError as exc:            # SQLite builds without math functions lack LN
        pytest.skip(f"sqlite lacks a function: {exc}")
    got = pl.DataFrame(cur.fetchall(), schema=[d[0] for d in cur.description], orient="row", infer_schema_length=None)
    assert sorted(got.columns) == sorted(expected.columns)
    for c in expected.columns:
        assert np.allclose(expected[c].cast(pl.Float64).to_numpy(), got[c].cast(pl.Float64).to_numpy(), equal_nan=True), c
    with pytest.raises(NotImplementedError, match="safe cast"):
        ac.to_sql(ac.Pipeline([ac.ConvertNumericText(columns=["x"])]).fit(pl.DataFrame({"x": ["1", "2"]})), "t",
                  dialect="postgresql")
    assert "TRY_CAST" in ac.to_sql(ac.Pipeline([ac.ConvertNumericText(columns=["x"])]).fit(pl.DataFrame({"x": ["1"]})),
                                   "t", dialect="snowflake")
    with pytest.raises(NotImplementedError, match="parse_dates"):
        ac.to_sql(ac.Pipeline([ac.ParseDates(columns=["d"])]).fit(pl.DataFrame({"d": ["2024-01-01"]})), "t")
