"""Serving: the feedback endpoint, hot reload without restarts, and live drift gauges."""

import io
import json
import shutil
import time

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_classification

import mlpilot as ac
from mlpilot.monitoring import join_feedback

FAST = dict(time_budget=30, cv=3, verbose=False, max_trials=1, include=["logistic_regression"])


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=600, n_features=4, n_informative=3, n_redundant=0, random_state=0)
    df = pl.DataFrame(X, schema=list("abcd")).with_columns(pl.Series("y", y))
    return df.head(400), df.tail(200)


@pytest.fixture(scope="module")
def model(data):
    return ac.AutoML("y", **FAST).fit(data[0])


@pytest.fixture(scope="module")
def flipped(data):
    """A deliberately different model: trained on inverted labels, so its predictions differ."""
    return ac.AutoML("y", **FAST).fit(data[0].with_columns((1 - pl.col("y")).alias("y")))


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


def recs(df, n=5):
    return [{k: v for k, v in r.items() if k != "y"} for r in df.head(n).to_dicts()]


# ── feedback ────────────────────────────────────────────────────────────────
def test_feedback_endpoint_classification(tmp_path, model, data):
    _, test = data
    log = tmp_path / "pred.jsonl"
    app = ac.make_app(model, api_key="k", name="churn", version="1", prediction_log=log)
    status, res = call(app, "POST", "/predict", {"records": recs(test, 10)}, key="k", headers={"X-Request-ID": "req1"})
    assert status == 200 and res["prediction_ids"] == [f"req1-{i}" for i in range(10)]
    truth = test["y"].head(10).to_list()
    fb = [{"prediction_id": pid, "label": t} for pid, t in zip(res["prediction_ids"], truth)]
    fb.append({"prediction_id": "req1-0", "rating": 2, "comment": "looked wrong to the agent"})
    fb.append({"prediction_id": "nope-1", "label": 1})
    status, out = call(app, "POST", "/feedback", {"feedback": fb}, key="k")
    assert status == 200 and out == {"accepted": 12, "matched": 11, "unknown_ids": ["nope-1"]}
    correct = sum(int(p == t) for p, t in zip(res["predictions"], truth))
    text = call(app, "GET", "/metrics")[1]
    assert f'ac_feedback_total{{model="churn",version="1",variant="primary",result="correct"}} {correct}' in text
    assert 'result="unmatched"} 1' in text and 'result="rating"} 1' in text
    assert f'ac_live_accuracy{{model="churn",version="1",variant="primary"}} {correct / 10:.6f}' in text
    # validation and auth
    assert call(app, "POST", "/feedback", {"prediction_id": "req1-0"}, key="k")[0] == 422          # nothing to record
    assert call(app, "POST", "/feedback", {"prediction_id": 5, "label": 1}, key="k")[0] == 422
    assert call(app, "POST", "/feedback", {"prediction_id": "req1-0", "comment": "x" * 3000}, key="k")[0] == 422
    assert call(app, "POST", "/feedback", {"prediction_id": "req1-0", "label": 1})[0] == 401
    assert call(app, "GET", "/feedback", key="k")[0] == 405
    # the feedback log joins back to the predictions and feeds live performance checks
    joined = join_feedback(log, f"{log}.feedback.jsonl")
    assert joined.height == 10 and set(joined.columns) >= {"a", "prediction", "label", "rating", "comment"}
    assert joined.filter(pl.col("prediction_id") == "req1-0")["rating"][0] == 2            # latest feedback wins
    mon = ac.Monitor(model, test)
    live = mon.check_labels(joined.filter(pl.col("label").is_not_null()))
    assert live["rows"] == 9 and live["metric"] == "accuracy"


def test_feedback_regression_records_abs_error(data):
    reg = ac.AutoML("a", time_budget=30, cv=3, verbose=False, max_trials=1, include=["ridge"]).fit(data[0].drop("y"))
    app = ac.make_app(reg)
    _, res = call(app, "POST", "/predict", {"records": [{"b": 0.1, "c": 0.2, "d": 0.3}]}, headers={"X-Request-ID": "r"})
    _, out = call(app, "POST", "/feedback", {"prediction_id": "r-0", "label": res["predictions"][0] + 2.0})
    assert out["matched"] == 1
    text = call(app, "GET", "/metrics")[1]
    assert "ac_feedback_abs_error_count" in text and 'result="scored"} 1' in text
    assert call(app, "POST", "/feedback", {"prediction_id": "r-0", "label": "high"})[0] == 422


# ── hot reload ──────────────────────────────────────────────────────────────
def test_hot_reload_follows_registry_promotions(tmp_path, model, flipped, data):
    _, test = data
    reg = ac.ModelRegistry(tmp_path / "reg")
    reg.register(model, "m", stage="production")
    app = ac.deploy.app_from_registry(reg, "m", api_key="k")
    before = call(app, "POST", "/predict", {"records": recs(test, 20)}, key="k")[1]
    assert before["version"] == 1 and app.metadata["version"] == 1
    assert app.reload() == {"reloaded": False, "version": 1}                # nothing new yet
    reg.register(flipped, "m", stage="staging")
    reg.promote("m", 2, "production")
    status, res = call(app, "POST", "/admin/reload", key="k")
    assert status == 200 and res == {"reloaded": True, "version": 2}
    after = call(app, "POST", "/predict", {"records": recs(test, 20)}, key="k")[1]
    assert after["version"] == 2 and after["predictions"] != before["predictions"]
    assert app.metadata["version"] == 2 and app.model is not model
    text = call(app, "GET", "/metrics")[1]
    assert 'ac_model_reloads_total{model="m",version="2",result="ok"} 1' in text
    assert call(app, "POST", "/admin/reload")[0] == 401                     # needs the key
    assert call(ac.make_app(model), "POST", "/admin/reload")[0] == 403        # no key configured -> refused
    app.close()


def test_background_watch_and_failed_reload_keeps_serving(tmp_path, model, flipped, data):
    _, test = data
    path = tmp_path / "served_model"
    model.save(path)
    app = ac.make_app(str(path), watch=0.2, version="a")
    first = call(app, "POST", "/predict", {"records": recs(test, 20)})[1]["predictions"]
    staged = tmp_path / "staged"
    flipped.save(staged)
    shutil.rmtree(path)
    shutil.copytree(staged, path)                                           # a deploy replaces the directory
    for _ in range(50):
        if call(app, "POST", "/predict", {"records": recs(test, 20)})[1]["predictions"] != first:
            break
        time.sleep(0.2)
    else:
        pytest.fail("the watcher never picked up the replaced model")
    app.close()

    def broken():
        raise RuntimeError("registry unavailable")
    app2 = ac.make_app(model, reloader=broken)
    out = app2.reload()
    assert out["reloaded"] is False and "registry unavailable" in out["error"]
    assert call(app2, "POST", "/predict", {"records": recs(test, 3)})[0] == 200        # still serving
    assert 'result="failed"} 1' in call(app2, "GET", "/metrics")[1]


# ── live drift gauges ───────────────────────────────────────────────────────
def test_live_drift_and_estimated_performance_gauges(model, data):
    train, test = data
    mon = ac.Monitor(model, test)
    app = ac.make_app(model, monitor=mon, drift_window=400, drift_every=200)
    rng = np.random.default_rng(0)
    shifted = [{"a": float(r["a"]) + 3.0 + rng.normal(0, 0.1), "b": r["b"], "c": r["c"], "d": r["d"]}
               for r in train.to_dicts()]
    for i in range(0, 400, 100):
        assert call(app, "POST", "/predict", {"records": shifted[i:i + 100]})[0] == 200
    app.drift_pool.shutdown(wait=True)
    text = call(app, "GET", "/metrics")[1]
    psi_a = float(next(line.split()[-1] for line in text.splitlines() if line.startswith("ac_feature_psi") and 'feature="a"' in line))
    psi_b = float(next(line.split()[-1] for line in text.splitlines() if line.startswith("ac_feature_psi") and 'feature="b"' in line))
    assert psi_a > 0.25 > psi_b
    assert 'ac_performance{model="model",version="",metric="roc_auc",kind="estimated"}' in text
    assert "ac_drift_share" in text and "ac_monitor_alerts" in text and "ac_reference_performance" in text
