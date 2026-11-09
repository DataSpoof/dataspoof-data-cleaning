"""Tests for drift detection, the Monitor, alerts and retraining."""

import json

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_classification, make_regression

import mlpilot as ac
from mlpilot.monitoring import Monitor, psi

FAST = dict(time_budget=30, cv=3, verbose=False, max_trials=2, include=["logistic_regression", "ridge",
                                                                         "random_forest"])


def _frame(n, seed, shift=0.0, cat_p=(0.5, 0.3, 0.2)):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"num": rng.normal(shift, 1, n), "cat": rng.choice(["a", "b", "c"], n, p=cat_p),
                         "flag": rng.integers(0, 2, n)})


def test_psi_basics():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(p, p) == pytest.approx(0)
    assert psi(p, np.array([0.7, 0.1, 0.1, 0.1])) > 0.25


def test_drift_report_detects_shift_and_schema_changes():
    ref = _frame(2000, 0)
    same = ac.drift_report(ref, _frame(2000, 1))
    assert same.drifted == [] and same.drift_share == 0
    moved = _frame(2000, 2, shift=1.5, cat_p=(0.1, 0.2, 0.7)).with_columns(
        pl.when(pl.int_range(pl.len()) < 600).then(None).otherwise(pl.col("flag")).alias("flag"))
    rep = ac.drift_report(ref, moved)
    rows = {r["column"]: r for r in rep.columns.iter_rows(named=True)}
    assert rows["num"]["severity"] == "major" and rows["num"]["mean_shift_std"] > 1
    assert rows["cat"]["severity"] == "major"
    assert rows["flag"]["severity"] == "major" and rows["flag"]["missing_cur"] == pytest.approx(0.3)
    assert "num" in rep.summary()
    schema = ac.drift_report(ref, _frame(500, 3).drop("cat").with_columns(pl.col("flag").cast(pl.String),
                                                                          pl.lit(1).alias("new")))
    assert schema.missing_columns == ["cat"] and schema.new_columns == ["new"] and "flag" in schema.dtype_changes
    unseen = ac.drift_report(ref, _frame(500, 4).with_columns(pl.lit("zzz").alias("cat")))
    assert unseen.columns.filter(pl.col("column") == "cat")["unseen_share"][0] == 1.0


def test_baseline_is_json_and_has_no_raw_rows(tmp_path):
    from mlpilot.monitoring import Baseline
    secret = pl.DataFrame({"email": [f"user{i}@corp.com" for i in range(300)], "x": np.arange(300.0)})
    b = Baseline.fit(secret)
    text = json.dumps(b.to_dict())
    assert "user1@corp.com" not in text and "email" not in b.columns     # ID-like column skipped
    assert Baseline.from_dict(json.loads(text)).columns["x"].edges == b.columns["x"].edges


@pytest.fixture(scope="module")
def clf():
    X, y = make_classification(n_samples=1200, n_features=4, n_informative=3, n_redundant=0, random_state=0)
    df = pl.DataFrame(X, schema=list("abcd")).with_columns(pl.Series("y", y))
    model = ac.AutoML("y", **FAST).fit(df.head(600))
    return model, df.slice(600, 300), df.tail(300)


def test_monitor_healthy_then_drift_and_decay(tmp_path, clf):
    model, ref, live = clf
    alerts = []
    mon = Monitor(model, ref, log_path=tmp_path / "mon.jsonl", name="m").on_alert(lambda r, a: alerts.extend(a))
    ok = mon.check(live)
    assert not ok.alerts and not ok.retrain_recommended and ok.performance["metric"] == "roc_auc"

    flipped = live.with_columns((1 - pl.col("y")).alias("y"), (pl.col("a") * 3 + 2).alias("a"),
                                (pl.col("b") - 3).alias("b"))
    bad = mon.check(flipped)
    kinds = {a.kind for a in bad.critical}
    assert {"performance", "data_drift"} <= kinds and bad.retrain_recommended
    assert alerts and "RETRAIN RECOMMENDED" in bad.summary()
    hist = mon.history()
    assert hist["retrain_recommended"].to_list() == [False, True]

    small = mon.check(live.head(20).drop("y"))                     # unlabelled + tiny batch
    # no labels: performance is *estimated* (CBPE), never reported as measured
    assert small.performance["estimated"] and any(a.kind == "volume" for a in small.alerts)
    assert not any(a.kind == "performance" for a in small.alerts)
    missing = mon.check(live.drop("a"))
    assert any(a.kind == "schema" and a.severity == "critical" for a in missing.alerts)

    mon.save(tmp_path / "mon.json")
    loaded = Monitor.load(tmp_path / "mon.json", model)
    assert loaded.check(flipped).retrain_recommended
    assert json.loads((tmp_path / "mon.json").read_text())["format"] == "dataspoof.monitor"


def test_regression_monitor_uses_error_increase():
    X, y = make_regression(n_samples=900, n_features=3, noise=5, random_state=1)
    df = pl.DataFrame(X, schema=list("abc")).with_columns(pl.Series("t", y))
    model = ac.AutoML("t", **FAST).fit(df.head(500))
    mon = Monitor(model, df.slice(500, 200))
    noisy = df.tail(200).with_columns(pl.col("t") + np.random.default_rng(0).normal(0, 80, 200))
    rep = mon.check(noisy)
    assert rep.performance["metric"] == "rmse" and rep.performance["change"] > 0.1
    assert any(a.kind == "performance" for a in rep.critical)


def test_webhook_alert_failure_never_raises():
    from mlpilot.monitoring import webhook_alert
    send = webhook_alert("http://127.0.0.1:9/hook", timeout=0.5)
    rep = ac.monitoring.MonitorReport("t", 1, ac.drift_report(_frame(50, 0), _frame(50, 1)), None, None, [], False)
    with pytest.warns(RuntimeWarning, match="webhook failed"):
        send(rep, [])


def test_retrain_champion_vs_challenger(tmp_path, clf):
    model, ref, live = clf
    reg = ac.ModelRegistry(tmp_path / "reg")
    reg.register(model, "m", stage="production")
    res = ac.retrain(model, pl.concat([ref, live.head(150)]), live.tail(150), registry=reg, name="m",
                     **{k: v for k, v in FAST.items() if k != "verbose"})
    assert res.metric == "roc_auc" and res.version.version == 2
    assert res.challenger_score > 0.8 and "holdout_roc_auc" in res.version.metrics
    if res.improved:
        assert reg.get("m", version=2).stage == "staging"
    assert "Retrained" in res.summary()
    mon = Monitor(model, ref)
    report, result = ac.monitoring.retrain_if_needed(mon, live, ref, live)
    assert result is None and not report.retrain_recommended


def test_integer_ids_are_not_monitored():
    ref = pl.DataFrame({"id": np.arange(500), "x": np.random.default_rng(0).normal(size=500)})
    cur = pl.DataFrame({"id": np.arange(1000, 1200), "x": np.random.default_rng(1).normal(size=200)})
    assert ac.drift_report(ref, cur).columns["column"].to_list() == ["x"]
