"""Concept drift before outcomes arrive (CBPE, direct loss estimation, domain classifier), and
the real Evidently integration."""

import importlib.util

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

import mlpilot as ac
from mlpilot.monitoring import PerformanceEstimator, domain_classifier_drift

FAST = dict(time_budget=30, cv=3, verbose=False, max_trials=1)


def _clf_data(n, rng, shift=0.0, spread=2.0):
    x1 = rng.normal(shift, spread, n)
    x2 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(1.5 * x1 + 0.5 * x2)))
    return pl.DataFrame({"x1": x1, "x2": x2, "y": rng.binomial(1, p)})


@pytest.fixture(scope="module")
def clf():
    rng = np.random.default_rng(0)
    train, ref = _clf_data(3000, rng), _clf_data(2000, rng)
    model = ac.AutoML("y", include=["logistic_regression"], **FAST).fit(train)
    return model, ref, rng


def test_cbpe_tracks_accuracy_and_auc_without_labels(clf):
    model, ref, rng = clf
    est = PerformanceEstimator(model).fit(ref)
    same = _clf_data(2000, rng)
    hard = _clf_data(2000, rng, spread=0.6)                 # inputs moved toward the decision boundary
    for batch in (same, hard):
        e = est.estimate(batch.drop("y"))
        truth = float((model.predict(batch).cast(pl.Int64) == batch["y"]).mean())
        assert abs(e["estimated"] - truth) < 0.03, (e, truth)
        auc = roc_auc_score(batch["y"], model.predict_proba(batch)[:, 1])
        assert abs(e["roc_auc"]["estimated"] - auc) < 0.04
        assert e["low"] <= e["estimated"] <= e["high"]
    assert est.estimate(hard.drop("y"))["change"] < -0.05


def test_monitor_warns_before_labels_arrive(clf, tmp_path):
    model, ref, rng = clf
    mon = ac.Monitor(model, ref, log_path=tmp_path / "m.jsonl")
    quiet = mon.check(_clf_data(1500, rng).drop("y"))
    assert quiet.performance["estimated"] and not [a for a in quiet.alerts if a.kind == "estimated_performance"]
    hard = mon.check(_clf_data(1500, rng, spread=0.6).drop("y"))
    alerts = [a for a in hard.alerts if a.kind == "estimated_performance"]
    assert alerts and alerts[0].severity == "warning" and "no labels yet" in alerts[0].message
    assert hard.performance["metric"] == "roc_auc"
    labelled = mon.check(_clf_data(1500, rng, spread=0.6))         # with labels: measured, not estimated
    assert not labelled.performance.get("estimated")
    loaded = ac.Monitor.load(mon.save(tmp_path / "mon.json"), model)
    again = loaded.check(_clf_data(1500, rng, spread=0.6).drop("y"))
    assert any(a.kind == "estimated_performance" for a in again.alerts)
    assert mon.history()["metric"].to_list()[:2] == ["estimated_roc_auc", "estimated_roc_auc"]


def test_direct_loss_estimation_for_regression():
    rng = np.random.default_rng(4)

    def data(n, lo, hi):
        x = rng.uniform(lo, hi, n)
        return pl.DataFrame({"x": x, "y": 3 * x + rng.normal(0, 0.2 + 0.5 * x, n)})   # noise grows with x
    model = ac.AutoML("y", include=["ridge"], **FAST).fit(data(3000, 0, 4))
    est = PerformanceEstimator(model).fit(data(2000, 0, 4))
    shifted = data(2000, 3, 4)
    e = est.estimate(shifted.drop("y"))
    truth = float(np.mean(np.abs(model.predict(shifted).to_numpy() - shifted["y"].to_numpy())))
    assert e["metric"] == "mae" and e["estimated"] > e["reference"] * 1.3
    assert abs(e["estimated"] - truth) / truth < 0.15


def test_domain_classifier_drift():
    rng = np.random.default_rng(2)
    ref = pl.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000), "c": rng.choice(["x", "y"], 2000)})
    same = pl.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000), "c": rng.choice(["x", "y"], 2000)})
    moved = same.with_columns(pl.col("b") * 0.4 + 0.3)              # joint shift, mostly in b
    assert domain_classifier_drift(ref, same)["auc"] < 0.56
    d = domain_classifier_drift(ref, moved)
    assert d["drift"] and d["auc"] > 0.7 and d["top_features"][0][0] == "b"


@pytest.mark.skipif(importlib.util.find_spec("evidently") is None, reason="needs evidently")
def test_evidently_report_agrees_with_ours(clf, tmp_path):
    from mlpilot.integrations.evidently import evidently_report
    model, ref, rng = clf
    cur = _clf_data(2000, rng).with_columns(pl.col("x1") + 1.5)
    res = evidently_report(ref, cur, model=model, html=tmp_path / "drift.html")
    assert res.drifted_columns == ["x1"] and res.drift_share == 0.5
    cmp = res.compare(ac.drift_report(ref.drop("y"), cur.drop("y")))
    assert cmp["agreement"] == 1.0 and cmp["both"] == ["x1"]
    assert any("Accuracy" in k for k in res.metrics), res.metrics
    assert (tmp_path / "drift.html").stat().st_size > 10_000
    plain = evidently_report(ref.drop("y"), cur.drop("y"))
    assert plain.drifted_columns == ["x1"] and not plain.metrics
