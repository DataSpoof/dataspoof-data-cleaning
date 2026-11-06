"""Tests for the AutoML module."""

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_blobs, make_classification, make_regression

import automatedcleaning as ac

ROOT = Path(__file__).resolve().parent.parent
FAST = dict(time_budget=60, cv=3, verbose=False, max_trials=5,
            include=["logistic_regression", "ridge", "random_forest", "hist_gradient_boosting"])


def _frame(X, y, target):
    return pl.DataFrame(X, schema=[f"f{i}" for i in range(X.shape[1])]).with_columns(pl.Series(target, y))


# ── task detection ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("values,task", [
    ([0, 1] * 10, "binary"), (["yes", "no"] * 10, "binary"), ([True, False] * 10, "binary"),
    (["a", "b", "c"] * 10, "multiclass"), ([1, 2, 3, 4, 5] * 6, "multiclass"),
    ([i / 3 for i in range(30)], "regression"), (list(range(30)), "regression"),
])
def test_detect_task(values, task):
    df = pl.DataFrame({"y": values, "x": list(range(len(values)))})
    assert ac.detect_task(df, "y", time_col=None).task == task


def test_detect_clustering_and_time_series():
    days = pl.date_range(dt.date(2024, 1, 1), dt.date(2024, 1, 30), eager=True)
    df = pl.DataFrame({"day": days, "y": [float(i) for i in range(30)]})
    assert ac.detect_task(df).task == "clustering"
    info = ac.detect_task(df, "y")
    assert info.time_series and info.time_col == "day"
    assert not ac.detect_task(df.sample(fraction=1.0, shuffle=True, seed=3), "y").time_series


def test_optional_boosters_are_used_only_when_installed(monkeypatch):
    from automatedcleaning.automl import models
    monkeypatch.setattr(models, "_has", lambda name: False)
    names = ac.available_models("binary")
    assert "lightgbm" not in names and "xgboost" not in names
    assert {"baseline", "logistic_regression", "random_forest", "hist_gradient_boosting"} <= set(names)
    assert "ridge" in ac.available_models("regression")


# ── supervised ───────────────────────────────────────────────────────────────

def test_binary_end_to_end_with_save_load(tmp_path):
    X, y = make_classification(n_samples=400, n_features=6, n_informative=4, random_state=0)
    df = _frame(X, np.where(y == 1, "churn", "stay"), "label")
    train, test = df.head(300), df.tail(100)
    model = ac.AutoML("label", **FAST).fit(train)
    assert model.task_.task == "binary" and model.metric_ == "roc_auc"
    board = model.leaderboard_
    assert "baseline" in board["model"].to_list() and "ensemble" in board["model"].to_list()
    assert board["score"][0] >= 0.85 and model.best_["model"] != "baseline"
    assert len([t for t in model.trials_ if t["model"] not in ("baseline", "ensemble")]) <= 5

    pred = model.predict(test)
    assert set(pred.unique().to_list()) <= {"churn", "stay"}
    proba = model.predict_proba(test)
    assert proba.columns == ["proba_churn", "proba_stay"]
    assert np.allclose(proba.sum_horizontal().to_numpy(), 1.0)
    assert model.evaluate(test)["roc_auc"] > 0.85
    assert model.feature_importance().height == len(model.feature_kinds_)
    assert "Best:" in model.summary()

    path = model.save(tmp_path / "model")
    loaded = ac.AutoML.load(path)
    assert loaded.predict(test).equals(pred)
    (path / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        ac.AutoML.load(path)


def test_regression():
    X, y = make_regression(n_samples=300, n_features=5, noise=5.0, random_state=0)
    df = _frame(X, y, "target")
    model = ac.AutoML("target", **FAST).fit(df.head(240))
    assert model.task_.task == "regression" and model.metric_ == "neg_root_mean_squared_error"
    metrics = model.evaluate(df.tail(60))
    assert metrics["r2"] > 0.8 and {"rmse", "mae", "r2"} <= set(metrics)
    assert model.leaderboard_["rmse"].drop_nulls().min() > 0      # shown as a positive error
    assert model.predict(df.tail(5)).dtype == pl.Float64


def test_multiclass():
    X, y = make_classification(n_samples=450, n_features=6, n_informative=4, n_classes=3,
                               n_clusters_per_class=1, random_state=1)
    df = _frame(X, np.array(["red", "green", "blue"])[y], "color")
    model = ac.AutoML("color", **FAST).fit(df.head(350))
    assert model.task_.task == "multiclass" and model.task_.n_classes == 3 and model.metric_ == "f1_macro"
    assert model.evaluate(df.tail(100))["accuracy"] > 0.7
    assert model.predict_proba(df.tail(10)).width == 3


def test_task_override_and_no_ensemble():
    X, y = make_regression(n_samples=200, n_features=3, random_state=2)
    df = _frame(X, np.digitize(y, [-50, 0, 50]), "bucket")        # 4 whole-number classes
    model = ac.AutoML("bucket", task="regression", ensemble=False, **FAST).fit(df)
    assert model.task_.task == "regression"
    assert "ensemble" not in [t["model"] for t in model.trials_]


def test_leaky_columns_are_dropped_and_ignored_at_prediction():
    rng = np.random.default_rng(1)
    n = 300
    y = rng.integers(0, 2, n)
    df = pl.DataFrame({"a": rng.normal(size=n) + y, "b": rng.normal(size=n),
                       "refund_issued": y * 10.0, "churn": y})
    model = ac.AutoML("churn", **FAST).fit(df)
    assert model.dropped_leaky_ == ["refund_issued"]
    assert "refund_issued" not in model.feature_kinds_
    assert model.predict(df).len() == n                            # leaky column present: ignored
    with pytest.raises(ac.LeakageError):
        ac.AutoML("churn", on_leakage="raise", **FAST).fit(df)


def test_time_series_uses_time_ordered_cv():
    n = 200
    days = pl.date_range(dt.date(2022, 1, 1), dt.date(2022, 1, 1) + dt.timedelta(days=n - 1), eager=True)
    t = np.arange(n)
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"day": days, "promo": rng.integers(0, 2, n),
                       "sales": t * 0.5 + 10 * np.sin(t / 7) + rng.normal(0, 1, n)})
    model = ac.AutoML("sales", **FAST).fit(df)
    assert model.task_.time_series and model.task_.time_col == "day"
    assert "time-ordered" in model.cv_description_


def test_every_class_needs_two_examples():
    df = pl.DataFrame({"x": list(range(12)), "y": ["a"] * 6 + ["b"] * 5 + ["c"]})
    with pytest.raises(ValueError, match="only 1 example"):
        ac.AutoML("y", **FAST).fit(df)


# ── clustering ───────────────────────────────────────────────────────────────

def test_autocluster_finds_the_blobs(tmp_path):
    X, _ = make_blobs(n_samples=300, centers=3, n_features=4, cluster_std=0.6, random_state=0)
    df = pl.DataFrame(X, schema=["a", "b", "c", "d"])
    model = ac.AutoCluster(k_range=(2, 6), verbose=False).fit(df)
    assert model.best_["k"] == 3
    assert model.labels_.len() == 300 and model.labels_.n_unique() == 3
    profiles = model.cluster_profiles(df)
    assert profiles.height == 3 and profiles["size"].sum() == 300
    loaded = ac.AutoCluster.load(model.save(tmp_path / "clusters"))
    assert loaded.predict(df).equals(model.labels_)


def test_automl_dispatches_on_target():
    X, _ = make_blobs(n_samples=120, centers=2, n_features=3, random_state=0)
    df = pl.DataFrame(X, schema=["a", "b", "c"])
    assert isinstance(ac.automl(df, k_range=(2, 3), verbose=False), ac.AutoCluster)
    labelled = df.with_columns(pl.Series("y", [0, 1] * 60))
    assert isinstance(ac.automl(labelled, target="y", **FAST), ac.AutoML)


# ── real data ────────────────────────────────────────────────────────────────

def test_titanic_end_to_end():
    path = ROOT / "titanic.csv"
    if not path.exists():
        pytest.skip("titanic.csv not available")
    df = pl.read_csv(path, infer_schema_length=10000)
    model = ac.AutoML("Survived", **FAST).fit(df.head(700))
    assert model.evaluate(df.tail(191))["roc_auc"] > 0.8
