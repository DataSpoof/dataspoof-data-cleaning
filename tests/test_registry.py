"""Tests for experiment tracking and the model registry."""

import json
import threading

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_classification

import mlpilot as ac
from mlpilot.registry import dataset_fingerprint

FAST = dict(time_budget=30, cv=3, verbose=False, max_trials=2, include=["logistic_regression", "random_forest"])


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=300, n_features=4, n_informative=3, n_redundant=0, random_state=0)
    df = pl.DataFrame(X, schema=list("abcd")).with_columns(pl.Series("y", y))
    return df.head(220), df.tail(80)


@pytest.fixture(scope="module")
def model(data):
    return ac.AutoML("y", **FAST).fit(data[0])


def test_dataset_fingerprint_changes_with_content():
    a = pl.DataFrame({"x": [1, 2, 3], "s": ["a", "b", None]})
    fa, fb = dataset_fingerprint(a), dataset_fingerprint(a.clone())
    assert fa["content_hash"] == fb["content_hash"] and fa["rows"] == 3 and fa["null_cells"] == 1
    changed = dataset_fingerprint(a.with_columns(pl.Series("x", [1, 2, 4])))
    assert changed["content_hash"] != fa["content_hash"] and changed["schema_hash"] == fa["schema_hash"]


def test_tracker_run_lifecycle(tmp_path, data, model):
    train, test = data
    tr = ac.Tracker(tmp_path / "runs", experiment="exp")
    with tr.start_run("first", tags={"team": "ds"}) as run:
        run.log_automl(model, train, test)
        run.log_metric("loss", 0.5, step=1)
        run.log_metric("loss", 0.3, step=2)
    assert run.status == "finished" and run.metrics["loss"] == 0.3
    assert run.metric_history("loss")["value"].to_list() == [0.5, 0.3]
    assert {"model", "leaderboard.csv", "evaluation.json", "model_card.html"} <= set(run.to_dict()["artifacts"])
    assert [d["name"] for d in run.datasets] == ["train", "test"]
    assert run.load_model().predict(test).len() == test.height
    with pytest.raises(RuntimeError, match="ended"):
        run.log_param("late", 1)

    with pytest.raises(ZeroDivisionError):
        with tr.start_run("broken") as bad:
            bad.log_params({"lr": 0.1})
            1 / 0
    assert tr.get_run(bad.id[:8]).status == "failed" and "ZeroDivisionError" in bad.to_dict()["error"]

    runs = tr.runs()
    assert runs.height == 2 and runs["name"].to_list() == ["broken", "first"]
    assert tr.best_run("test_roc_auc").id == run.id
    cmp = tr.compare([run.id, bad.id])
    assert cmp.width == 3 and "param.lr" in cmp["field"].to_list()
    assert tr.experiments() == ["exp"]


def test_registry_versions_promotion_and_rollback(tmp_path, data, model):
    train, test = data
    reg = ac.ModelRegistry(tmp_path / "reg")
    tr = ac.Tracker(tmp_path / "runs")
    with tr.start_run() as run:
        run.log_automl(model, train, test, card=False)
    v1 = reg.register(model, "churn", run=run, description="first")
    v2 = reg.register(model, "churn", metrics={"test_roc_auc": 0.5})
    assert (v1.version, v2.version) == (1, 2) and v1.stage == "dev" and v1.run_id == run.id
    assert v1.dataset["name"] == "train" and "cv_roc_auc" in v1.metrics

    reg.promote("churn", 1, "production", reason="launch", require={"test_roc_auc": 0.5})
    with pytest.raises(ac.PromotionError, match="needs >= 0.99"):
        reg.promote("churn", 2, "production", require={"test_roc_auc": 0.99})
    with pytest.raises(ac.PromotionError, match="worse than production"):
        reg.promote("churn", 2, "production", must_beat_production="test_roc_auc")
    assert reg.get("churn").version == 1                         # unchanged after failed gates

    reg.promote("churn", 2, "production", reason="forced")
    assert reg.get("churn", version=1).stage == "archived" and reg.get("churn").version == 2
    reg.rollback("churn")
    assert reg.get("churn").version == 1 and reg.get("churn", version=2).stage == "archived"

    hist = reg.history("churn")
    assert hist.filter(pl.col("to") == "production").height == 3
    assert reg.models().row(0, named=True)["production"] == 1
    assert reg.versions("churn").height == 2
    assert reg.load("churn", stage="production").predict(test).len() == test.height


def test_registry_rejects_bad_names_and_tampering(tmp_path, model):
    reg = ac.ModelRegistry(tmp_path)
    for bad in ("../evil", "a/b", "", ".."):
        with pytest.raises(ValueError):
            reg.register(model, bad)
    v = reg.register(model, "m")
    (v.model_path / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        reg.load("m")


def test_concurrent_registration_gets_distinct_versions(tmp_path, model):
    reg = ac.ModelRegistry(tmp_path)
    path = model.save(tmp_path / "saved")
    errors = []

    def go():
        try:
            reg.register(path, "par")
        except Exception as exc:          # pragma: no cover - reported below
            errors.append(exc)
    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and sorted(reg.versions("par")["version"].to_list()) == [1, 2, 3, 4]
