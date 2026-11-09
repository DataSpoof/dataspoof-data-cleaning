"""Adversarial robustness tests and poisoned-training-data detection, on planted attacks."""

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression

import automatedcleaning as ac
from automatedcleaning.poisoning import poisoning_report
from automatedcleaning.robustness import robustness_report

FAST = dict(time_budget=30, cv=3, verbose=False, max_trials=1)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n = 1500
    x1, x2, noise = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    plan = rng.choice(["basic", "pro", "team"], n)
    y = ((2.5 * x1 + 0.8 * x2 + rng.normal(0, 0.3, n)) > 0).astype(int)
    return pl.DataFrame({"x1": x1, "x2": x2, "noise": noise, "plan": plan, "y": y})


@pytest.fixture(scope="module")
def model(data):
    return ac.AutoML("y", include=["logistic_regression"], **FAST).fit(data.head(1000))


# ── robustness ──────────────────────────────────────────────────────────────
def test_robustness_report_on_automl(model, data):
    rep = robustness_report(model, data.tail(500), n=120, budget=3.0, step=0.05)
    assert set(rep.noise) == {0.01, 0.05, 0.1}
    assert rep.noise[0.01] <= rep.noise[0.1] < 0.2                     # more noise -> more flips, still small
    a = rep.attack
    assert a["attacked"] == 120 and a["success_rate"] > 0.9              # a linear model is always attackable
    assert a["most_used_feature"] == "x1"                                # the attack exploits the strongest feature
    assert 0 < a["median_distance"] < 3.0 and a["within"]["1"] >= a["within"]["0.1"]
    feats = rep.features
    assert feats["feature"][0] == "x1"
    assert feats.filter(pl.col("feature") == "noise")["flip_rate"][0] < 0.05
    assert rep.extreme == {"crashed": [], "non_finite": []}
    assert "adversarial" in rep.summary() and isinstance(rep.passed(), bool)
    assert not rep.passed(min_median_distance=10.0)


def test_robustness_with_sklearn_and_regression(data):
    X = data.select("x1", "x2", "noise")
    clf = LogisticRegression().fit(X.to_pandas(), data["y"].to_numpy())
    rep = robustness_report(clf, data.select("x1", "x2", "noise", "y"), target="y", n=60)
    assert rep.task == "binary" and rep.attack["success_rate"] > 0.9
    reg = ac.AutoML("x1", include=["ridge"], **FAST).fit(data.select("x1", "x2", "noise").head(800)
                                                          .with_columns(x1=pl.col("x2") * 3 + pl.col("noise") * 0.1))
    rr = robustness_report(reg, data.select("x2", "noise").head(200), n=50, budget=1.0)
    assert rr.task == "regression" and rr.features["feature"][0] == "x2"


# ── poisoning ───────────────────────────────────────────────────────────────
def test_label_flipping_is_detected(data):
    rng = np.random.default_rng(7)
    flipped = rng.choice(data.height, 60, replace=False)
    poisoned = data.with_columns(pl.when(pl.int_range(pl.len()).is_in(flipped.tolist())).then(1 - pl.col("y"))
                                 .otherwise(pl.col("y")).alias("y"))
    rep = poisoning_report(poisoned, "y")
    flagged = set(rep.suspicious.filter(pl.col("reasons").str.contains("mislabelled"))["row"].to_list())
    recall = len(flagged & set(flipped.tolist())) / len(flipped)
    precision = len(flagged & set(flipped.tolist())) / max(len(flagged), 1)
    assert recall >= 0.7 and precision >= 0.5, (recall, precision)
    cleaned = rep.clean(poisoned)
    assert cleaned.height == poisoned.height - rep.suspicious.height


def test_backdoor_triggers_and_conflicts(data):
    rng = np.random.default_rng(3)
    base = data.with_columns(code=pl.Series(rng.choice([f"c{i}" for i in range(60)], data.height)),
                             amount=pl.Series(rng.normal(100, 20, data.height)))
    trig = rng.choice(base.height, 45, replace=False).tolist()
    trig2 = rng.choice(list(set(range(base.height)) - set(trig)), 45, replace=False).tolist()
    idx = pl.int_range(pl.len())
    poisoned = base.with_columns(
        code=pl.when(idx.is_in(trig)).then(pl.lit("ZX9")).otherwise(pl.col("code")),
        amount=pl.when(idx.is_in(trig2)).then(pl.lit(777.0)).otherwise(pl.col("amount")),
        y=pl.when(idx.is_in(trig + trig2)).then(pl.lit(1)).otherwise(pl.col("y")))
    dup = poisoned.head(5).with_columns(y=1 - pl.col("y"))               # same features, opposite labels
    poisoned = pl.concat([poisoned, dup])
    rep = poisoning_report(poisoned, "y")
    found = {(r["feature"], r["value"]) for r in rep.triggers.iter_rows(named=True)}
    assert ("code", "ZX9") in found and ("amount", "777.0") in found, rep.triggers
    assert rep.conflicting_duplicates == 10
    assert "possible trigger" in rep.summary()
    trig_rows = set(rep.suspicious.filter(pl.col("reasons").str.contains("trigger"))["row"].to_list())
    assert set(trig) <= trig_rows


def test_clean_data_has_few_false_alarms(data):
    rep = poisoning_report(data, "y")
    assert rep.triggers.height == 0 and rep.conflicting_duplicates == 0
    assert rep.suspicious.height / data.height < 0.06


def test_regression_target_outliers(data):
    df = data.select("x1", "x2").with_columns(t=pl.col("x1") * 2 + pl.col("x2"))
    bad = df.with_columns(pl.when(pl.int_range(pl.len()) < 15).then(pl.lit(80.0)).otherwise(pl.col("t")).alias("t"))
    rep = poisoning_report(bad, "t")
    assert rep.task == "regression" and set(range(15)) <= set(rep.suspicious["row"].to_list())
