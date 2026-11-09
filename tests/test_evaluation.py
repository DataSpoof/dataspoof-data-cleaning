"""Tests for the evaluation & explainability module."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression

import mlpilot as ac
from mlpilot.evaluation import cramers_v, shapley_values
from mlpilot.evaluation._slicing import bh_adjust, segments

ROOT = Path(__file__).resolve().parent.parent
FAST = dict(time_budget=60, cv=3, verbose=False, max_trials=3,
            include=["logistic_regression", "ridge", "random_forest"])


@pytest.fixture(scope="module")
def binary():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=3, random_state=0)
    rng = np.random.default_rng(0)
    df = pl.DataFrame(X, schema=[f"f{i}" for i in range(5)]).with_columns(
        pl.Series("group", rng.choice(["a", "b"], 500)), pl.Series("label", np.where(y == 1, "yes", "no")))
    model = ac.AutoML("label", **FAST).fit(df.head(350))
    return model, df.tail(150)


@pytest.fixture(scope="module")
def regression():
    X, y = make_regression(n_samples=400, n_features=4, noise=10, random_state=0)
    df = pl.DataFrame(X, schema=["a", "b", "c", "d"]).with_columns(pl.Series("y", y))
    return ac.AutoML("y", **FAST).fit(df.head(300)), df.tail(100)


# ── metrics ─────────────────────────────────────────────────────────────────

def test_classification_report(binary):
    model, test = binary
    r = ac.evaluate_model(model, test)
    assert r.positive_class == "yes" and r.n_rows == 150
    assert abs(r.metrics["accuracy"] - model.evaluate(test)["accuracy"]) < 1e-9
    assert r.metrics["roc_auc"] > 0.85 and 0 <= r.metrics["ece"] <= 1
    assert r.confusion_matrix.columns == ["actual", "pred_no", "pred_yes"]
    assert r.confusion_matrix.select(pl.exclude("actual")).sum().sum_horizontal()[0] == 150
    assert r.per_class["support"].sum() == 150
    lo, hi = r.confidence_intervals["roc_auc"]
    assert lo <= r.metrics["roc_auc"] <= hi
    assert r.thresholds.height == 19 and 0.05 <= r.best_threshold("f1") <= 0.95
    assert r.calibration["count"].sum() == 150
    strict = ac.evaluate_model(model, test, threshold=0.9)
    assert strict.metrics["recall"] <= r.metrics["recall"]
    assert json.loads(r.to_json())["metrics"]["roc_auc"] == r.metrics["roc_auc"]
    assert "roc_auc" in r.summary()


def test_regression_report(regression):
    model, test = regression
    r = model.report(test)
    assert r.metrics["r2"] > 0.8 and r.metrics["rmse"] >= r.metrics["mae"]
    assert set(r.residuals) == {"mean", "std", "skew", "p05", "p50", "p95"}
    assert r.error_by_range["count"].sum() == 100
    with pytest.raises(ValueError, match="binary"):
        r.best_threshold()


def test_sklearn_estimator_needs_target():
    X, y = make_classification(n_samples=200, n_features=3, n_informative=2, n_redundant=0, random_state=1)
    df = pl.DataFrame(X, schema=["a", "b", "c"]).with_columns(pl.Series("t", y))
    est = LogisticRegression().fit(df.drop("t").to_pandas(), y)
    with pytest.raises(ValueError, match="target"):
        ac.evaluate_model(est, df)
    assert ac.evaluate_model(est, df, target="t").metrics["accuracy"] > 0.7


# ── SHAP ────────────────────────────────────────────────────────────────────

def test_exact_shapley_matches_linear_model_closed_form():
    # For a linear model, interventional SHAP = coef * (x - mean(background)).
    rng = np.random.default_rng(0)
    X = pl.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100), "c": rng.normal(size=100)})
    y = 3 * X["a"] - 2 * X["b"] + 0.5 * X["c"]
    est = LinearRegression().fit(X.to_pandas(), y.to_numpy())
    Xp = X.to_pandas()
    vals, base, method = shapley_values(lambda f: est.predict(f)[:, None], Xp.head(5), Xp)
    expected = est.coef_ * (Xp.head(5).to_numpy() - Xp.to_numpy().mean(axis=0))
    assert method == "exact Shapley"
    assert np.allclose(vals[:, :, 0], expected, atol=1e-8)


def test_sampled_shapley_is_additive_and_close():
    rng = np.random.default_rng(1)
    X = pl.DataFrame({f"x{i}": rng.normal(size=60) for i in range(12)})
    w = np.arange(1, 13, dtype=float)
    est = LinearRegression().fit(X.to_pandas(), X.to_numpy() @ w)
    Xp = X.to_pandas()
    f = lambda frame: est.predict(frame)[:, None]           # noqa: E731
    vals, base, method = shapley_values(f, Xp.head(4), Xp.head(20), exact_max_features=8)
    assert method.startswith("permutation")
    assert np.allclose(base + vals.sum(axis=1)[:, 0], f(Xp.head(4))[:, 0])
    expected = est.coef_ * (Xp.head(4).to_numpy() - Xp.head(20).to_numpy().mean(axis=0))
    assert np.allclose(vals[:, :, 0], expected, atol=1e-6)   # linear: exact even when sampled


def test_explain_automl(binary):
    model, test = binary
    exp = model.explain(test, rows=30)
    assert exp.values.shape == (30, len(exp.features)) and exp.positive_class == "yes"
    proba = model.model_.predict_proba(exp.data)[:, 1]
    assert np.allclose(exp.predictions, proba, atol=1e-9)
    imp = exp.global_importance()
    assert abs(imp["share"].sum() - 1) < 1e-9 and imp["feature"][-1] != imp["feature"][0]
    assert exp.for_row(0).height == len(exp.features)
    assert exp.explain_row(0).startswith("P(label = yes)")
    assert exp.to_frame().shape == (30, len(exp.features))


def test_explain_multiclass_and_partial_dependence():
    X, y = make_classification(n_samples=300, n_features=4, n_informative=3, n_redundant=0, n_classes=3,
                               n_clusters_per_class=1, random_state=2)
    df = pl.DataFrame(X, schema=["a", "b", "c", "d"]).with_columns(pl.Series("k", np.array(["r", "g", "b"])[y]))
    model = ac.AutoML("k", **FAST).fit(df)
    exp = ac.explain(model, df, rows=10)
    assert exp.values.shape == (10, len(exp.features), 3) and exp.is_multiclass
    assert np.allclose(exp.predictions, model.model_.predict_proba(exp.data), atol=1e-9)
    assert exp.global_importance(class_="g").height == len(exp.features)
    pd_frame = ac.partial_dependence(model, df, exp.features[0], grid=5)
    assert pd_frame.height <= 5 and {"P(b)", "P(g)", "P(r)"} <= set(pd_frame.columns)
    with pytest.raises(ValueError, match="not a model feature"):
        ac.partial_dependence(model, df, "nope")


# ── error analysis ──────────────────────────────────────────────────────────

def test_segments():
    s = pl.Series("x", [1.0, 2.0, None] + [float(i) for i in range(3, 40)])
    seg = segments(s, bins=4)
    assert seg[-1][0] == "x is missing" and sum(m.sum() for _, m in seg) == s.len()
    cat = segments(pl.Series("c", ["a", "b", "a", None, "c"] * 10))
    assert [lab for lab, _ in cat] == ["c = a", "c = b", "c = c", "c is missing"]
    assert segments(pl.Series("id", [f"u{i}" for i in range(100)])) == []
    assert bh_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_error_analysis_finds_planted_weak_segment():
    rng = np.random.default_rng(3)
    n = 1200
    x = rng.normal(size=n)
    region = rng.choice(["north", "south", "east"], n)
    y = (x > 0).astype(int)
    noisy = (region == "east") & (rng.random(n) < 0.45)    # labels in "east" are noisy
    y = np.where(noisy, 1 - y, y)
    df = pl.DataFrame({"x": x, "region": region, "y": y})
    model = ac.AutoML("y", **FAST).fit(df.head(800))
    rep = ac.error_analysis(model, df.tail(400))
    assert rep.problem_slices["slice"][0] == "region = east"
    assert rep.problem_slices["lift"][0] > 1.5
    assert rep.worst.height > 0 and {"row", "actual", "predicted", "confidence", "region"} <= set(rep.worst.columns)
    assert rep.worst["row"].max() < 400                     # row numbers index the frame passed in
    assert rep.confusions["count"].sum() == int(rep.overall * 400 + 0.5)
    assert "region = east" in rep.summary()


def test_error_analysis_regression(regression):
    model, test = regression
    rep = model.error_analysis(test, top=5)
    assert rep.metric == "mae" and rep.worst.height == 5 and "bias" in rep.slices.columns
    assert rep.worst["error"].abs().is_sorted(descending=True)


# ── fairness ────────────────────────────────────────────────────────────────

def _biased_model():
    rng = np.random.default_rng(5)
    n = 1500
    sex = rng.choice(["f", "m"], n)
    skill = rng.normal(size=n)
    hired = ((skill + np.where(sex == "m", 1.2, 0.0) + rng.normal(0, 0.5, n)) > 0.6).astype(int)
    zip_code = np.where(rng.random(n) < 0.9, np.where(sex == "m", "z1", "z2"), "z3")
    df = pl.DataFrame({"skill": skill, "sex": sex, "zip": zip_code, "hired": hired})
    return ac.AutoML("hired", **FAST).fit(df.head(1000)), df.tail(500)


def test_fairness_flags_disparity_and_proxy():
    model, test = _biased_model()
    rep = model.fairness(test, "sex", pos_label=1)
    res = rep["sex"]
    assert not rep.passed and res.used_by_model
    rates = dict(zip(res.groups["group"], res.groups["selection_rate"]))
    assert rates["m"] > rates["f"]
    assert res.disparities["disparate_impact_ratio"] < 0.8
    assert any("four-fifths" in f for f in res.flags)
    proxy = rep.proxies.filter(pl.col("feature") == "zip")
    assert proxy["flag"][0] and proxy["cramers_v"][0] > 0.5
    assert not rep.proxies.filter(pl.col("feature") == "skill")["flag"][0]
    assert json.loads(rep.to_json())["passed"] is False


def test_fairness_fair_model_passes_and_intersections(binary):
    model, test = binary
    rep = ac.fairness_report(model, test, ["group", "f0"], min_group_size=10, limits={
        k: (0.0 if k == "disparate_impact_ratio" else 9.9) for k in ac.evaluation.DEFAULT_LIMITS})
    assert rep.passed
    assert [r.attribute for r in rep.results] == ["group", "f0", "group & f0"]
    assert rep["f0"].groups.height == 4                     # numeric attribute -> 4 quantile bands
    with pytest.raises(ValueError, match="not in the data"):
        ac.fairness_report(model, test, "nope")


def test_fairness_regression(regression):
    model, test = regression
    test = test.with_columns(pl.Series("site", ["x", "y"] * 50))
    res = ac.fairness_report(model, test, "site")["site"]
    assert {"mae", "bias", "mean_prediction"} <= set(res.groups.columns)
    assert "error_ratio" in res.disparities


def test_cramers_v():
    a = np.array(["x", "y"] * 50)
    assert cramers_v(a, a) > 0.95
    rng = np.random.default_rng(0)
    assert cramers_v(a, rng.choice(["p", "q", "r"], 100)) < 0.2


# ── model card ──────────────────────────────────────────────────────────────

def test_model_card_full(tmp_path):
    model, test = _biased_model()
    card = model.model_card(test, sensitive=["sex"], pos_label=1, name="Hiring screen",
                            intended_use="Demo only.", owners="Data team")
    d = card.to_dict()
    assert d["details"]["name"] == "Hiring screen" and d["details"]["owners"] == ["Data team"]
    assert d["evaluation"]["metrics"]["roc_auc"] > 0.8
    assert d["explainability"]["top_features"][0]["feature"] in ("skill", "sex", "zip")
    assert d["fairness"]["passed"] is False and d["data_quality"]["grade"] in "ABCDF"
    assert any("Fairness (sex)" in e for e in card.ethical_considerations)
    assert any("proxy" in e for e in card.ethical_considerations)
    md = card.to_markdown()
    for heading in ("## Model details", "## Intended use", "## Held-out evaluation", "## Explainability",
                    "## Error analysis", "## Fairness", "## Limitations", "## Ethical considerations"):
        assert heading in md
    html_text = card.save(tmp_path / "card.html").read_text(encoding="utf-8")
    assert html_text.startswith("<!doctype html>") and "Hiring screen" in html_text and "<script" not in html_text
    assert json.loads(card.save(tmp_path / "card.json").read_text())["title"] == "Hiring screen"
    with pytest.raises(ValueError):
        card.save(tmp_path / "card.pdf")


def test_model_card_without_test_data_suggests_sensitive_columns(binary):
    model, _ = binary
    card = ac.model_card(model)
    assert card.data["evaluation"] is None and card.data["explainability"] is not None
    assert any("Not evaluated on held-out data" in x for x in card.limitations)
    assert any("No fairness review" in x for x in card.ethical_considerations)

    df = pl.DataFrame({"gender": ["m", "f"] * 40, "x": list(range(80)), "y": [0, 1, 1, 0] * 20})
    m2 = ac.AutoML("y", **FAST).fit(df)
    assert any("gender" in x for x in ac.model_card(m2).ethical_considerations)


def test_model_card_html_escapes_user_text():
    X, y = make_classification(n_samples=120, n_features=3, n_informative=2, n_redundant=0, random_state=0)
    df = pl.DataFrame(X, schema=["a", "b", "c"]).with_columns(pl.Series("t", y))
    model = ac.AutoML("t", **FAST).fit(df)
    page = ac.model_card(model, name="<script>alert(1)</script>").to_html()
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_titanic_card():
    path = ROOT / "titanic.csv"
    if not path.exists():
        pytest.skip("titanic.csv not available")
    df = pl.read_csv(path, infer_schema_length=10000)
    model = ac.AutoML("Survived", **FAST).fit(df.head(700))
    card = model.model_card(df.tail(191), sensitive=["Sex"])
    assert card.data["evaluation"]["metrics"]["roc_auc"] > 0.8
    assert not card.data["fairness"]["passed"]
