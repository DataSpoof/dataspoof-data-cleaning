"""Statistical testing toolkit."""

import numpy as np
import polars as pl
import pytest
from scipy import stats as st

from mlpilot import stats


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


def test_compare_two_groups_picks_the_right_test(rng):
    df = pl.DataFrame({"g": ["a"] * 200 + ["b"] * 200,
                       "v": np.concatenate([rng.normal(10, 2, 200), rng.normal(11, 2, 200)])})
    r = stats.compare_groups(df, "v", "g")
    assert r.test == "Welch's t-test" and r.significant and r.assumptions["chosen"] == "parametric"
    assert r.effect_name == "hedges_g" and -0.75 < r.effect_size < -0.25          # true d = -0.5
    assert r.ci[0] < -1 + 0.5 and r.ci[1] > -1 - 0.5 and r.ci[0] < r.ci[1] < 0
    skew = pl.DataFrame({"g": ["a"] * 60 + ["b"] * 60,
                         "v": np.concatenate([rng.exponential(1, 60) ** 3, rng.exponential(2, 60) ** 3])})
    r2 = stats.compare_groups(skew, "v", "g")
    assert r2.test == "Mann-Whitney U" and r2.assumptions["chosen"] == "rank-based" and r2.effect_size < 0


def test_three_groups_anova_and_kruskal(rng):
    df = pl.DataFrame({"g": ["x"] * 80 + ["y"] * 80 + ["z"] * 80,
                       "v": np.concatenate([rng.normal(0, 1, 80), rng.normal(0, 1, 80), rng.normal(1, 1, 80)])})
    r = stats.compare_groups(df, "v", "g")
    assert r.test == "one-way ANOVA" and r.significant and r.effect_name == "omega_squared"
    pairs = {(p["a"], p["b"]): p["p_value"] for p in r.details["pairwise_tukey"]}
    assert pairs[("x", "z")] < 0.01 and pairs[("x", "y")] > 0.05
    k = stats.kruskal(*[df.filter(pl.col("g") == g)["v"] for g in "xyz"], names=list("xyz"))
    assert k.significant and all("p_value_holm" in p for p in k.details["pairwise_mann_whitney"])
    hetero = stats.anova(rng.normal(0, 1, 50), rng.normal(0, 5, 50), rng.normal(0, 10, 50))
    assert "Alexander-Govern" in hetero.test


def test_chi_square_and_fisher(rng):
    plan = rng.choice(["monthly", "annual"], 1000)
    churn = np.where(plan == "monthly", rng.random(1000) < 0.4, rng.random(1000) < 0.1)
    df = pl.DataFrame({"plan": plan, "churn": churn})
    r = stats.chi_square(df, "plan", "churn")
    assert r.test.startswith("chi-square") and r.significant and 0.25 < r.effect_size < 0.45
    small = pl.DataFrame({"a": ["x"] * 6 + ["y"] * 6, "b": [1] * 5 + [0] + [0] * 5 + [1]})
    f = stats.chi_square(small, "a", "b")
    assert f.test == "Fisher's exact test"
    assert f.p_value == pytest.approx(st.fisher_exact([[1, 5], [5, 1]])[1], rel=1e-9)


def test_correlation_with_ci(rng):
    x = rng.normal(size=500)
    y = 0.6 * x + 0.8 * rng.normal(size=500)                                  # true r = 0.6
    r = stats.correlation_test(pl.DataFrame({"x": x, "y": y}), "x", "y")
    assert r.ci[0] < 0.6 < r.ci[1] and r.significant and "not causation" in r.interpretation
    s = stats.correlation_test(pl.DataFrame({"x": x, "y": np.exp(x)}), "x", "y", method="spearman")
    assert s.statistic == pytest.approx(1.0)


def test_proportions_ab_test_and_power():
    r = stats.proportion_test(120, 1000, 100, 1000)
    assert r.details["difference"] == pytest.approx(0.02) and not r.significant          # z ~ 1.43
    assert r.statistic == pytest.approx(1.4285, abs=1e-3)
    rng = np.random.default_rng(3)
    df = pl.DataFrame({"variant": ["control"] * 5000 + ["new"] * 5000,
                       "converted": np.concatenate([rng.random(5000) < 0.10, rng.random(5000) < 0.12]).astype(int)})
    ab = stats.ab_test(df, "variant", "converted", control="control")
    assert ab.significant and ab.details["relative_lift"] > 0.1 and "relative lift" in ab.interpretation
    assert stats.required_sample_size(baseline_rate=0.10, mde=0.02) == pytest.approx(3841, abs=5)
    assert stats.required_sample_size(std=10, mde=2) == pytest.approx(393, abs=2)


def test_paired_and_normality(rng):
    before = rng.normal(70, 10, 40)
    after = before + rng.normal(2, 3, 40)
    r = stats.paired_test(before, after)
    assert r.test == "paired t-test" and r.significant and r.details["mean_change"] > 0
    assert not stats.normality_test(rng.normal(size=300)).significant
    assert stats.normality_test(rng.exponential(size=300)).significant


def test_adjust_pvalues_matches_reference():
    p = [0.01, 0.04, 0.03, 0.005]
    assert stats.adjust_pvalues(p, "bonferroni") == pytest.approx([0.04, 0.16, 0.12, 0.02])
    assert stats.adjust_pvalues(p, "holm") == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert stats.adjust_pvalues(p, "bh") == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_test_all_finds_real_associations(rng):
    n = 600
    region = rng.choice(["n", "s", "e"], n)
    spend = rng.normal(50, 10, n) + np.where(region == "e", 15, 0)
    df = pl.DataFrame({"region": region, "tenure": rng.normal(10, 3, n), "noise": rng.normal(size=n),
                       "spend": spend + 2 * rng.normal(size=n)})
    df = df.with_columns((pl.col("spend") + pl.col("tenure") * 3).alias("spend"))
    res = stats.test_all(df, "spend")
    sig = dict(zip(res["column"], res["significant"]))
    assert sig["region"] and sig["tenure"] and not sig["noise"]
    assert "p_adjusted" in res.columns
