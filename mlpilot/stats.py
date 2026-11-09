"""Statistical testing toolkit: the right test for the data, with effect sizes, confidence
intervals, assumption checks and a plain-English reading — not just a p-value.

* ``compare_groups(df, value="revenue", group="region")``  picks Welch's t-test / Mann-Whitney
  (2 groups) or ANOVA / Welch-type ANOVA / Kruskal-Wallis (3+), checks normality and equal
  variances first, adds effect sizes and post-hoc pairwise comparisons
* ``ttest``, ``mann_whitney``, ``anova``, ``kruskal``, ``paired_test``
* ``chi_square(df, "plan", "churned")``   contingency test with Cramér's V; Fisher's exact for small 2x2
* ``correlation_test(df, "x", "y")``      Pearson / Spearman / Kendall with a confidence interval
* ``normality_test``, ``proportion_test`` (two-proportion z-test), ``ab_test`` (conversion or
  continuous metric: lift, CI, p-value), ``required_sample_size`` (power analysis)
* ``adjust_pvalues`` (Holm / Benjamini-Hochberg / Bonferroni) and ``test_all(df, target)`` — every
  column's association with a target, corrected for multiple testing
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import polars as pl
from scipy import stats as st

from ._detect import is_num


@dataclasses.dataclass
class TestResult:
    test: str
    statistic: float
    p_value: float
    effect_size: float | None = None
    effect_name: str | None = None
    ci: tuple | None = None
    n: int | None = None
    alpha: float = 0.05
    assumptions: dict = dataclasses.field(default_factory=dict)
    details: dict = dataclasses.field(default_factory=dict)
    interpretation: str = ""

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def summary(self) -> str:
        lines = [f"{self.test}: statistic {self.statistic:.4g}, p = {self.p_value:.4g}"
                 + (f", {self.effect_name} = {self.effect_size:.3f} ({effect_label(self.effect_name, self.effect_size)})"
                    if self.effect_size is not None else "")
                 + (f", 95% CI [{self.ci[0]:.4g}, {self.ci[1]:.4g}]" if self.ci else "")]
        if self.assumptions:
            lines.append("  assumptions: " + ", ".join(f"{k}: {v}" for k, v in self.assumptions.items()))
        if self.interpretation:
            lines.append(f"  -> {self.interpretation}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["significant"] = self.significant
        return d

    def __repr__(self) -> str:
        return self.summary()


def effect_label(name: str | None, value: float | None) -> str:
    """Conventional size labels (Cohen 1988 and common practice)."""
    if value is None or name is None:
        return ""
    v = abs(value)
    cuts = {"cohen_d": (0.2, 0.5, 0.8), "hedges_g": (0.2, 0.5, 0.8), "cohen_h": (0.2, 0.5, 0.8), "rank_biserial": (0.1, 0.3, 0.5),
            "eta_squared": (0.01, 0.06, 0.14), "epsilon_squared": (0.01, 0.06, 0.14), "omega_squared": (0.01, 0.06, 0.14),
            "cramers_v": (0.1, 0.3, 0.5), "r": (0.1, 0.3, 0.5), "rho": (0.1, 0.3, 0.5), "tau": (0.1, 0.3, 0.5),
            "phi": (0.1, 0.3, 0.5)}.get(name, (0.1, 0.3, 0.5))
    return "negligible" if v < cuts[0] else "small" if v < cuts[1] else "medium" if v < cuts[2] else "large"


def _arr(x) -> np.ndarray:
    if isinstance(x, pl.Series):
        x = x.drop_nulls().cast(pl.Float64).to_numpy()
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


def adjust_pvalues(pvalues, method: str = "holm") -> list:
    """Multiple-testing correction: ``holm`` (default, controls family-wise error), ``bh``
    (Benjamini-Hochberg false-discovery rate) or ``bonferroni``."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    out = np.empty(m)
    if method == "bonferroni":
        out = np.minimum(p * m, 1.0)
    elif method == "holm":
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * p[i]))
            out[i] = running
    elif method in ("bh", "fdr"):
        running = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            running = min(running, p[i] * m / (rank + 1))
            out[i] = running
    else:
        raise ValueError("method must be 'holm', 'bh' or 'bonferroni'")
    return out.tolist()


def normality_test(x, alpha: float = 0.05) -> TestResult:
    """Shapiro-Wilk (n <= 5000) or D'Agostino-Pearson (larger samples)."""
    a = _arr(x)
    if len(a) < 3:
        raise ValueError("Need at least 3 values")
    if len(a) <= 5000:
        res, name = st.shapiro(a), "Shapiro-Wilk"
    else:
        res, name = st.normaltest(a), "D'Agostino-Pearson"
    r = TestResult(name, float(res.statistic), float(res.pvalue), n=len(a), alpha=alpha)
    r.details = {"skew": float(st.skew(a)), "kurtosis": float(st.kurtosis(a))}
    r.interpretation = ("consistent with a normal distribution" if not r.significant
                        else "not normally distributed" + (" (with this many rows even tiny departures are significant; "
                                                           "check the skew)" if len(a) > 500 else ""))
    return r


def _normal_enough(a: np.ndarray) -> bool:
    if len(a) >= 50:                     # CLT: means are ~normal; only reject heavy skew
        return abs(st.skew(a)) < 2
    return len(a) >= 3 and st.shapiro(a).pvalue >= 0.05


def ttest(a, b, *, equal_var: bool = False, alpha: float = 0.05, names=("A", "B")) -> TestResult:
    """Two-sample t-test (Welch's by default — safe when variances differ) with Hedges' g and a CI
    for the difference in means."""
    x, y = _arr(a), _arr(b)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Each group needs at least 2 values")
    res = st.ttest_ind(x, y, equal_var=equal_var)
    diff = x.mean() - y.mean()
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    if equal_var:
        df = len(x) + len(y) - 2
        sp = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / df)
        se = sp * math.sqrt(1 / len(x) + 1 / len(y))
    else:
        se = math.sqrt(vx + vy)
        df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if (vx + vy) > 0 else len(x) + len(y) - 2
    t = st.t.ppf(1 - alpha / 2, df)
    pooled = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    d = diff / pooled if pooled > 0 else 0.0
    g = d * (1 - 3 / (4 * (len(x) + len(y)) - 9))
    r = TestResult("Welch's t-test" if not equal_var else "Student's t-test", float(res.statistic), float(res.pvalue),
                   float(g), "hedges_g", (diff - t * se, diff + t * se), len(x) + len(y), alpha)
    r.details = {"mean_" + names[0]: float(x.mean()), "mean_" + names[1]: float(y.mean()), "difference": float(diff)}
    r.interpretation = (f"mean of {names[0]} is {abs(diff):.4g} {'higher' if diff > 0 else 'lower'} than {names[1]} "
                        f"({'statistically significant' if r.significant else 'not statistically significant'}, "
                        f"{effect_label('hedges_g', g)} effect)")
    return r


def mann_whitney(a, b, *, alpha: float = 0.05, names=("A", "B")) -> TestResult:
    """Mann-Whitney U (no normality assumption) with the rank-biserial correlation."""
    x, y = _arr(a), _arr(b)
    res = st.mannwhitneyu(x, y, alternative="two-sided")
    rb = 1 - 2 * res.statistic / (len(x) * len(y))
    rb = -rb                              # positive = first group tends to be larger
    r = TestResult("Mann-Whitney U", float(res.statistic), float(res.pvalue), float(rb), "rank_biserial",
                   n=len(x) + len(y), alpha=alpha)
    r.details = {"median_" + names[0]: float(np.median(x)), "median_" + names[1]: float(np.median(y))}
    r.interpretation = (f"{names[0]} tends to be {'higher' if rb > 0 else 'lower'} than {names[1]} "
                        f"({'significant' if r.significant else 'not significant'}, {effect_label('rank_biserial', rb)} effect)")
    return r


def paired_test(before, after, *, alpha: float = 0.05) -> TestResult:
    """Paired t-test if the differences look normal, otherwise the Wilcoxon signed-rank test."""
    a, b = np.asarray(before, float), np.asarray(after, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = b[ok] - a[ok]
    if _normal_enough(d):
        res = st.ttest_rel(b[ok], a[ok])
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
        se, t = d.std(ddof=1) / math.sqrt(len(d)), st.t.ppf(1 - alpha / 2, len(d) - 1)
        r = TestResult("paired t-test", float(res.statistic), float(res.pvalue), float(dz), "cohen_d",
                       (d.mean() - t * se, d.mean() + t * se), len(d), alpha, {"differences normal": "yes"})
    else:
        res = st.wilcoxon(d)
        r = TestResult("Wilcoxon signed-rank", float(res.statistic), float(res.pvalue), n=len(d), alpha=alpha,
                       assumptions={"differences normal": "no"})
    r.details = {"mean_change": float(d.mean()), "median_change": float(np.median(d))}
    r.interpretation = f"average change {d.mean():+.4g} ({'significant' if r.significant else 'not significant'})"
    return r


def anova(*groups, names=None, alpha: float = 0.05, equal_var: bool | None = None) -> TestResult:
    """One-way ANOVA (equal variances) or Alexander-Govern (unequal variances — the robust choice),
    with eta-squared / omega-squared and Tukey HSD pairwise comparisons."""
    gs = [_arr(g) for g in groups]
    names = list(names or [f"group_{i}" for i in range(len(gs))])
    if equal_var is None:
        equal_var = st.levene(*gs).pvalue >= 0.05
    res = st.f_oneway(*gs) if equal_var else st.alexandergovern(*gs)
    allv = np.concatenate(gs)
    grand = allv.mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in gs)
    ss_t = ((allv - grand) ** 2).sum()
    ss_w = ss_t - ss_b
    k, n = len(gs), len(allv)
    eta = ss_b / ss_t if ss_t > 0 else 0.0
    ms_w = ss_w / (n - k) if n > k else 0.0
    omega = max(0.0, (ss_b - (k - 1) * ms_w) / (ss_t + ms_w)) if (ss_t + ms_w) > 0 else 0.0
    r = TestResult("one-way ANOVA" if equal_var else "Alexander-Govern (unequal-variance ANOVA)", float(res.statistic),
                   float(res.pvalue), float(omega), "omega_squared", n=n, alpha=alpha,
                   assumptions={"equal variances (Levene)": "yes" if equal_var else "no"})
    tukey = st.tukey_hsd(*gs)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            pairs.append({"a": names[i], "b": names[j], "difference": float(gs[i].mean() - gs[j].mean()),
                          "p_value": float(tukey.pvalue[i, j])})
    r.details = {"eta_squared": float(eta), "means": {nm: float(g.mean()) for nm, g in zip(names, gs)},
                 "pairwise_tukey": pairs}
    sig_pairs = [f"{p['a']} vs {p['b']}" for p in pairs if p["p_value"] < alpha]
    r.interpretation = ("group means differ" if r.significant else "no evidence the group means differ") + \
        (f"; significantly different pairs: {', '.join(sig_pairs[:6])}" if r.significant and sig_pairs else "")
    return r


def kruskal(*groups, names=None, alpha: float = 0.05) -> TestResult:
    """Kruskal-Wallis (no normality assumption) with epsilon-squared and Holm-corrected pairwise
    Mann-Whitney tests."""
    gs = [_arr(g) for g in groups]
    names = list(names or [f"group_{i}" for i in range(len(gs))])
    res = st.kruskal(*gs)
    n = sum(len(g) for g in gs)
    eps = float(res.statistic) / ((n ** 2 - 1) / (n + 1)) if n > 1 else 0.0
    pairs, ps = [], []
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            p = float(st.mannwhitneyu(gs[i], gs[j], alternative="two-sided").pvalue)
            pairs.append({"a": names[i], "b": names[j], "median_difference": float(np.median(gs[i]) - np.median(gs[j]))})
            ps.append(p)
    for pr, adj in zip(pairs, adjust_pvalues(ps, "holm")):
        pr["p_value_holm"] = adj
    r = TestResult("Kruskal-Wallis", float(res.statistic), float(res.pvalue), eps, "epsilon_squared", n=n, alpha=alpha)
    r.details = {"medians": {nm: float(np.median(g)) for nm, g in zip(names, gs)}, "pairwise_mann_whitney": pairs}
    r.interpretation = "group distributions differ" if r.significant else "no evidence the groups differ"
    return r


def compare_groups(df: pl.DataFrame, value: str, group: str, *, alpha: float = 0.05, parametric: bool | None = None,
                   min_group: int = 3) -> TestResult:
    """Compare a numeric ``value`` across the levels of ``group``, choosing the test for you:
    normality per group decides parametric vs rank-based, Levene decides equal variances."""
    df = df.select(pl.col(group).cast(pl.String), pl.col(value).cast(pl.Float64)).drop_nulls()
    levels = [lv for lv, n in df[group].value_counts(sort=True).iter_rows() if n >= min_group]
    if len(levels) < 2:
        raise ValueError(f"Need at least 2 groups with {min_group}+ values in '{group}'")
    levels = sorted(levels)
    gs = [df.filter(pl.col(group) == lv)[value].to_numpy() for lv in levels]
    normal = all(_normal_enough(g) for g in gs)
    use_param = normal if parametric is None else parametric
    equal_var = bool(st.levene(*gs).pvalue >= 0.05)
    if len(gs) == 2:
        r = ttest(gs[0], gs[1], alpha=alpha, names=tuple(levels)) if use_param else \
            mann_whitney(gs[0], gs[1], alpha=alpha, names=tuple(levels))
    else:
        r = anova(*gs, names=levels, alpha=alpha) if use_param else kruskal(*gs, names=levels, alpha=alpha)
    r.assumptions = {**r.assumptions, "normal (per group)": "yes" if normal else "no",
                     "equal variances (Levene)": "yes" if equal_var else "no",
                     "chosen": "parametric" if use_param else "rank-based"}
    r.interpretation = f"'{value}' by '{group}': " + r.interpretation
    return r


def chi_square(df: pl.DataFrame, a: str, b: str, *, alpha: float = 0.05) -> TestResult:
    """Independence of two categorical columns; Fisher's exact test for 2x2 tables with small
    expected counts. Effect size: Cramér's V (bias-corrected)."""
    t = df.select(pl.col(a).cast(pl.String), pl.col(b).cast(pl.String)).drop_nulls()
    table = t.group_by(a, b).len().pivot(on=b, index=a, values="len").fill_null(0)
    obs = table.drop(a).to_numpy().astype(float)
    if obs.shape[0] < 2 or obs.shape[1] < 2:
        raise ValueError("Each column needs at least two categories")
    chi2, p, dof, expected = st.chi2_contingency(obs, correction=False)
    n = obs.sum()
    r_, k_ = obs.shape
    phi2 = max(0.0, chi2 / n - (k_ - 1) * (r_ - 1) / (n - 1))
    rc, kc = r_ - (r_ - 1) ** 2 / (n - 1), k_ - (k_ - 1) ** 2 / (n - 1)
    v = math.sqrt(phi2 / max(min(kc - 1, rc - 1), 1e-12))
    small = (expected < 5).mean()
    if obs.shape == (2, 2) and small > 0:
        odds, p = st.fisher_exact(obs)
        res = TestResult("Fisher's exact test", float(odds), float(p), v, "cramers_v", n=int(n), alpha=alpha)
    else:
        res = TestResult("chi-square test of independence", float(chi2), float(p), v, "cramers_v", n=int(n), alpha=alpha)
    res.assumptions = {"cells with expected count < 5": f"{small:.0%}"}
    rates = (obs / obs.sum(axis=1, keepdims=True))
    res.details = {"dof": int(dof), "table": {str(r): dict(zip(table.columns[1:], map(int, row)))
                                              for r, row in zip(table[a].to_list(), obs)},
                   "row_shares": {str(r): dict(zip(table.columns[1:], map(float, row)))
                                  for r, row in zip(table[a].to_list(), rates)}}
    res.interpretation = (f"'{a}' and '{b}' are {'associated' if res.significant else 'not detectably associated'}"
                          f" ({effect_label('cramers_v', v)} association)")
    if small > 0.2 and res.test.startswith("chi"):
        res.interpretation += "; many expected counts are small - merge rare categories"
    return res


def correlation_test(df: pl.DataFrame, x: str, y: str, *, method: str = "pearson", alpha: float = 0.05) -> TestResult:
    """Correlation with a p-value and (Pearson/Spearman) a Fisher-z confidence interval."""
    t = df.select(pl.col(x).cast(pl.Float64), pl.col(y).cast(pl.Float64)).drop_nulls()
    a, b = t[x].to_numpy(), t[y].to_numpy()
    n = len(a)
    if n < 4:
        raise ValueError("Need at least 4 complete pairs")
    fn = {"pearson": st.pearsonr, "spearman": st.spearmanr, "kendall": st.kendalltau}.get(method)
    if fn is None:
        raise ValueError("method must be pearson, spearman or kendall")
    res = fn(a, b)
    rho = float(res.statistic)
    ci = None
    if method != "kendall" and abs(rho) < 1:
        z, se = math.atanh(rho), 1 / math.sqrt(n - 3)
        q = st.norm.ppf(1 - alpha / 2)
        ci = (math.tanh(z - q * se), math.tanh(z + q * se))
    name = {"pearson": "r", "spearman": "rho", "kendall": "tau"}[method]
    r = TestResult(f"{method.title()} correlation", rho, float(res.pvalue), rho, name, ci, n, alpha)
    r.interpretation = (f"{'positive' if rho > 0 else 'negative'} {effect_label(name, rho)} correlation between "
                        f"'{x}' and '{y}' ({'significant' if r.significant else 'not significant'}); correlation is not causation")
    return r


def proportion_test(successes_a: int, total_a: int, successes_b: int, total_b: int, *, alpha: float = 0.05) -> TestResult:
    """Two-proportion z-test with a Newcombe-style CI for the difference."""
    p1, p2 = successes_a / total_a, successes_b / total_b
    pooled = (successes_a + successes_b) / (total_a + total_b)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / total_a + 1 / total_b))
    z = (p1 - p2) / se0 if se0 > 0 else 0.0
    p = 2 * (1 - st.norm.cdf(abs(z)))
    se = math.sqrt(p1 * (1 - p1) / total_a + p2 * (1 - p2) / total_b)
    q = st.norm.ppf(1 - alpha / 2)
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))         # Cohen's h
    r = TestResult("two-proportion z-test", float(z), float(p), float(h), "cohen_h",
                   (p1 - p2 - q * se, p1 - p2 + q * se), total_a + total_b, alpha)
    r.details = {"rate_a": p1, "rate_b": p2, "difference": p1 - p2,
                 "relative_lift": (p1 - p2) / p2 if p2 else None}
    r.interpretation = (f"rate {p1:.2%} vs {p2:.2%} ({p1 - p2:+.2%} points; "
                        f"{'significant' if r.significant else 'not significant'})")
    return r


def ab_test(df: pl.DataFrame, variant: str, metric: str, *, control=None, alpha: float = 0.05) -> TestResult:
    """A/B test of a binary (conversion) or numeric metric between two variants; reports the lift
    of the treatment over ``control`` (default: the first variant alphabetically)."""
    t = df.select(pl.col(variant).cast(pl.String), pl.col(metric)).drop_nulls()
    levels = sorted(t[variant].unique().to_list())
    if len(levels) != 2:
        raise ValueError(f"ab_test needs exactly 2 variants, found {levels}")
    control = str(control) if control is not None else levels[0]
    treat = next(lv for lv in levels if lv != control)
    a, b = t.filter(pl.col(variant) == treat)[metric], t.filter(pl.col(variant) == control)[metric]
    binary = t[metric].n_unique() <= 2 and set(map(float, t[metric].unique().to_list())) <= {0.0, 1.0}
    if binary:
        r = proportion_test(int(a.sum()), a.len(), int(b.sum()), b.len(), alpha=alpha)
    else:
        r = ttest(a, b, alpha=alpha, names=(treat, control))
        ma, mb = float(a.mean() or 0.0), float(b.mean() or 0.0)  # type: ignore[arg-type]
        r.details["relative_lift"] = (ma - mb) / abs(mb) if mb else None
    r.test = f"A/B test ({r.test})"
    lift = r.details.get("relative_lift")
    r.interpretation = (f"'{treat}' vs control '{control}': " + r.interpretation +
                        (f"; relative lift {lift:+.1%}" if lift is not None else ""))
    return r


def required_sample_size(*, baseline_rate: float | None = None, mde: float, std: float | None = None,
                         alpha: float = 0.05, power: float = 0.8) -> int:
    """Rows needed **per variant** to detect an absolute effect ``mde``: on a conversion rate
    (give ``baseline_rate``) or on a mean (give the metric's ``std``)."""
    za, zb = st.norm.ppf(1 - alpha / 2), st.norm.ppf(power)
    if baseline_rate is not None:
        p1, p2 = baseline_rate, baseline_rate + mde
        pbar = (p1 + p2) / 2
        n = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / mde ** 2
    elif std is not None:
        n = 2 * ((za + zb) * std / mde) ** 2
    else:
        raise ValueError("Give baseline_rate (proportions) or std (means)")
    return int(math.ceil(n))


def test_all(df: pl.DataFrame, target: str, *, alpha: float = 0.05, correction: str = "bh", max_categories: int = 30) -> pl.DataFrame:
    """Every column's association with ``target`` using the appropriate test (numeric target:
    correlation / group comparison; categorical target: group comparison / chi-square), with
    p-values corrected for multiple testing. Sorted by effect size."""
    y = df[target]
    y_num = is_num(y.dtype) and y.dtype != pl.Boolean and y.n_unique() > 10
    rows = []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        x_num = is_num(s.dtype) and s.dtype != pl.Boolean and s.n_unique() > 10
        if not x_num and s.drop_nulls().n_unique() > max_categories:
            continue
        try:
            if x_num and y_num:
                r = correlation_test(df, c, target, method="spearman")
            elif x_num and not y_num:
                r = compare_groups(df, c, target, alpha=alpha)
            elif not x_num and y_num:
                r = compare_groups(df, target, c, alpha=alpha)
            else:
                r = chi_square(df, c, target, alpha=alpha)
        except (ValueError, ZeroDivisionError):
            continue
        rows.append({"column": c, "test": r.test, "statistic": r.statistic, "p_value": r.p_value,
                     "effect_size": r.effect_size, "effect_name": r.effect_name,
                     "effect": effect_label(r.effect_name, r.effect_size), "n": r.n})
    schema = {"column": pl.String, "test": pl.String, "statistic": pl.Float64, "p_value": pl.Float64,
              "effect_size": pl.Float64, "effect_name": pl.String, "effect": pl.String, "n": pl.Int64}
    out = pl.DataFrame(rows, schema=schema)
    if out.height:
        out = out.with_columns(pl.Series("p_adjusted", adjust_pvalues(out["p_value"].to_list(), correction)))
        out = out.with_columns((pl.col("p_adjusted") < alpha).alias("significant"))
        out = out.sort(pl.col("effect_size").abs(), descending=True, nulls_last=True)
    return out
