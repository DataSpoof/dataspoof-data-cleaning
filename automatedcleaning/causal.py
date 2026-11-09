"""Causal effects and uplift: what a treatment (discount, campaign, feature) actually *causes*.

``estimate_effect(df, treatment="got_discount", outcome="revenue")`` estimates the average
treatment effect with **AIPW** (augmented inverse-propensity weighting, "doubly robust"): models
of the outcome and of who got treated are fitted with cross-fitting, so the estimate stays
consistent if either model is right. It reports a 95% confidence interval, the naive
(confounded) difference for comparison, and diagnostics that tell you when not to trust it:
**overlap** (treated and untreated must be comparable) and **covariate balance** after weighting.

``UpliftModel`` estimates the effect *per customer* (T-learner) so you can target those who
respond; ``qini`` scores how well a ranking finds them.

Causal estimates rest on an assumption no data can prove: every common cause of treatment and
outcome is among the ``confounders``. The report says so.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import polars as pl

from .automl.frame import infer_kinds, make_preprocessor, to_model_frame


def _binary(series: pl.Series, name: str) -> np.ndarray:
    vals = series.drop_nulls().unique().to_list()
    if len(vals) != 2 and not set(vals) <= {0, 1, True, False}:
        raise ValueError(f"'{name}' must be binary (two values); got {vals[:5]}")
    if series.dtype == pl.Boolean or set(vals) <= {0, 1}:
        return series.cast(pl.Int8).to_numpy().astype(int)
    hi = sorted(vals, key=str)[1]
    return (series == hi).cast(pl.Int8).to_numpy().astype(int)


def _models(binary_outcome: bool, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    prop = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
    out = (HistGradientBoostingClassifier if binary_outcome else HistGradientBoostingRegressor)(
        max_iter=200, learning_rate=0.05, random_state=seed)
    return prop, out


@dataclasses.dataclass
class EffectEstimate:
    treatment: str
    outcome: str
    ate: float
    ci: tuple
    p_value: float
    naive_difference: float
    n: int
    n_treated: int
    overlap: dict
    balance: pl.DataFrame
    confounders: list
    method: str = "AIPW (doubly robust, cross-fitted)"

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        sign = "increases" if self.ate > 0 else "decreases"
        lines = [f"Effect of '{self.treatment}' on '{self.outcome}' ({self.method}, n={self.n:,}, "
                 f"{self.n_treated:,} treated)",
                 f"  Estimated average effect: {self.ate:+.4g} (95% CI {self.ci[0]:+.4g} to {self.ci[1]:+.4g}, "
                 f"p={self.p_value:.3g})",
                 f"  Naive difference (treated - untreated): {self.naive_difference:+.4g}"
                 + (" - confounding changes the answer" if abs(self.naive_difference - self.ate) > abs(self.ci[1] - self.ci[0]) / 2
                    else ""),
                 f"  Plain English: '{self.treatment}' {sign} '{self.outcome}' by about {abs(self.ate):.4g} on average"
                 + ("" if self.significant else " - but this is not distinguishable from zero")]
        o = self.overlap
        lines.append(f"  Overlap: propensity {o['min']:.3f}-{o['max']:.3f}; {o['extreme_share']:.1%} of rows have "
                     f"propensity < 0.05 or > 0.95" + (" - WEAK OVERLAP: estimate relies on extrapolation"
                                                     if o["extreme_share"] > 0.1 else ""))
        worst = self.balance.sort("smd_weighted", descending=True, nulls_last=True).head(1)
        if worst.height:
            r = worst.row(0, named=True)
            ok = r["smd_weighted"] is None or r["smd_weighted"] < 0.1
            lines.append(f"  Balance after weighting: largest standardised difference {r['smd_weighted']:.3f} "
                         f"({r['feature']}) - {'good (< 0.1)' if ok else 'IMBALANCED: add confounders or check overlap'}")
        lines.append("  Assumes every common cause of treatment and outcome is among the confounders "
                     f"({len(self.confounders)} used).")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _smd(x: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> float | None:
    w = np.ones_like(x, dtype=float) if w is None else w
    a, b = t == 1, t == 0
    if a.sum() < 2 or b.sum() < 2:
        return None
    m1, m0 = np.average(x[a], weights=w[a]), np.average(x[b], weights=w[b])
    v1, v0 = np.average((x[a] - m1) ** 2, weights=w[a]), np.average((x[b] - m0) ** 2, weights=w[b])
    s = math.sqrt((v1 + v0) / 2)
    return abs(m1 - m0) / s if s > 0 else 0.0


def estimate_effect(df, treatment: str, outcome: str, *, confounders=None, folds: int = 5, clip: float = 0.01,
                    seed: int = 42) -> EffectEstimate:
    """Average treatment effect of a binary ``treatment`` on ``outcome`` (numeric or binary).

    ``confounders`` default to every other column. Propensities are clipped to
    [``clip``, 1 - ``clip``] to keep weights finite."""
    from scipy.stats import norm
    from sklearn.model_selection import KFold
    df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    df = df.filter(pl.col(treatment).is_not_null() & pl.col(outcome).is_not_null())
    confounders = list(confounders) if confounders is not None else [c for c in df.columns if c not in (treatment, outcome)]
    if not confounders:
        raise ValueError("Need at least one confounder column")
    t = _binary(df[treatment], treatment)
    out = df[outcome]
    binary_outcome = out.n_unique() == 2
    y = _binary(out, outcome).astype(float) if binary_outcome else out.cast(pl.Float64).to_numpy()
    kinds = infer_kinds(df.select(confounders))
    X = to_model_frame(df.select(confounders), kinds)
    n = len(y)
    if t.sum() < 10 or (1 - t).sum() < 10:
        raise ValueError("Need at least 10 treated and 10 untreated rows")
    e, mu1, mu0 = np.zeros(n), np.zeros(n), np.zeros(n)
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline as SkPipeline
    prop_base, out_base = _models(binary_outcome, seed)
    kf = KFold(folds, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in kf.split(X):
            prop = SkPipeline([("prep", make_preprocessor(kinds, False)), ("m", clone(prop_base))]).fit(X.iloc[tr], t[tr])
            e[te] = prop.predict_proba(X.iloc[te])[:, 1]
            for arm, target in ((1, mu1), (0, mu0)):
                idx = tr[t[tr] == arm]
                model = SkPipeline([("prep", make_preprocessor(kinds, False)), ("m", clone(out_base))]).fit(
                    X.iloc[idx], y[idx])
                target[te] = model.predict_proba(X.iloc[te])[:, 1] if binary_outcome else model.predict(X.iloc[te])
    e_raw = e.copy()
    e = np.clip(e, clip, 1 - clip)
    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / math.sqrt(n))
    z = ate / se if se > 0 else 0.0
    p = float(2 * (1 - norm.cdf(abs(z))))
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    num = [c for c, k in kinds.items() if k == "num"]
    rows = []
    for c in num:
        x = X[c].to_numpy(dtype=float)
        x = np.where(np.isnan(x), np.nanmedian(x), x)
        rows.append({"feature": c, "smd_raw": _smd(x, t), "smd_weighted": _smd(x, t, w)})
    for c in [c for c, k in kinds.items() if k == "cat"]:
        for level in X[c].value_counts().index[:5]:
            x = (X[c] == level).to_numpy(dtype=float)
            rows.append({"feature": f"{c}={level}", "smd_raw": _smd(x, t), "smd_weighted": _smd(x, t, w)})
    balance = pl.DataFrame(rows, schema={"feature": pl.String, "smd_raw": pl.Float64, "smd_weighted": pl.Float64})
    overlap = {"min": float(e_raw.min()), "max": float(e_raw.max()),
               "extreme_share": float(np.mean((e_raw < 0.05) | (e_raw > 0.95)))}
    return EffectEstimate(treatment, outcome, ate, (ate - 1.96 * se, ate + 1.96 * se), p,
                          float(y[t == 1].mean() - y[t == 0].mean()), n, int(t.sum()), overlap, balance, confounders)


class UpliftModel:
    """Per-row treatment effect (T-learner: separate outcome models for treated and untreated).

    >>> up = UpliftModel("got_offer", "converted").fit(campaign)
    >>> up.predict_uplift(customers)      # who gains most from the offer
    """

    def __init__(self, treatment: str, outcome: str, *, features=None, seed: int = 42):
        self.treatment, self.outcome, self.features, self.seed = treatment, outcome, features, seed

    def fit(self, df) -> "UpliftModel":
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline as SkPipeline
        df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        df = df.filter(pl.col(self.treatment).is_not_null() & pl.col(self.outcome).is_not_null())
        self.features_ = list(self.features or [c for c in df.columns if c not in (self.treatment, self.outcome)])
        self.kinds_ = infer_kinds(df.select(self.features_))
        X = to_model_frame(df.select(self.features_), self.kinds_)
        t = _binary(df[self.treatment], self.treatment)
        self.binary_ = df[self.outcome].n_unique() == 2
        y = _binary(df[self.outcome], self.outcome).astype(float) if self.binary_ else df[self.outcome].cast(pl.Float64).to_numpy()
        _, base = _models(self.binary_, self.seed)
        self.models_ = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for arm in (0, 1):
                self.models_[arm] = SkPipeline([("prep", make_preprocessor(self.kinds_, False)),
                                                ("m", clone(base))]).fit(X[t == arm], y[t == arm])
        return self

    def predict_uplift(self, df) -> pl.Series:
        df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        X = to_model_frame(df.select([c for c in self.features_ if c in df.columns]), self.kinds_)
        pred = {arm: (m.predict_proba(X)[:, 1] if self.binary_ else m.predict(X)) for arm, m in self.models_.items()}
        return pl.Series("uplift", pred[1] - pred[0])


def qini(uplift, treatment, outcome, bins: int = 10) -> dict:
    """Qini curve and coefficient: how much more of the treatment effect is captured by targeting
    in ``uplift`` order than at random (0 = no better than random; higher is better)."""
    u = np.asarray(uplift, float)
    t = np.asarray(treatment).astype(int)
    y = np.asarray(outcome, float)
    order = np.argsort(-u)
    t, y = t[order], y[order]
    n = len(u)
    points = [0.0]
    fractions = [0.0]
    for k in range(1, bins + 1):
        m = int(round(n * k / bins))
        tt, yy = t[:m], y[:m]
        nt, nc = tt.sum(), (1 - tt).sum()
        gain = (yy[tt == 1].sum() - (yy[tt == 0].sum() * nt / nc if nc else 0.0))
        points.append(float(gain))
        fractions.append(k / bins)
    total = points[-1]
    random_curve = [total * f for f in fractions]
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")       # numpy 2 renamed trapz
    area = float(integrate(points, fractions) - integrate(random_curve, fractions))
    return {"fractions": fractions, "qini": points, "random": random_curve, "coefficient": area}
