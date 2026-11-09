"""Model security, part 2: is the training data poisoned or mislabelled?

>>> rep = poisoning_report(train_df, target="fraud")
>>> print(rep.summary())
>>> rep.suspicious                    # row index, score, reasons - review these first
>>> clean = rep.clean(train_df)       # drop the flagged rows (or rep.clean(train_df, min_score=...))

Checks (no model needed; works for classification and regression targets):

* **Label issues** (classification, *confident learning*): out-of-fold probabilities from a
  quick model; a row is suspect when the model is confidently sure it belongs to another class
  (P(given label) below that class's average self-confidence and another class above its own).
  Catches label-flipping attacks and ordinary annotation mistakes.
* **Conflicting duplicates**: identical feature values with different labels.
* **Backdoor triggers**: a rare value (or a numeric value repeated far more than its
  neighbours) whose rows almost all carry one label - the signature of a planted trigger.
  Tested with a one-sided binomial test, corrected for the number of candidates (Holm).
* **Outliers**: Isolation Forest on numeric features - injected points often sit far from the
  real data.
* **Regression**: rows whose target is extreme given out-of-fold predictions (robust z > 4).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import polars as pl


@dataclasses.dataclass
class PoisoningReport:
    rows: int
    task: str
    suspicious: pl.DataFrame             # row, score, reasons
    label_issues: int
    conflicting_duplicates: int
    triggers: pl.DataFrame               # feature, value, rows, label, purity, base_rate, p_value
    outliers: int

    def clean(self, df, *, min_score: float = 0.0) -> pl.DataFrame:
        """``df`` without the flagged rows (those scoring above ``min_score``)."""
        df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        bad = self.suspicious.filter(pl.col("score") > min_score)["row"].to_list()
        return df.with_row_index("_r").filter(~pl.col("_r").is_in(bad)).drop("_r")

    def summary(self) -> str:
        lines = [f"Poisoning / label-quality check ({self.task}, {self.rows:,} rows): "
                 f"{self.suspicious.height:,} suspicious rows ({self.suspicious.height / max(self.rows, 1):.1%})",
                 f"  likely label errors: {self.label_issues:,}",
                 f"  identical rows with different labels: {self.conflicting_duplicates:,}",
                 f"  outliers: {self.outliers:,}"]
        for r in self.triggers.head(5).iter_rows(named=True):
            lines.append(f"  possible trigger: {r['feature']} = {r['value']!r} -> label {r['label']!r} in "
                         f"{r['purity']:.0%} of {r['rows']} rows (base rate {r['base_rate']:.0%}, p={r['p_value']:.2g})")
        if not self.triggers.height:
            lines.append("  no backdoor-trigger patterns found")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _features(df: pl.DataFrame, target: str, *, with_names: bool = False):
    """A numeric matrix for a quick model: numerics as-is (median-filled), low-cardinality
    categoricals one-hot, high-cardinality ones frequency-encoded. ``with_names`` also returns
    the source column of every matrix column."""
    X = df.drop(target)
    cols, names = [], []
    for c, t in X.schema.items():
        s = X[c]
        if t.is_numeric() or t == pl.Boolean:
            v = s.cast(pl.Float64)
            cols.append(v.fill_null(v.median() if v.drop_nulls().len() else 0.0).to_numpy())
            names.append(c)
        elif t in (pl.String, pl.Categorical) or isinstance(t, pl.Enum):
            s = s.cast(pl.String).fill_null("<null>")
            if s.n_unique() <= 30:
                for v in s.unique().to_list():
                    cols.append((s == v).cast(pl.Float64).to_numpy())
                    names.append(c)
            else:
                freq = s.value_counts().rename({"count": "_f"})
                cols.append(pl.DataFrame({c: s}).join(freq, on=c, how="left")["_f"].cast(pl.Float64).to_numpy())
                names.append(c)
    M = np.column_stack(cols) if cols else np.zeros((df.height, 1))
    return (M, names or ["_none"]) if with_names else M


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def poisoning_report(df, target: str, *, cv: int = 5, seed: int = 0, contamination: float = 0.01,
                     trigger_max_support: float = 0.05, trigger_min_rows: int = 8, alpha: float = 0.01,
                     classification: bool | None = None) -> PoisoningReport:
    """Run all checks; see the module docstring. ``trigger_max_support`` is the largest share of
    rows a trigger value may cover; ``contamination`` the expected share of outliers."""
    from scipy import stats as st
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

    from .splitting import _is_classification
    df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    original_rows = np.flatnonzero(df[target].is_not_null().to_numpy())   # report rows of the caller's frame
    df = df.filter(pl.col(target).is_not_null())
    n = df.height
    y_raw = df[target]
    is_clf = _is_classification(y_raw) if classification is None else classification
    X = _features(df, target)
    reasons: dict = {}
    scores = np.zeros(n)

    def flag(rows, reason, weight):
        for r in np.asarray(rows).tolist():
            reasons.setdefault(r, []).append(reason)
            scores[r] += weight

    # 1. label issues
    label_issues = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_clf:
            labels = y_raw.cast(pl.String).to_numpy()
            classes, y = np.unique(labels, return_inverse=True)
            if len(classes) >= 2 and np.min(np.bincount(y)) >= 2:
                folds = min(cv, int(np.min(np.bincount(y))))
                proba = cross_val_predict(HistGradientBoostingClassifier(max_iter=150, random_state=seed), X, y,
                                          cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                                          method="predict_proba")
                thresholds = np.array([proba[y == k, k].mean() for k in range(len(classes))])
                given = proba[np.arange(n), y]
                above = proba >= thresholds[None, :]
                above[np.arange(n), y] = False
                suspect = (given < thresholds[y]) & above.any(axis=1)
                margin = proba.max(axis=1) - given
                idx = np.flatnonzero(suspect & (margin > 0.3))
                label_issues = len(idx)
                flag(idx, "likely mislabelled", 1.0)
                for i in idx:
                    scores[i] += float(margin[i])
        else:
            yv = y_raw.cast(pl.Float64).to_numpy()
            pred = cross_val_predict(HistGradientBoostingRegressor(max_iter=150, random_state=seed), X, yv,
                                     cv=KFold(cv, shuffle=True, random_state=seed))
            resid = yv - pred
            mad = np.median(np.abs(resid - np.median(resid))) * 1.4826 or (np.std(resid) or 1.0)
            z = np.abs(resid - np.median(resid)) / mad
            idx = np.flatnonzero(z > 4)
            label_issues = len(idx)
            flag(idx, "target far from what similar rows have", 1.0)

    # 2. conflicting duplicates
    feat_cols = [c for c in df.columns if c != target]
    conflicts = 0
    if feat_cols:
        g = (df.with_row_index("_r").group_by(feat_cols)
             .agg(pl.col(target).n_unique().alias("_labels"), pl.col("_r")).filter(pl.col("_labels") > 1))
        rows = [r for lst in g["_r"].to_list() for r in lst]
        conflicts = len(rows)
        flag(rows, "identical features, different label", 1.0)

    # 3. backdoor triggers
    trig_rows = []
    if is_clf:
        labels = y_raw.cast(pl.String)
        base = labels.value_counts(normalize=True)
        base_rate = dict(zip(base[target].to_list(), base["proportion"].to_list()))
        for c in feat_cols:
            s = df[c]
            vc = pl.DataFrame({"v": s, "y": labels}).drop_nulls("v")
            counts = vc.group_by("v").len()
            if s.dtype.is_numeric():
                # only values repeated unusually often (a planted constant), not every distinct number
                if s.n_unique() < 20:
                    cand = counts
                else:
                    cand = counts.filter(pl.col("len") >= max(trigger_min_rows, 20 * counts["len"].median()))
            else:
                cand = counts
            cand = cand.filter((pl.col("len") >= trigger_min_rows) & (pl.col("len") <= trigger_max_support * n))
            for v, cnt in cand.iter_rows():
                sub = vc.filter(pl.col("v") == v)["y"].value_counts(sort=True)
                top_label, top_n = sub.row(0)
                p0 = base_rate.get(top_label, 0.0)
                if top_n / cnt < 0.9 or p0 >= 0.9:
                    continue
                p = float(st.binomtest(int(top_n), int(cnt), p0, alternative="greater").pvalue)
                trig_rows.append({"feature": c, "value": str(v), "rows": int(cnt), "label": str(top_label),
                                  "purity": top_n / cnt, "base_rate": p0, "p_value": p})
    trig = pl.DataFrame(trig_rows, schema={"feature": pl.String, "value": pl.String, "rows": pl.Int64,
                                           "label": pl.String, "purity": pl.Float64, "base_rate": pl.Float64,
                                           "p_value": pl.Float64})
    if trig.height:
        trig = trig.with_columns(pl.Series("p_value", _holm(trig["p_value"].to_numpy()))).filter(
            pl.col("p_value") < alpha).sort("p_value")
        for r in trig.iter_rows(named=True):
            col = df[r["feature"]].cast(pl.String)
            hit = np.flatnonzero(((col == r["value"]) & (y_raw.cast(pl.String) == r["label"])).fill_null(False).to_numpy())
            flag(hit, f"carries possible trigger {r['feature']}={r['value']}", 1.5)

    # 4. outliers
    outliers = 0
    num = [c for c, t in df.drop(target).schema.items() if t.is_numeric()]
    if num and n >= 50:
        Z = df.select([pl.col(c).cast(pl.Float64).fill_null(pl.col(c).median()) for c in num]).to_numpy()
        iso = IsolationForest(contamination=contamination, random_state=seed).fit(Z)
        idx = np.flatnonzero(iso.predict(Z) == -1)
        outliers = len(idx)
        flag(idx, "outlier", 0.5)

    sus = pl.DataFrame({"row": [int(original_rows[r]) for r in reasons], "score": [float(scores[r]) for r in reasons],
                        "reasons": ["; ".join(v) for v in reasons.values()]},
                       schema={"row": pl.Int64, "score": pl.Float64, "reasons": pl.String}).sort("score", descending=True)
    return PoisoningReport(rows=n, task="classification" if is_clf else "regression", suspicious=sus,
                           label_issues=label_issues, conflicting_duplicates=conflicts, triggers=trig,
                           outliers=outliers)
