"""Concept drift *before* the outcomes arrive: estimate live performance without labels.

Labels often arrive weeks after a prediction (did the customer churn? was the loan repaid?).
These estimators warn about a model getting worse in the meantime:

>>> est = PerformanceEstimator(model).fit(reference_with_labels)     # held-out data with labels
>>> est.estimate(todays_batch)                                       # no labels needed
{'metric': 'accuracy', 'estimated': 0.81, 'reference': 0.90, 'low': 0.79, 'high': 0.83, ...}
>>> domain_classifier_drift(reference, todays_batch)                 # multivariate drift

* **Classification - confidence-based performance estimation (CBPE):** the model's confidence is
  calibrated on the reference data (isotonic regression of "was it right?" on confidence, and of
  the true label on P(positive) for binary models). On new data, the expected accuracy is the
  mean calibrated confidence; for binary models the expected ROC AUC is computed by simulating
  labels from the calibrated probabilities. This tracks performance under *covariate shift*
  (inputs moving into regions where the model is less sure). It cannot see a change in P(y|x)
  that leaves the model confident - only real labels can (``Monitor.check_labels``).
* **Regression - direct loss estimation:** expected absolute error per prediction-range bin,
  learnt on the reference data, re-weighted by today's predictions.
* **Multivariate drift - domain classifier:** a model trained to tell reference rows from current
  rows. ROC AUC near 0.5 means no detectable change; the features it relies on are the drift.

Everything is stored as plain numbers (JSON-safe), never raw data.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl


def _iso_fit(x: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)
    return {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}


def _iso_apply(curve: dict, x: np.ndarray) -> np.ndarray:
    return np.interp(x, curve["x"], curve["y"])


class PerformanceEstimator:
    """Estimates a model's accuracy (and ROC AUC for binary models) or MAE without labels."""

    def __init__(self, model, *, bins: int = 10, simulations: int = 200, seed: int = 0):
        from ..deploy.scoring import resolve_model
        self.model = resolve_model(model)
        self.bins, self.simulations, self.seed = bins, simulations, seed
        self.state_: dict | None = None

    @property
    def _classes(self):
        return getattr(self.model, "classes_", None)

    def fit(self, reference) -> "PerformanceEstimator":
        df = reference if isinstance(reference, pl.DataFrame) else pl.from_pandas(reference)
        target = self.model.target
        df = df.filter(pl.col(target).is_not_null())
        if df.height < 50:
            raise ValueError("Need at least 50 labelled reference rows to calibrate the estimator")
        y = df[target]
        if self._classes is not None:
            labels, proba = self.model.predict_with_proba(df)
            P = proba.to_numpy()
            conf = P.max(axis=1)
            correct = (labels.cast(pl.String) == y.cast(pl.String)).cast(pl.Float64).to_numpy()
            st: dict = {"kind": "classification", "confidence": _iso_fit(conf, correct),
                        "reference": {"accuracy": float(correct.mean())}, "rows": df.height}
            if P.shape[1] == 2:
                pos = str(self._classes[1])
                yb = (y.cast(pl.String) == pos).cast(pl.Float64).to_numpy()
                st["positive"] = _iso_fit(P[:, 1], yb)
                from sklearn.metrics import roc_auc_score
                st["reference"]["roc_auc"] = float(roc_auc_score(yb, P[:, 1])) if 0 < yb.mean() < 1 else None
        else:
            pred = self.model.predict(df).to_numpy()
            err = np.abs(y.cast(pl.Float64).to_numpy() - pred)
            edges = np.unique(np.quantile(pred, np.linspace(0, 1, self.bins + 1)))
            which = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(edges) - 2)
            mae = [float(err[which == b].mean()) if np.any(which == b) else float(err.mean())
                   for b in range(len(edges) - 1)]
            st = {"kind": "regression", "edges": [float(e) for e in edges], "bin_mae": mae,
                  "reference": {"mae": float(err.mean())}, "rows": df.height}
        self.state_ = st
        return self

    def estimate(self, batch) -> dict:
        """Estimated metric for an unlabelled batch, with a 90% interval and the reference value."""
        if self.state_ is None:
            raise RuntimeError("Call fit(reference) first")
        df = batch if isinstance(batch, pl.DataFrame) else pl.from_pandas(batch)
        st = self.state_
        rng = np.random.default_rng(self.seed)
        if st["kind"] == "classification":
            _, proba = self.model.predict_with_proba(df)
            P = proba.to_numpy()
            exp_correct = _iso_apply(st["confidence"], P.max(axis=1))
            acc = float(exp_correct.mean())
            sims = rng.binomial(1, exp_correct[None, :].repeat(self.simulations, 0)).mean(axis=1)
            out = {"metric": "accuracy", "estimated": acc, "reference": st["reference"]["accuracy"],
                   "low": float(np.quantile(sims, 0.05)), "high": float(np.quantile(sims, 0.95)), "rows": df.height}
            if "positive" in st and st["reference"].get("roc_auc") is not None and df.height >= 20:
                from sklearn.metrics import roc_auc_score
                p_true = _iso_apply(st["positive"], P[:, 1])
                aucs = []
                for _ in range(self.simulations):
                    yy = rng.binomial(1, p_true)
                    if 0 < yy.sum() < len(yy):
                        aucs.append(roc_auc_score(yy, P[:, 1]))
                if aucs:
                    out["roc_auc"] = {"estimated": float(np.mean(aucs)), "reference": st["reference"]["roc_auc"],
                                      "low": float(np.quantile(aucs, 0.05)), "high": float(np.quantile(aucs, 0.95))}
        else:
            pred = self.model.predict(df).to_numpy()
            edges = np.asarray(st["edges"])
            which = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(edges) - 2)
            outside = (pred < edges[0]) | (pred > edges[-1])
            per = np.asarray(st["bin_mae"])[which]
            per = np.where(outside, per * 2.0, per)          # extrapolating: assume errors at least double
            boot = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(self.simulations)]
            out = {"metric": "mae", "estimated": float(per.mean()), "reference": st["reference"]["mae"],
                   "low": float(np.quantile(boot, 0.05)), "high": float(np.quantile(boot, 0.95)), "rows": df.height,
                   "outside_reference_range": float(outside.mean())}
        out["change"] = out["estimated"] - out["reference"]
        return out

    def to_dict(self) -> dict:
        return {"bins": self.bins, "simulations": self.simulations, "seed": self.seed, "state": self.state_}

    @classmethod
    def from_dict(cls, model, d: dict) -> "PerformanceEstimator":
        est = cls(model, bins=d["bins"], simulations=d["simulations"], seed=d["seed"])
        est.state_ = d["state"]
        return est


def domain_classifier_drift(reference, current, *, columns: list | None = None, cv: int = 3, seed: int = 0,
                            max_rows: int = 20_000) -> dict:
    """Train a classifier to separate reference rows from current rows (cross-validated).
    Returns ``{"auc", "drift": bool, "top_features": [(column, importance)...]}``; AUC above
    ~0.55-0.6 means the two samples are distinguishable, i.e. the joint distribution moved."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    from ..poisoning import _features
    ref = reference if isinstance(reference, pl.DataFrame) else pl.from_pandas(reference)
    cur = current if isinstance(current, pl.DataFrame) else pl.from_pandas(current)
    cols = columns or [c for c in ref.columns if c in cur.columns]
    ref, cur = ref.select(cols), cur.select(cols)
    if ref.height > max_rows:
        ref = ref.sample(max_rows, seed=seed)
    if cur.height > max_rows:
        cur = cur.sample(max_rows, seed=seed)
    both = pl.concat([ref, cur], how="vertical_relaxed").with_columns(
        pl.Series("_is_current", [0] * ref.height + [1] * cur.height))
    X, col_of = _features(both, "_is_current", with_names=True)
    y = both["_is_current"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
        p = cross_val_predict(clf, X, y, cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, p))
        top = []
        if auc > 0.55:
            # importance per original column: permute the column, measure the AUC drop
            fitted = clf.fit(X, y)
            imp = permutation_importance(fitted, X, y, scoring="roc_auc", n_repeats=3, random_state=seed)
            agg: dict = {}
            for c, v in zip(col_of, imp.importances_mean):
                agg[c] = agg.get(c, 0.0) + float(v)
            top = sorted(agg.items(), key=lambda kv: -kv[1])[:10]
    return {"auc": auc, "drift": auc > 0.6, "top_features": top, "reference_rows": ref.height,
            "current_rows": cur.height}
