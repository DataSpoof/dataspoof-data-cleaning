"""Automatic feature selection."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from .._detect import is_num, target_kind
from ..core import Step, register_step

METHODS = ("importance", "mutual_info", "variance")


def feature_matrix(df: pl.DataFrame, cols: list) -> np.ndarray:
    """Float matrix of ``cols`` with nulls filled by the column median (0 if all null)."""
    if not cols:
        return np.empty((df.height, 0))
    X = df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
    if not X.size:
        return X
    with warnings.catch_warnings():               # all-null columns: "All-NaN slice"
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    return np.where(np.isnan(X), medians, X)


def target_vector(y: pl.Series):
    """(kind, y as numpy) with labels encoded as integers for classification."""
    kind = target_kind(y)
    if kind == "regression":
        return kind, y.cast(pl.Float64).to_numpy()
    labels = y.cast(pl.String).to_list()
    codes = {v: i for i, v in enumerate(sorted(set(labels)))}
    return kind, np.array([codes[v] for v in labels])


@register_step
class SelectFeatures(Step):
    """Keep the numeric features that matter; drop the rest.

    * ``importance`` (default): random-forest importance; keep features until their cumulative
      importance reaches ``cumulative_importance`` (0.95), capped at ``max_features``.
    * ``mutual_info``: mutual information with the target; keep scores above ``min_score``
      (top ``max_features`` if set).
    * ``variance``: only drop constant columns (no target needed).

    Constant columns are always dropped. Columns in ``keep``, text columns and the target are
    never touched.
    """

    name = "select_features"
    uses_target = True

    def __init__(self, *, target=None, method="importance", max_features=None,
                 cumulative_importance=0.95, min_score=0.0, keep=None, sample=20000, seed=42,
                 fit_only=None):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        super().__init__(fit_only=fit_only, target=target, method=method, max_features=max_features,
                         cumulative_importance=cumulative_importance, min_score=min_score,
                         keep=keep, sample=sample, seed=seed)

    def _fit(self, df):
        p = self.params
        target = self._target(required=p["method"] != "variance")
        keep = set(p["keep"] or [])
        feats = [c for c, t in df.schema.items()
                 if c != target and (is_num(t) or t == pl.Boolean) and c not in keep]
        if target:
            df = df.filter(pl.col(target).is_not_null())
        if df.height > p["sample"]:
            df = df.sample(p["sample"], seed=p["seed"])
        X = feature_matrix(df, feats)
        variances = X.var(axis=0) if X.size else np.array([])
        constant = [c for c, v in zip(feats, variances) if v == 0]
        live = [c for c in feats if c not in constant]
        Xl = X[:, [feats.index(c) for c in live]] if live else X[:, :0]

        if p["method"] == "variance" or not live:
            scores = {c: float(v) for c, v in zip(feats, variances)}
            selected = live
        else:
            kind, y = target_vector(df[target])
            classify = kind != "regression"
            if p["method"] == "importance":
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                model = (RandomForestClassifier if classify else RandomForestRegressor)(
                    n_estimators=150, random_state=p["seed"], n_jobs=-1, min_samples_leaf=2)
                raw = model.fit(Xl, y).feature_importances_
            else:
                from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
                raw = (mutual_info_classif if classify else mutual_info_regression)(
                    Xl, y, random_state=p["seed"])
            total = float(np.sum(raw)) or 1.0
            scores = {c: float(s) / total for c, s in zip(live, raw)}
            ranked = sorted(live, key=lambda c: -scores[c])
            selected = []
            if p["method"] == "importance":
                running = 0.0
                for c in ranked:
                    if running >= p["cumulative_importance"]:
                        break
                    selected.append(c)
                    running += scores[c]
            else:
                selected = [c for c in ranked if scores[c] > p["min_score"]]
            if p["max_features"]:
                selected = selected[: p["max_features"]]
            selected = selected or ranked[:1]
        dropped = [c for c in feats if c not in selected]
        return {"target": target, "selected": selected, "dropped": dropped,
                "constant": constant, "scores": scores}

    def _transform(self, df):
        drop = [c for c in self.state_["dropped"] if c in df.columns]
        return df.drop(drop) if drop else df
