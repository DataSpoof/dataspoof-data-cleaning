"""Leakage-safe target encoding."""

from __future__ import annotations

import numpy as np
import polars as pl

from .._detect import is_text, target_kind
from ..core import Step, register_step

_KEY, _FOLD, _ROW = "__te_key__", "__te_fold__", "__te_row__"


def _plain(v):
    if v is None:
        return None
    v = float(v)
    return None if v != v else v


@register_step
class TargetEncode(Step):
    """Replace categories with the (smoothed) average target value of that category.

    Leakage-safe: while fitting, each training row is encoded with statistics from the
    *other* folds (out-of-fold), so a row never sees its own label. New data uses statistics
    from all training rows. Unseen or missing categories get the overall average.

    * numeric / binary targets: one column with the mean (binary: share of the positive class,
      the later of the two labels in sorted order);
    * multiclass targets: one ``<col>_te_<class>`` column per class (up to ``max_classes``).

    ``smoothing`` blends rare categories towards the overall average:
    ``(n·mean + smoothing·prior) / (n + smoothing)``.
    """

    name = "target_encode"
    uses_target = True

    def __init__(self, *, target=None, columns=None, min_unique=0, max_unique_ratio=0.9,
                 smoothing=10.0, n_folds=5, max_classes=10, seed=42, fit_only=None):
        super().__init__(fit_only=fit_only, target=target, columns=columns, min_unique=min_unique,
                         max_unique_ratio=max_unique_ratio, smoothing=smoothing, n_folds=n_folds,
                         max_classes=max_classes, seed=seed)

    # -- learning ------------------------------------------------------------
    def _targets(self, y: pl.Series):
        """(kind, [(suffix, numeric target series)], classes or positive label)."""
        kind = target_kind(y)
        if y.dtype == pl.Boolean:
            y = y.cast(pl.Int8)
        if kind == "regression" or (kind == "binary" and y.dtype.is_numeric()):
            return "mean", [("", y.cast(pl.Float64))], None
        labels = y.drop_nulls().cast(pl.String)
        if kind == "binary":
            positive = sorted(labels.unique().to_list())[-1]
            return "mean", [("", (y.cast(pl.String) == positive).cast(pl.Float64))], positive
        counts = labels.value_counts().sort(["count", labels.name], descending=[True, False])
        classes = counts[labels.name].head(self.params["max_classes"]).to_list()
        return "classes", [(f"_te_{k}", (y.cast(pl.String) == k).cast(pl.Float64)) for k in classes], classes

    def _columns(self, df, target):
        if self.params["columns"] is not None:
            return [c for c in self.params["columns"] if c in df.columns and c != target]
        limit = self.params["max_unique_ratio"] * max(df.height, 1)   # skip identifiers
        return [c for c, t in df.schema.items()
                if c != target and is_text(t) and self.params["min_unique"] <= df[c].n_unique() <= limit]

    def _stats(self, keys: pl.Series, ys: list, smoothing: float, priors: list) -> list:
        """Per-target ``{category: smoothed mean}`` maps learned from ``keys``/``ys``."""
        frame = pl.DataFrame({_KEY: keys, **{f"y{j}": y for j, (_, y) in enumerate(ys)}})
        maps = []
        for j in range(len(ys)):
            stats = (frame.filter(pl.col(f"y{j}").is_not_null() & pl.col(_KEY).is_not_null())
                     .group_by(_KEY).agg(pl.col(f"y{j}").mean().alias("m"), pl.len().alias("n")))
            prior = priors[j]
            maps.append({k: (n * m + smoothing * prior) / (n + smoothing)
                         for k, m, n in zip(stats[_KEY].to_list(), stats["m"].to_list(),
                                            stats["n"].to_list())})
        return maps

    def _fit(self, df):
        target = self._target()
        kind, ys, labels = self._targets(df[target])
        priors = [_plain(y.mean()) or 0.0 for _, y in ys]
        cols = self._columns(df, target)
        maps = {c: self._stats(df[c].cast(pl.String), ys, self.params["smoothing"], priors) for c in cols}
        return {"target": target, "kind": kind, "labels": labels, "suffixes": [s for s, _ in ys],
                "priors": priors, "maps": maps}

    # -- applying ------------------------------------------------------------
    def _encode(self, df, maps_by_col, priors):
        st = self.state_
        exprs, replaced = [], []
        for c, maps in maps_by_col.items():
            if c not in df.columns:
                continue
            key = pl.col(c).cast(pl.String)
            for suffix, mapping, prior in zip(st["suffixes"], maps, priors):
                value = (key.replace_strict(mapping, default=prior, return_dtype=pl.Float64)
                         if mapping else pl.lit(prior, dtype=pl.Float64))
                exprs.append(value.fill_null(prior).alias(f"{c}{suffix}"))
            if st["kind"] == "classes":
                replaced.append(c)
        out = df.with_columns(exprs) if exprs else df
        return out.drop(replaced) if replaced else out

    def _transform(self, df):
        return self._encode(df, self.state_["maps"], self.state_["priors"])

    def fit_transform(self, df):
        """Fit, then encode the training rows out-of-fold so no row sees its own label."""
        self.fit(df)
        st = self.state_
        if not st["maps"]:
            return df
        n_folds = max(2, min(int(self.params["n_folds"]), df.height))
        rng = np.random.default_rng(self.params["seed"])
        folds = pl.Series(_FOLD, rng.permutation(df.height) % n_folds)
        kind, ys, _ = self._targets(df[st["target"]])
        pieces = []
        work = df.with_row_index(_ROW).with_columns(folds)
        for k in range(n_folds):
            train, held = work.filter(pl.col(_FOLD) != k), work.filter(pl.col(_FOLD) == k)
            ys_train = [(s, y.filter(work[_FOLD] != k)) for s, y in ys]
            priors = [_plain(y.mean()) or 0.0 for _, y in ys_train]
            maps = {c: self._stats(train[c].cast(pl.String), ys_train, self.params["smoothing"], priors)
                    for c in st["maps"]}
            pieces.append(self._encode(held, maps, priors))
        return pl.concat(pieces, how="vertical_relaxed").sort(_ROW).drop([_ROW, _FOLD])
