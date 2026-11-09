"""SHAP explanations: per-row and global feature contributions, plus partial dependence.

Values are interventional Shapley values against a background sample: for every explained
row, ``base_value + sum(values) == model output`` exactly. With up to ``exact_max_features`` (8)
features every coalition is evaluated (exact Shapley); otherwise antithetic permutation
sampling is used, which is unbiased and still exactly additive. Works with any model,
including ensembles and the full AutoML preprocessing, because it only calls the model.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pandas as pd
import polars as pl

from .adapter import ModelAdapter, as_polars

_BATCH_ROWS = 200_000          # rows per model call when evaluating coalitions


@dataclasses.dataclass
class Explanation:
    """Shapley values for a set of rows.

    ``values`` is ``(rows, features)`` for regression and binary models (contributions to
    the predicted value / positive-class probability) and ``(rows, features, classes)``
    for multiclass models (contributions to each class probability).
    """

    values: np.ndarray
    base_value: np.ndarray | float
    features: list
    data: pd.DataFrame
    output: str
    method: str
    classes: list | None = None
    positive_class: object = None

    @property
    def is_multiclass(self) -> bool:
        return self.values.ndim == 3

    @property
    def predictions(self) -> np.ndarray:
        """Model output reconstructed from the explanation (base + contributions)."""
        return self.base_value + self.values.sum(axis=1)

    def _class(self, class_):
        if class_ is None:
            return None
        for i, c in enumerate(self.classes or []):
            if c == class_ or str(c) == str(class_):
                return i
        raise ValueError(f"{class_!r} is not one of {self.classes}")

    def global_importance(self, class_=None) -> pl.DataFrame:
        """Mean |SHAP| per feature (the share of total attribution), with the direction of effect.

        ``direction`` is the rank correlation between a numeric feature's value and its
        contribution: positive means higher values push the output up."""
        vals = self.values
        if self.is_multiclass:
            c = self._class(class_)
            vals = np.abs(vals).sum(axis=2) if c is None else vals[:, :, c]
        mean_abs = np.abs(vals).mean(axis=0)
        total = mean_abs.sum() or 1.0
        directions = []
        for j, f in enumerate(self.features):
            col = self.data[f]
            direction = None
            if pd.api.types.is_numeric_dtype(col) and not (self.is_multiclass and class_ is None):
                x, v = col.to_numpy(dtype=float), vals[:, j]
                ok = ~np.isnan(x)
                if ok.sum() > 2 and np.ptp(x[ok]) > 0 and np.ptp(v[ok]) > 0:
                    rx = pd.Series(x[ok]).rank().to_numpy()
                    rv = pd.Series(v[ok]).rank().to_numpy()
                    direction = round(float(np.corrcoef(rx, rv)[0, 1]), 3)
            directions.append(direction)
        frame = pl.DataFrame({"feature": self.features, "mean_abs_shap": mean_abs,
                              "share": mean_abs / total,
                              "mean_shap": vals.mean(axis=0) if vals.ndim == 2 else np.zeros(len(mean_abs)),
                              "direction": pl.Series(directions, dtype=pl.Float64)})
        return frame.sort("mean_abs_shap", descending=True)

    def for_row(self, i: int, class_=None, top: int | None = None) -> pl.DataFrame:
        """Contributions for one explained row, largest first.

        For multiclass models the predicted class is explained unless ``class_`` is given."""
        vals = self.values[i]
        if self.is_multiclass:
            c = self._class(class_)
            if c is None:
                c = int(np.argmax(self.base_value + vals.sum(axis=0)))
            vals = vals[:, c]
        row = self.data.iloc[i]
        frame = pl.DataFrame({"feature": self.features,
                              "value": [_show(row[f]) for f in self.features],
                              "contribution": vals}).with_columns(
            pl.col("contribution").abs().alias("_abs")).sort("_abs", descending=True).drop("_abs")
        return frame.head(top) if top else frame

    def explain_row(self, i: int, class_=None, top: int = 5) -> str:
        """A one-paragraph, human-readable explanation of one prediction."""
        frame = self.for_row(i, class_)
        if self.is_multiclass:
            c = self._class(class_)
            pred = self.base_value + self.values[i].sum(axis=0)
            c = int(np.argmax(pred)) if c is None else c
            output, base, value = f"P(class = {(self.classes or [])[c]})", np.asarray(self.base_value)[c], pred[c]
        else:
            output, base, value = self.output, float(self.base_value), float(self.predictions[i])
        parts = [f"{r['feature']} = {r['value']} ({r['contribution']:+.3g})"
                 for r in frame.head(top).iter_rows(named=True)]
        rest = float(frame["contribution"][top:].sum()) if frame.height > top else 0.0
        tail = f", other features {rest:+.3g}" if frame.height > top else ""
        return f"{output} = {value:.4g} (average {base:.4g}): " + ", ".join(parts) + tail

    def to_frame(self, class_=None) -> pl.DataFrame:
        """One row per explained row, one column per feature (``shap_<feature>``)."""
        vals = self.values
        if self.is_multiclass:
            c = self._class(class_)
            if c is None:
                raise ValueError("Pass class_=... for a multiclass explanation.")
            vals = vals[:, :, c]
        return pl.DataFrame({f"shap_{f}": vals[:, j] for j, f in enumerate(self.features)})

    def to_shap(self):
        """The same values as a ``shap.Explanation`` (requires ``pip install shap``), for shap's plots."""
        try:
            import shap
        except ImportError as exc:
            raise ImportError("to_shap() needs the shap package: pip install shap") from exc
        base = np.broadcast_to(self.base_value, (len(self.values),) + np.shape(self.base_value))
        return shap.Explanation(values=self.values, base_values=np.array(base), data=self.data.to_numpy(),
                                feature_names=list(self.features))

    def summary(self, top: int = 10) -> str:
        imp = self.global_importance().head(top)
        lines = [f"SHAP explanation of {self.output} over {len(self.values):,} rows ({self.method})"]
        for r in imp.iter_rows(named=True):
            d = r["direction"]
            arrow = "" if d is None or abs(d) < 0.2 else ("  (higher value -> higher output)" if d > 0 else "  (higher value -> lower output)")
            lines.append(f"  {r['feature']:<28} {r['mean_abs_shap']:.4f}  ({r['share']:.0%}){arrow}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _show(v):
    if isinstance(v, float):
        return "missing" if v != v else f"{v:.4g}"
    return str(v)


# ── the estimator ───────────────────────────────────────────────────────────

def _output_fn(ad: ModelAdapter, pos: int | None):
    if not ad.is_classification:
        return lambda X: np.asarray(ad.predict(X), dtype=float)[:, None]
    if pos is not None:
        return lambda X: ad.proba(X)[:, [pos]]
    return ad.proba


def _coalitions(p: int, exact: bool, n_permutations: int, rng):
    """Masks (which features come from the explained row) and how to turn their values into
    Shapley values. Returns ``(masks, combine)`` with ``combine(v) -> (p, k)``."""
    if exact:
        ints = np.arange(2 ** p)
        masks = ((ints[:, None] >> np.arange(p)) & 1).astype(bool)
        size = masks.sum(axis=1)
        weight = np.array([math.factorial(s) * math.factorial(p - s - 1) / math.factorial(p)
                           for s in range(p)])

        def combine_exact(v):
            phi = np.zeros((p, v.shape[1]))
            for i in range(p):
                without = ints[((ints >> i) & 1) == 0]
                phi[i] = (weight[size[without]][:, None] * (v[without | (1 << i)] - v[without])).sum(axis=0)
            return phi
        return masks, combine_exact

    half = max(1, (n_permutations + 1) // 2)
    perms = []
    for _ in range(half):                         # antithetic pairs reduce variance
        perm = rng.permutation(p)
        perms += [perm, perm[::-1]]
    steps = np.zeros((len(perms), p + 1, p), dtype=bool)
    for m, perm in enumerate(perms):
        for j in range(1, p + 1):
            steps[m, j, perm[:j]] = True
    masks, inverse = np.unique(steps.reshape(-1, p), axis=0, return_inverse=True)
    inverse = inverse.reshape(len(perms), p + 1)

    def combine(v):
        phi = np.zeros((p, v.shape[1]))
        for m, perm in enumerate(perms):
            vals = v[inverse[m]]                  # (p + 1, k): output as features are added
            phi[perm] += vals[1:] - vals[:-1]
        return phi / len(perms)
    return masks, combine


def shapley_values(f, X: pd.DataFrame, background: pd.DataFrame, *, exact_max_features: int = 8,
                   n_permutations: int = 10, seed: int = 42):
    """Shapley values of ``f`` (frame -> ``(n, k)`` outputs) for every row of ``X``.

    Returns ``(values (n, p, k), base (k,), method)``."""
    cols = list(X.columns)
    p, B = len(cols), len(background)
    exact = p <= exact_max_features
    masks, combine = _coalitions(p, exact, n_permutations, np.random.default_rng(seed))
    U = len(masks)
    bg = {c: background[c].to_numpy() for c in cols}
    xs = {c: X[c].to_numpy() for c in cols}
    dtypes = X.dtypes
    rows_per_call = max(1, _BATCH_ROWS // (U * B))
    values: list = []
    base = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = f(background).mean(axis=0)
        for start in range(0, len(X), rows_per_call):
            idx = np.arange(start, min(start + rows_per_call, len(X)))
            r = len(idx)
            frame = {}
            for j, c in enumerate(cols):
                use_x = np.tile(np.repeat(masks[:, j], B), r)              # (r * U * B,)
                x_vals = np.repeat(xs[c][idx], U * B)
                bg_vals = np.tile(bg[c], r * U)
                frame[c] = np.where(use_x, x_vals, bg_vals)
            batch = pd.DataFrame(frame).astype(dtypes.to_dict())
            out = f(batch)
            k = out.shape[1]
            v = out.reshape(r, U, B, k).mean(axis=2)                        # (r, U, k)
            values.extend(combine(v[i]) for i in range(r))
    orderings = 2 * max(1, (n_permutations + 1) // 2)
    method = "exact Shapley" if exact else f"permutation Shapley ({orderings} orderings)"
    return np.stack(values), base, method


def _explain_X(ad: ModelAdapter, X: pd.DataFrame, background: pd.DataFrame, *, pos_label=None,
               exact_max_features=8, n_permutations=10, seed=42) -> Explanation:
    pos = ad.positive_index(pos_label) if ad.task == "binary" else None
    f = _output_fn(ad, pos)
    values, base, method = shapley_values(f, X, background, exact_max_features=exact_max_features,
                                          n_permutations=n_permutations, seed=seed)
    if values.shape[2] == 1:
        values, base = values[:, :, 0], float(base[0])
    return Explanation(values=values, base_value=base, features=list(X.columns),
                       data=X.reset_index(drop=True), output=ad.output_name(pos), method=method,
                       classes=ad.classes, positive_class=ad.classes[pos] if pos is not None else None)


def _sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.reset_index(drop=True)
    return frame.sample(n=n, random_state=seed).reset_index(drop=True)


def explain(model, df, *, target: str | None = None, rows: int | None = 100, background=None,
            n_background: int = 20, n_permutations: int = 10, exact_max_features: int = 8,
            pos_label=None, seed: int = 42) -> Explanation:
    """SHAP values for (up to ``rows`` randomly chosen) rows of ``df``.

    Features are the model's inputs after AutoML preprocessing (e.g. one-hot columns made by
    the feature pipeline). ``background`` (default: a sample of ``df``) is the reference the
    contributions are measured against; its mean output is the ``base_value``.
    Pass ``rows=None`` to explain every row.
    """
    ad = ModelAdapter.of(model, target)
    X_all = ad.X(as_polars(df))
    X = X_all if rows is None else _sample(X_all, rows, seed)
    bg = _sample(ad.X(as_polars(background)) if background is not None else X_all, n_background, seed + 1)
    return _explain_X(ad, X.reset_index(drop=True), bg, pos_label=pos_label,
                      exact_max_features=exact_max_features, n_permutations=n_permutations, seed=seed)


def partial_dependence(model, df, feature: str, *, target: str | None = None, grid: int = 20,
                       sample: int = 500, pos_label=None, seed: int = 42) -> pl.DataFrame:
    """Average model output as one feature is swept across its range (others left as they are).

    Numeric features use ``grid`` quantile points; categorical features use their values."""
    ad = ModelAdapter.of(model, target)
    X = _sample(ad.X(as_polars(df)), sample, seed)
    if feature not in X.columns:
        raise ValueError(f"'{feature}' is not a model feature. Features: {list(X.columns)}")
    col = X[feature]
    if pd.api.types.is_numeric_dtype(col):
        points = np.unique(np.nanquantile(col.to_numpy(dtype=float), np.linspace(0, 1, grid)))
    else:
        points = col.value_counts().index[:grid].to_numpy()
    pos = ad.positive_index(pos_label) if ad.task == "binary" else None
    f = _output_fn(ad, pos)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for v in points:
            out = f(X.assign(**{feature: v}).astype(X.dtypes.to_dict())).mean(axis=0)
            row = {"value": v.item() if hasattr(v, "item") else v}
            if out.shape[0] == 1:
                row["mean_output"] = float(out[0])
            else:
                row.update({f"P({c})": float(o) for c, o in zip(ad.classes, out)})
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)
