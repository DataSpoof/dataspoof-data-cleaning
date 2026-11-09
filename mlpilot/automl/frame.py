"""Turning a Polars frame into a model-ready matrix."""

from __future__ import annotations

import polars as pl

from .._detect import is_num, is_temporal

MISSING = "__missing__"


def infer_kinds(df: pl.DataFrame) -> dict:
    """``{column: "num" | "cat"}`` for every column a model can use (others are ignored)."""
    kinds = {}
    for c, t in df.schema.items():
        if is_num(t) or t == pl.Boolean or is_temporal(t) or t == pl.Duration or t == pl.Null:
            kinds[c] = "num"
        elif t in (pl.String, pl.Categorical) or t == pl.Enum:
            kinds[c] = "cat"
    return kinds


def to_model_frame(df: pl.DataFrame, kinds: dict):
    """pandas frame with exactly ``kinds``' columns: numbers as floats, text with a MISSING token.

    Columns missing from ``df`` are added empty, so prediction-time data always matches the
    columns the model was trained on.
    """
    exprs = []
    for c, kind in kinds.items():
        if c not in df.columns:
            exprs.append((pl.lit(None, dtype=pl.Float64) if kind == "num" else pl.lit(MISSING)).alias(c))
            continue
        t = df.schema[c]
        col = pl.col(c)
        if kind == "cat":
            exprs.append(col.cast(pl.String).fill_null(MISSING).alias(c))
        elif t == pl.Duration:
            exprs.append(col.dt.total_seconds().cast(pl.Float64).alias(c))
        elif is_temporal(t) or t == pl.Boolean:
            exprs.append(col.cast(pl.Int64).cast(pl.Float64).alias(c))
        else:
            exprs.append(col.cast(pl.Float64).alias(c))
    return df.select(exprs).to_pandas()


def make_preprocessor(kinds: dict, scale: bool):
    """Impute (+ scale) numbers, one-hot encode text; unknown categories are ignored."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = [c for c, k in kinds.items() if k == "num"]
    cat = [c for c, k in kinds.items() if k == "cat"]
    parts = []
    if num:
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scale:
            steps.append(("scale", StandardScaler()))
        parts.append(("num", SkPipeline(steps), num))
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", max_categories=20,
                                           sparse_output=False), cat))
    if not parts:
        raise ValueError("No usable feature columns left for modelling.")
    return ColumnTransformer(parts, remainder="drop")
