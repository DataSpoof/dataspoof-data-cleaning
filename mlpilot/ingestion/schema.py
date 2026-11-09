"""Schema inference: turn text columns that really hold numbers, dates or booleans into real types.

The result is a fitted, JSON-serialisable pipeline, so every later batch (e.g. incremental
loads) gets exactly the same types instead of being re-guessed batch by batch.
"""

from __future__ import annotations

import contextlib
import io

import polars as pl

from .._detect import is_text
from ..core import Pipeline, Step, register_step

TRUE = {"true", "t", "yes", "y", "on"}
FALSE = {"false", "f", "no", "n", "off"}


@register_step
class InferBooleans(Step):
    """Learn text columns whose values are all yes/no-like and convert them to Boolean."""

    name = "infer_booleans"

    def __init__(self, *, columns=None, fit_only=None):
        super().__init__(fit_only=fit_only, columns=columns)

    def _fit(self, df):
        cols = []
        for c in self.params["columns"] or [c for c, t in df.schema.items() if is_text(t)]:
            if c not in df.columns:
                continue
            values = df[c].drop_nulls().cast(pl.String).str.strip_chars().str.to_lowercase().unique().to_list()
            if values and set(values) <= TRUE | FALSE | {"1", "0"} and set(values) & (TRUE | FALSE):
                cols.append(c)
        return {"columns": cols}

    def _transform(self, df):
        exprs = []
        for c in self.state_["columns"]:
            if c in df.columns and is_text(df.schema[c]):
                v = pl.col(c).cast(pl.String).str.strip_chars().str.to_lowercase()
                exprs.append(pl.when(v.is_in(list(TRUE | {"1"}))).then(True)
                             .when(v.is_in(list(FALSE | {"0"}))).then(False).otherwise(None).alias(c))
        return df.with_columns(exprs) if exprs else df


def type_pipeline() -> Pipeline:
    from ..cleaners import ConvertNumericText, ParseDates
    return Pipeline([ConvertNumericText(), ParseDates(), InferBooleans()], name="ingest-types", verbose=False,
                    on_schema_mismatch="ignore")


def infer_types(df: pl.DataFrame, pipeline: Pipeline | None = None):
    """Return ``(typed_df, fitted_pipeline, casts)``; ``casts`` maps column -> "String -> Float64".

    Pass a previously fitted ``pipeline`` to apply the same conversions without re-learning."""
    before = dict(df.schema)
    with contextlib.redirect_stdout(io.StringIO()):
        if pipeline is None:
            pipeline = type_pipeline()
            out = pipeline.fit_transform(df)
        else:
            out = pipeline.transform(df)
    casts = {c: f"{before[c]} -> {t}" for c, t in out.schema.items() if c in before and before[c] != t}
    return out, pipeline, casts
