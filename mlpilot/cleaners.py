"""Stateful cleaning steps built on the pipeline core.

Each step learns its rules in ``fit`` (stored as plain JSON in ``state_``) and applies only
those rules in ``transform``. Cleaning a test set or a production batch therefore never
learns from that batch, and a saved pipeline reproduces training-time cleaning exactly.

Quick start::

    pipe = ac.default_cleaning_pipeline()
    train_clean = pipe.fit_transform(train_df)
    pipe.save("cleaning.json")
    test_clean = ac.Pipeline.load("cleaning.json").transform(test_df)
"""

from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import polars as pl

from . import _rustcore
from ._detect import CODE_TOKENS as _CODE_TOKENS
from ._detect import DATE_RE as _DATE_RE
from ._detect import NUM_RE as _NUM_RE
from ._detect import floats as _floats
from ._detect import is_num as _is_num
from ._detect import is_text as _is_text
from ._detect import real_text as _real_text
from ._detect import tokens as _tokens
from .core import Pipeline, Step, register_step

__all__ = [
    "DropColumns", "ConvertNumericText", "ParseDates", "Impute", "ClipOutliers",
    "RemoveOutliers", "LogTransformSkewed", "Scale", "GroupRareCategories",
    "EncodeCategorical", "DropDuplicates", "DropCollinear",
    "default_cleaning_pipeline", "DEFAULT_CLEANING_STEPS",
]

# ── helpers ──────────────────────────────────────────────────────────────────

def _cols(df: pl.DataFrame, columns, pred) -> list:
    return [c for c, t in df.schema.items() if (columns is None or c in columns) and pred(t)]


def _plain(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _fences(s: pl.Series, method: str, k: float, lower_q: float, upper_q: float):
    """Outlier fences for the non-null values ``s``; ``None`` if there is too little data."""
    if s.len() < 8:
        return None
    if method == "iqr":
        # The Rust backend returns the 1.5×IQR fences; recover Q1/Q3 to apply any k.
        lo15, hi15 = _rustcore.iqr_bounds([float(v) for v in s.to_list()])
        iqr = (hi15 - lo15) / 4.0
        q1, q3 = lo15 + 1.5 * iqr, hi15 - 1.5 * iqr
        return [q1 - k * iqr, q3 + k * iqr]
    return [float(s.quantile(lower_q) or 0.0), float(s.quantile(upper_q) or 0.0)]


def _learn_fences(df, columns, method, k, lower_q, upper_q) -> dict:
    bounds = {}
    for c in _cols(df, columns, _is_num):
        s = df[c].drop_nulls()
        if s.n_unique() <= 2:          # flags / binary columns have no outliers
            continue
        b = _fences(s, method, k, lower_q, upper_q)
        if b and b[1] > b[0]:          # zero-width fences (e.g. IQR = 0) would flatten the column
            bounds[c] = b
    return bounds


# ── column-level steps ───────────────────────────────────────────────────────

@register_step
class DropColumns(Step):
    """Learn which columns to drop (mostly missing, constant, ID-like, duplicated) and drop them."""

    name = "drop_columns"

    def __init__(self, *, max_missing=0.5, drop_constant=True, drop_id_like=True,
                 id_uniqueness=0.95, drop_duplicate_columns=True, columns=None, keep=None,
                 fit_only=None):
        super().__init__(fit_only=fit_only, max_missing=max_missing, drop_constant=drop_constant,
                         drop_id_like=drop_id_like, id_uniqueness=id_uniqueness,
                         drop_duplicate_columns=drop_duplicate_columns, columns=columns, keep=keep)

    def _fit(self, df):
        p, n = self.params, df.height or 1
        keep = set(p["keep"] or [])
        reasons = {c: "requested" for c in (p["columns"] or []) if c in df.columns}
        for c, dtype in df.schema.items():
            if c in keep or c in reasons:
                continue
            s = df[c]
            missing = s.null_count() / n
            if p["max_missing"] is not None and missing > p["max_missing"]:
                reasons[c] = f"{missing:.0%} missing"
                continue
            n_unique = s.drop_nulls().n_unique()
            if p["drop_constant"] and n_unique <= 1:
                reasons[c] = "constant"
                continue
            if (p["drop_id_like"] and df.height > 10 and not dtype.is_float()
                    and dtype != pl.Boolean and n_unique / n >= p["id_uniqueness"]):
                reasons[c] = f"ID-like ({n_unique / n:.0%} unique)"
        if p["drop_duplicate_columns"]:
            buckets: dict = {}
            for c in df.columns:
                if c in reasons or c in keep:
                    continue
                s = df[c]
                key = (str(s.dtype), s.null_count(), int(np.sum(s.hash().to_numpy(), dtype=np.uint64)))
                twin = next((o for o in buckets.get(key, []) if s.equals(df[o])), None)
                if twin is not None:
                    reasons[c] = f"duplicate of '{twin}'"
                else:
                    buckets.setdefault(key, []).append(c)
        return {"drop": list(reasons), "reasons": reasons}

    def _transform(self, df):
        drop = [c for c in self.state_["drop"] if c in df.columns]
        return df.drop(drop) if drop else df


@register_step
class DropCollinear(Step):
    """Learn numeric columns that are near-copies of another (|correlation| > threshold) and drop them."""

    name = "drop_collinear"

    def __init__(self, *, threshold=0.95, columns=None, fit_only=None):
        super().__init__(fit_only=fit_only, threshold=threshold, columns=columns)

    def _fit(self, df):
        cols = [c for c in _cols(df, self.params["columns"], _is_num)
                if df[c].drop_nulls().n_unique() > 2]
        if len(cols) < 2:
            return {"drop": []}
        drop = _rustcore.collinear_to_drop([_floats(df[c]) for c in cols], cols,
                                           float(self.params["threshold"]))
        return {"drop": list(drop)}

    def _transform(self, df):
        drop = [c for c in self.state_["drop"] if c in df.columns]
        return df.drop(drop) if drop else df


# ── type repair ──────────────────────────────────────────────────────────────

@register_step
class ConvertNumericText(Step):
    """Learn text columns that hold numbers ("1,200", "$50", "(30)") and convert them, keeping signs."""

    name = "convert_numeric_text"

    def __init__(self, *, min_ratio=0.9, columns=None, fit_only=None):
        super().__init__(fit_only=fit_only, min_ratio=min_ratio, columns=columns)

    def _fit(self, df):
        explicit = self.params["columns"] is not None
        convert = []
        for c in _cols(df, self.params["columns"], _is_text):
            if not explicit and _tokens(c) & _CODE_TOKENS:   # zip codes, phone numbers, IDs
                continue
            real = _real_text(df[c])
            if real.len() == 0 or (not explicit and real.len() < 5):
                continue
            is_num = real.str.contains(_NUM_RE)
            if is_num.mean() < self.params["min_ratio"]:
                continue
            if not explicit and real.filter(is_num).str.contains(r"^[-+]?0\d").mean() > 0.1:
                continue                                       # leading zeros => codes
            convert.append(c)
        return {"columns": convert}

    def _transform(self, df):
        exprs = []
        for c in self.state_["columns"]:
            if c not in df.columns:
                continue
            text = pl.col(c).cast(pl.String).str.strip_chars()
            negative = text.str.contains(r"^\(.*\)$")         # accounting style (30) = -30
            number = text.str.replace_all(r"[\s$₹€£¥,%()]", "").cast(pl.Float64, strict=False)
            exprs.append(pl.when(negative).then(-number).otherwise(number).alias(c))
        return df.with_columns(exprs) if exprs else df


@register_step
class ParseDates(Step):
    """Learn text columns that hold dates and parse them to native date/datetime types."""

    name = "parse_dates"

    def __init__(self, *, min_success=0.8, columns=None, fit_only=None):
        super().__init__(fit_only=fit_only, min_success=min_success, columns=columns)

    def _fit(self, df):
        from .steps import parse_dates as _parse
        explicit = self.params["columns"] is not None
        candidates = []
        for c in _cols(df, self.params["columns"], _is_text):
            real = _real_text(df[c])
            if real.len() == 0 or (not explicit and real.len() < 5):
                continue
            # Require date-shaped values first: dateutil alone would happily read "113803" as a time.
            if explicit or real.str.contains(_DATE_RE).mean() >= self.params["min_success"]:
                candidates.append(c)
        if not candidates:
            return {"columns": []}
        with _quiet():
            parsed = _parse(df.select(candidates), candidates, min_success=self.params["min_success"])
        return {"columns": [c for c in candidates if parsed[c].dtype != pl.String]}

    def _transform(self, df):
        from .steps import parse_dates as _parse
        cols = [c for c in self.state_["columns"] if c in df.columns and df[c].dtype == pl.String]
        if not cols:
            return df
        with _quiet():
            return _parse(df, cols, min_success=self.params["min_success"])


# ── missing values ───────────────────────────────────────────────────────────

@register_step
class Impute(Step):
    """Learn fill values on training data (median/mean/mode/constant) and reuse them on new data."""

    name = "impute"
    _NUMERIC = ("median", "mean", "constant", "none")
    _CATEGORICAL = ("mode", "constant", "none")

    def __init__(self, *, numeric="median", categorical="mode", fill_value_numeric=0,
                 fill_value_categorical="missing", add_indicators=False,
                 indicator_min_missing=0.0, columns=None, fit_only=None):
        if numeric not in self._NUMERIC:
            raise ValueError(f"numeric must be one of {self._NUMERIC}")
        if categorical not in self._CATEGORICAL:
            raise ValueError(f"categorical must be one of {self._CATEGORICAL}")
        super().__init__(fit_only=fit_only, numeric=numeric, categorical=categorical,
                         fill_value_numeric=fill_value_numeric,
                         fill_value_categorical=fill_value_categorical,
                         add_indicators=add_indicators,
                         indicator_min_missing=indicator_min_missing, columns=columns)

    def _fit(self, df):
        p = self.params
        n = df.height or 1
        fill, indicators = {}, []
        for c, dtype in df.schema.items():
            if p["columns"] is not None and c not in p["columns"]:
                continue
            s = df[c]
            value = None
            if _is_num(dtype) and p["numeric"] != "none":
                if p["numeric"] == "median":
                    value = s.median()
                elif p["numeric"] == "mean":
                    value = s.mean()
                else:
                    value = p["fill_value_numeric"]
                if value is not None and dtype.is_integer():
                    value = int(round(value))          # keep integer columns integer
            elif (_is_text(dtype) or dtype == pl.Boolean) and p["categorical"] != "none":
                if p["categorical"] == "mode":
                    counts = s.drop_nulls().value_counts()
                    if counts.height:
                        value = counts.sort(["count", c], descending=[True, False])[c][0]
                elif _is_text(dtype):
                    value = p["fill_value_categorical"]
            value = _plain(value)
            if value is None:
                continue
            fill[c] = value
            missing = s.null_count() / n
            if p["add_indicators"] and missing > 0 and missing >= p["indicator_min_missing"]:
                indicators.append(c)
        return {"fill": fill, "indicators": indicators}

    def _transform(self, df):
        flags = [pl.col(c).is_null().alias(f"{c}_was_missing")
                 for c in self.state_["indicators"] if c in df.columns]
        if flags:
            df = df.with_columns(flags)
        # An all-null batch column has dtype Null: fill it with the value's own type.
        fills = [pl.col(c).fill_null(pl.lit(v) if df.schema[c] == pl.Null
                                     else pl.lit(v).cast(df.schema[c], strict=False)).alias(c)
                 for c, v in self.state_["fill"].items() if c in df.columns]
        return df.with_columns(fills) if fills else df


# ── outliers & distributions ─────────────────────────────────────────────────

@register_step
class ClipOutliers(Step):
    """Learn outlier fences (IQR or quantiles) and cap values to them. Keeps every row."""

    name = "clip_outliers"

    def __init__(self, *, method="iqr", k=1.5, lower_quantile=0.01, upper_quantile=0.99,
                 columns=None, fit_only=None):
        if method not in ("iqr", "quantile"):
            raise ValueError("method must be 'iqr' or 'quantile'")
        super().__init__(fit_only=fit_only, method=method, k=k, lower_quantile=lower_quantile,
                         upper_quantile=upper_quantile, columns=columns)

    def _fit(self, df):
        p = self.params
        return {"bounds": _learn_fences(df, p["columns"], p["method"], p["k"],
                                        p["lower_quantile"], p["upper_quantile"])}

    def _transform(self, df):
        exprs = []
        for c, (lo, hi) in self.state_["bounds"].items():
            if c not in df.columns or not _is_num(df.schema[c]):
                continue
            dtype = df.schema[c]
            if dtype.is_integer():
                lo, hi = math.ceil(lo), math.floor(hi)
                # a fence beyond what the type can hold (Int8 column, fence -200) clips nothing
                info = np.iinfo(str(dtype).lower())          # "Int8" -> int8, "UInt16" -> uint16 ...
                lo, hi = max(lo, int(info.min)), min(hi, int(info.max))
                if lo > hi:
                    continue
            exprs.append(pl.col(c).clip(pl.lit(lo).cast(dtype), pl.lit(hi).cast(dtype)).alias(c))
        return df.with_columns(exprs) if exprs else df


@register_step
class RemoveOutliers(Step):
    """Learn IQR fences and drop rows outside them. Runs only while fitting unless fit_only=False."""

    name = "remove_outliers"
    fit_only = True

    def __init__(self, *, k=1.5, columns=None, fit_only=None):
        super().__init__(fit_only=fit_only, k=k, columns=columns)

    def _fit(self, df):
        return {"bounds": _learn_fences(df, self.params["columns"], "iqr", self.params["k"], 0, 1)}

    def row_mask(self, df):
        keep = None
        for c, (lo, hi) in self.state_["bounds"].items():
            if c not in df.columns:
                continue
            inside = pl.col(c).is_null() | ((pl.col(c) >= lo) & (pl.col(c) <= hi))
            keep = inside if keep is None else keep & inside
        if keep is None:
            return pl.Series("_keep", [True] * df.height, dtype=pl.Boolean)
        return df.select(keep.alias("_keep")).to_series()

    def _transform(self, df):
        return df.filter(self.row_mask(df))


@register_step
class LogTransformSkewed(Step):
    """Learn non-negative, right-skewed numeric columns and apply log1p to them."""

    name = "log_transform_skewed"

    def __init__(self, *, threshold=1.0, columns=None, fit_only=None):
        super().__init__(fit_only=fit_only, threshold=threshold, columns=columns)

    def _fit(self, df):
        chosen = []
        for c in _cols(df, self.params["columns"], _is_num):
            s = df[c].drop_nulls()
            if s.len() < 8 or s.n_unique() <= 2 or s.min() < 0:
                continue
            skew = _rustcore.skewness([float(v) for v in s.to_list()])
            if skew == skew and skew > self.params["threshold"]:
                chosen.append(c)
        return {"columns": chosen}

    def _transform(self, df):
        exprs = [pl.col(c).cast(pl.Float64).clip(lower_bound=0).log1p().alias(c)
                 for c in self.state_["columns"] if c in df.columns]
        return df.with_columns(exprs) if exprs else df


@register_step
class Scale(Step):
    """Learn scaling parameters (standard, min-max or robust) on training data and apply them."""

    name = "scale"

    def __init__(self, *, method="standard", columns=None, skip_binary=True, fit_only=None):
        if method not in ("standard", "minmax", "robust"):
            raise ValueError("method must be 'standard', 'minmax' or 'robust'")
        super().__init__(fit_only=fit_only, method=method, columns=columns, skip_binary=skip_binary)

    def _fit(self, df):
        params = {}
        for c in _cols(df, self.params["columns"], _is_num):
            s = df[c].drop_nulls().cast(pl.Float64)
            if s.len() == 0 or (self.params["skip_binary"] and s.n_unique() <= 2):
                continue
            method = self.params["method"]
            if method == "standard":
                center, scale = s.mean(), s.std()
            elif method == "minmax":
                center, scale = s.min(), s.max() - s.min()
            else:
                center, scale = s.median(), s.quantile(0.75) - s.quantile(0.25)
            center, scale = _plain(center), _plain(scale)
            if center is None or not scale:
                continue
            params[c] = [float(center), float(scale)]
        return {"params": params}

    def _transform(self, df):
        exprs = [((pl.col(c).cast(pl.Float64) - center) / scale).alias(c)
                 for c, (center, scale) in self.state_["params"].items() if c in df.columns]
        return df.with_columns(exprs) if exprs else df


# ── categories ───────────────────────────────────────────────────────────────

@register_step
class GroupRareCategories(Step):
    """Learn each column's common categories; rare and unseen values become ``other_label``."""

    name = "group_rare_categories"

    def __init__(self, *, min_frequency=0.01, other_label="Other", max_unique=100,
                 columns=None, fit_only=None):
        super().__init__(fit_only=fit_only, min_frequency=min_frequency, other_label=other_label,
                         max_unique=max_unique, columns=columns)

    def _fit(self, df):
        p, keep = self.params, {}
        for c in _cols(df, p["columns"], _is_text):
            s = df[c].drop_nulls()
            if s.len() == 0:
                continue
            if p["columns"] is None and s.n_unique() > p["max_unique"]:
                continue                                      # free text / identifiers
            counts = s.value_counts()
            kept = counts.filter(pl.col("count") >= p["min_frequency"] * s.len())[c].to_list()
            keep[c] = sorted(kept)
        return {"keep": keep}

    def _transform(self, df):
        other = self.params["other_label"]
        exprs = []
        for c, kept in self.state_["keep"].items():
            if c not in df.columns or not _is_text(df.schema[c]):
                continue
            col = pl.col(c)
            known = col.is_in(list(kept)) if kept else pl.lit(False)
            exprs.append(pl.when(col.is_null()).then(col).when(known).then(col)
                         .otherwise(pl.lit(other)).alias(c))
        return df.with_columns(exprs) if exprs else df


@register_step
class EncodeCategorical(Step):
    """Learn category codes on training data: one-hot, ordinal or frequency encoding."""

    name = "encode_categorical"

    def __init__(self, *, method="onehot", columns=None, max_categories=20, max_unique=100,
                 drop_original=True, fit_only=None):
        if method not in ("onehot", "ordinal", "frequency"):
            raise ValueError("method must be 'onehot', 'ordinal' or 'frequency'")
        super().__init__(fit_only=fit_only, method=method, columns=columns,
                         max_categories=max_categories, max_unique=max_unique,
                         drop_original=drop_original)

    def _fit(self, df):
        p, learned = self.params, {}
        for c in _cols(df, p["columns"], _is_text):
            s = df[c].drop_nulls()
            if s.len() == 0 or (p["columns"] is None and s.n_unique() > p["max_unique"]):
                continue
            counts = s.value_counts().sort(["count", c], descending=[True, False])
            if p["method"] == "onehot":
                learned[c] = counts[c].head(p["max_categories"]).to_list()
            elif p["method"] == "ordinal":
                learned[c] = {v: i for i, v in enumerate(sorted(counts[c].to_list()))}
            else:
                learned[c] = {v: n / s.len() for v, n in zip(counts[c].to_list(), counts["count"].to_list())}
        return {"categories": learned}

    def _transform(self, df):
        p = self.params
        exprs, encoded = [], []
        for c, learned in self.state_["categories"].items():
            if c not in df.columns:
                continue
            encoded.append(c)
            col = pl.col(c).cast(pl.String)
            if p["method"] == "onehot":
                exprs += [(col == cat).fill_null(False).cast(pl.Int8).alias(f"{c}_{cat}")
                          for cat in learned]
            elif p["method"] == "ordinal":
                exprs.append(pl.when(col.is_null()).then(None).otherwise(
                    col.replace_strict(learned, default=-1, return_dtype=pl.Int32)).alias(c))
            else:
                exprs.append(pl.when(col.is_null()).then(None).otherwise(
                    col.replace_strict(learned, default=0.0, return_dtype=pl.Float64)).alias(c))
        if exprs:
            df = df.with_columns(exprs)
        if p["method"] == "onehot" and p["drop_original"]:
            df = df.drop(encoded)
        return df


# ── rows ─────────────────────────────────────────────────────────────────────

@register_step
class DropDuplicates(Step):
    """Drop duplicate rows. Runs only while fitting unless fit_only=False."""

    name = "drop_duplicates"
    fit_only = True
    uses_target = True     # a duplicate is the whole row, label included

    def __init__(self, *, subset=None, keep="first", fit_only=None):
        super().__init__(fit_only=fit_only, subset=subset, keep=keep)

    def _transform(self, df):
        return df.unique(subset=self.params["subset"], keep=self.params["keep"], maintain_order=True)


# ── the default, non-interactive cleaning pipeline ───────────────────────────

DEFAULT_CLEANING_STEPS = [
    ("replace_disguised_missing", {}),
    ("normalize_whitespace", {}),
    ("normalize_unicode", {}),
    ("convert_numeric_text", {}),
    ("parse_dates", {}),
    # De-duplicate while ID columns still exist: after they're dropped, distinct records
    # that share every other value would wrongly look like duplicates.
    ("drop_duplicates", {}),
    ("drop_columns", {"max_missing": 0.6}),
    ("impute", {"numeric": "median", "categorical": "mode", "add_indicators": True,
                "indicator_min_missing": 0.05}),
    ("group_rare_categories", {"min_frequency": 0.01}),
    ("clip_outliers", {"k": 3.0}),
    ("drop_collinear", {"threshold": 0.95}),
]


def default_cleaning_pipeline(name: str = "automated-cleaning", *, on_schema_mismatch: str = "warn",
                              verbose: bool = False, audit_log=None, **overrides) -> Pipeline:
    """The recommended cleaning pipeline: no prompts, fit once, replay anywhere.

    Steps: fix missing-value placeholders → normalise text → repair numbers/dates stored as
    text → drop duplicate rows (training only) → drop useless columns → impute (+ "was
    missing" flags for columns ≥5% missing) → group rare categories → cap extreme outliers
    (3×IQR) → drop near-duplicate numeric columns.

    Override a step by name: ``clip_outliers=False`` removes it,
    ``impute={"numeric": "mean"}`` changes its parameters.
    """
    known = {n for n, _ in DEFAULT_CLEANING_STEPS}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown default step(s) {sorted(unknown)}; choose from {sorted(known)}")
    steps = []
    for step_name, params in DEFAULT_CLEANING_STEPS:
        override = overrides.get(step_name, True)
        if override is False:
            continue
        merged = {**params, **(override if isinstance(override, dict) else {})}  # type: ignore[dict-item]
        steps.append({"step": step_name, "params": merged})
    return Pipeline(steps, name=name, on_schema_mismatch=on_schema_mismatch, verbose=verbose,
                    audit_log=audit_log)
