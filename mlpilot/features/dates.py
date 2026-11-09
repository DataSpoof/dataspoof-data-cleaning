"""Date/time feature extraction."""

from __future__ import annotations

import datetime as _dt
import math

import polars as pl

from .._detect import is_temporal
from ..core import Step, register_step

DATE_PARTS = ("year", "month", "day", "dayofweek", "dayofyear", "quarter", "is_weekend",
              "is_month_start", "is_month_end", "hour", "cyclical", "elapsed_days")


@register_step
class DateTimeFeatures(Step):
    """Expand date/datetime columns into calendar parts, cyclical encodings and elapsed days.

    Outputs ``<col>_year``, ``_month``, ``_day``, ``_dayofweek`` (1 = Monday), ``_dayofyear``,
    ``_quarter``, ``_is_weekend``, ``_is_month_start``, ``_is_month_end``, ``_hour`` (only for
    columns that carry a time of day), sine/cosine encodings of month, weekday and hour (so
    December sits next to January), and ``_elapsed_days`` since the earliest training date.
    """

    name = "datetime_features"

    def __init__(self, *, columns=None, parts=None, drop_original=True, fit_only=None):
        parts = list(parts) if parts is not None else list(DATE_PARTS)
        unknown = set(parts) - set(DATE_PARTS)
        if unknown:
            raise ValueError(f"Unknown date part(s) {sorted(unknown)}; choose from {DATE_PARTS}")
        super().__init__(fit_only=fit_only, columns=columns, parts=parts, drop_original=drop_original)

    def _fit(self, df):
        info = {}
        for c, dtype in df.schema.items():
            if self.params["columns"] is not None and c not in self.params["columns"]:
                continue
            if not is_temporal(dtype):
                continue
            s = df[c].drop_nulls()
            has_time = False
            if dtype == pl.Datetime and s.len():
                # hour() etc. are Int8: widen before multiplying (Polars 2 rejects the overflow)
                seconds = (s.dt.hour().cast(pl.Int32) * 3600 + s.dt.minute().cast(pl.Int32) * 60
                           + s.dt.second().cast(pl.Int32))
                has_time = bool(seconds.max() > 0)
            start = s.min() if s.len() else None
            info[c] = {"has_time": has_time, "kind": "date" if dtype == pl.Date else "datetime",
                       "start": start.isoformat() if start is not None else None}
        return {"columns": info}

    def _transform(self, df):
        parts = set(self.params["parts"])
        exprs, done = [], []
        for c, meta in self.state_["columns"].items():
            if c not in df.columns or not is_temporal(df.schema[c]):
                continue
            done.append(c)
            col = pl.col(c)
            month, dow = col.dt.month(), col.dt.weekday()
            simple = {
                "year": col.dt.year(), "month": month, "day": col.dt.day(), "dayofweek": dow,
                "dayofyear": col.dt.ordinal_day(), "quarter": col.dt.quarter(),
                "is_weekend": (dow >= 6).cast(pl.Int8),
                "is_month_start": (col.dt.day() == 1).cast(pl.Int8),
                "is_month_end": (col.dt.day() == col.dt.month_end().dt.day()).cast(pl.Int8),
            }
            for part, expr in simple.items():
                if part in parts:
                    exprs.append(expr.alias(f"{c}_{part}"))
            if meta["has_time"] and "hour" in parts:
                exprs.append(col.dt.hour().alias(f"{c}_hour"))
            if "cyclical" in parts:
                cycles = [("month", month, 12), ("dayofweek", dow, 7)]
                if meta["has_time"]:
                    cycles.append(("hour", col.dt.hour(), 24))
                for label, value, period in cycles:
                    angle = value.cast(pl.Float64) * (2 * math.pi / period)
                    exprs += [angle.sin().alias(f"{c}_{label}_sin"), angle.cos().alias(f"{c}_{label}_cos")]
            if "elapsed_days" in parts and meta["start"]:
                start = _dt.datetime.fromisoformat(meta["start"])
                anchor = pl.lit(start.date() if meta["kind"] == "date" else start).cast(df.schema[c])
                exprs.append(((col - anchor).dt.total_seconds() / 86400).alias(f"{c}_elapsed_days"))
        if exprs:
            df = df.with_columns(exprs)
        if self.params["drop_original"] and done:
            df = df.drop(done)
        return df
