"""Problem-type detection."""

from __future__ import annotations

import dataclasses

import polars as pl

from .._detect import is_temporal, target_kind

TASKS = ("binary", "multiclass", "regression", "clustering")


@dataclasses.dataclass
class TaskInfo:
    task: str                   # binary | multiclass | regression | clustering
    time_series: bool
    time_col: str | None
    n_classes: int | None
    reason: str

    @property
    def is_classification(self) -> bool:
        return self.task in ("binary", "multiclass")


def _time_column(df: pl.DataFrame, exclude) -> str | None:
    """A date/time column that is sorted and mostly unique: rows are a time sequence."""
    for c, dtype in df.schema.items():
        if c in exclude or not is_temporal(dtype):
            continue
        s = df[c].drop_nulls()
        if s.len() >= 10 and s.n_unique() / s.len() >= 0.5 and s.is_sorted():
            return c
    return None


def detect_task(df, target: str | None = None, *, time_col="auto") -> TaskInfo:
    """Work out what kind of problem ``df`` + ``target`` is.

    * no target -> ``clustering``;
    * two distinct target values -> ``binary``; text labels or ≤ 10 whole-number values ->
      ``multiclass``; otherwise ``regression``;
    * ``time_series`` is True when ``time_col`` is given, or (``"auto"``) when a date column is
      sorted and mostly unique, meaning rows form a time sequence. Time series get
      time-ordered cross-validation so the model is never tested on the past.
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    if target is None:
        return TaskInfo("clustering", False, None, None, "no target column given")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' is not in the data")
    y = df[target].drop_nulls()
    task = target_kind(df[target])
    n_unique = y.n_unique()
    if task == "binary":
        reason = f"'{target}' has 2 distinct values"
    elif task == "multiclass":
        reason = (f"'{target}' is text with {n_unique} classes" if not y.dtype.is_numeric()
                  else f"'{target}' has {n_unique} whole-number values")
    else:
        reason = f"'{target}' is numeric with {n_unique} distinct values"

    if time_col == "auto":
        time_col = _time_column(df, {target})
        if time_col:
            reason += f"; rows are ordered by '{time_col}', so this is treated as a time series"
    elif time_col:
        if time_col not in df.columns:
            raise ValueError(f"time_col '{time_col}' is not in the data")
        reason += f"; time series ordered by '{time_col}'"
    else:
        time_col = None
    return TaskInfo(task, bool(time_col), time_col,
                    n_unique if task != "regression" else None, reason)
