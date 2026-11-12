"""Data compression: smaller column types without changing values, and quantized vectors.

>>> small, rep = optimize_dtypes(df)          # Int64 ids -> Int32, 0/1 flags -> Int8, prices -> Float32, ...
>>> print(rep.summary())                      # 41.2 MB -> 12.9 MB (3.2x); one row per changed column
>>> pipe = ac.Pipeline([..., OptimizeDtypes()])        # learn the types on training data, reuse them

>>> q, params = quantize_vectors(embeddings, "int8")    # 8x smaller than float64
>>> approx = dequantize_vectors(q, params)

``optimize_dtypes`` never changes a value unless you ask it to:

* integers move to the smallest signed type that holds the column's min and max;
* floats (``floats="safe"``, the default) become Float32 only when every value survives the
  round trip at the precision the data is written in: prices like 19.99 do, 123456789.12 does
  not (Float32 keeps about 7 significant digits) and stays Float64. Columns of continuous
  measurements stay Float64 too; pass ``floats="float32"`` to convert them anyway (the report
  shows the largest change), or ``floats=None`` to leave floats alone;
* text with few distinct values becomes Categorical (``categorical`` = max distinct/rows).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import polars as pl

from .core import Step, register_step

_SIGNED: list = [(pl.Int8, -(2 ** 7), 2 ** 7 - 1), (pl.Int16, -(2 ** 15), 2 ** 15 - 1),
           (pl.Int32, -(2 ** 31), 2 ** 31 - 1), (pl.Int64, -(2 ** 63), 2 ** 63 - 1)]
_UNSIGNED: list = [(pl.UInt8, 0, 2 ** 8 - 1), (pl.UInt16, 0, 2 ** 16 - 1), (pl.UInt32, 0, 2 ** 32 - 1)]
_F32_MAX = float(np.finfo(np.float32).max)


def _int_type(lo, hi, unsigned: bool):
    if lo is None:                                             # all null
        return pl.Int8
    if unsigned and lo >= 0:
        for t, a, b in _UNSIGNED:
            if hi <= b:
                return t
    for t, a, b in _SIGNED:
        if a <= lo and hi <= b:
            return t
    return None


def _decimals(x: np.ndarray, max_decimals: int = 6):
    """Smallest number of decimals d with round(x, d) == x for every value, or None (continuous)."""
    for d in range(max_decimals + 1):
        scaled = x * 10.0 ** d
        if np.all(np.abs(scaled - np.round(scaled)) <= 1e-9 * np.maximum(1.0, np.abs(scaled))):
            return d
    return None


def _float_plan(s: pl.Series, mode: str):
    """``(Float32 or None, max abs change)`` for a float column."""
    x = s.drop_nulls().to_numpy().astype(np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return pl.Float32, 0.0
    if np.abs(x).max() > _F32_MAX:
        return None, None
    err = float(np.abs(x.astype(np.float32).astype(np.float64) - x).max())
    if mode == "float32":
        return pl.Float32, err
    d = _decimals(x)
    if d is None:                                              # continuous values: Float32 would round them
        return None, err
    return (pl.Float32, err) if err < 0.5 * 10.0 ** -d else (None, err)


@dataclasses.dataclass
class DtypeReport:
    frame: pl.DataFrame               # column, before, after, bytes_before, bytes_after, max_change
    bytes_before: int
    bytes_after: int

    @property
    def ratio(self) -> float:
        return self.bytes_before / max(self.bytes_after, 1)

    @property
    def changed(self) -> dict:
        """``{column: new dtype}`` for every column that changed type."""
        return {r["column"]: r["after"] for r in self.frame.iter_rows(named=True) if r["after"] != r["before"]}

    def summary(self) -> str:
        mb = lambda b: f"{b / 1e6:.1f} MB" if b >= 1e5 else f"{b / 1e3:.1f} KB"   # noqa: E731
        lines = [f"{mb(self.bytes_before)} -> {mb(self.bytes_after)} ({self.ratio:.1f}x)"]
        for r in self.frame.filter(pl.col("after") != pl.col("before")).iter_rows(named=True):
            change = "" if not r["max_change"] else f", values change by up to {r['max_change']:.3g}"
            lines.append(f"  {r['column']}: {r['before']} -> {r['after']}{change}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"DtypeReport({len(self.changed)} columns changed, {self.bytes_before:,} -> "
                f"{self.bytes_after:,} bytes, {self.ratio:.2f}x)")


def plan_dtypes(df: pl.DataFrame, *, floats: str | None = "safe", categorical: float | None = 0.5,
                unsigned: bool = False, exclude=()) -> dict:
    """``{column: (new dtype, max value change)}`` for the columns :func:`optimize_dtypes` would change."""
    if floats not in ("safe", "float32", None):
        raise ValueError("floats must be 'safe', 'float32' or None")
    plan: dict = {}
    for c, t in df.schema.items():
        if c in exclude:
            continue
        s = df[c]
        if t.is_integer():
            new = _int_type(s.min(), s.max(), unsigned)
            if new is not None and new != t:
                plan[c] = (new, 0.0)
        elif t == pl.Float64 and floats is not None:
            new, err = _float_plan(s, floats)
            if new is not None:
                plan[c] = (new, err)
        elif t == pl.String and categorical is not None and len(s) > 0:
            if s.n_unique() <= max(1, categorical * len(s)):
                plan[c] = (pl.Categorical, 0.0)
    return plan


def optimize_dtypes(df: pl.DataFrame, *, floats: str | None = "safe", categorical: float | None = 0.5,
                    unsigned: bool = False, exclude=()) -> tuple:
    """Shrink ``df``'s column types; returns ``(smaller_df, DtypeReport)`` (see the module docstring).
    ``unsigned=True`` also uses UInt types for non-negative integers (beware: subtracting
    unsigned columns can wrap around)."""
    plan = plan_dtypes(df, floats=floats, categorical=categorical, unsigned=unsigned, exclude=exclude)
    out = df.with_columns([pl.col(c).cast(t) for c, (t, _) in plan.items()]) if plan else df
    rows = [{"column": c, "before": str(df.schema[c]), "after": str(out.schema[c]),
             "bytes_before": int(df[c].estimated_size()), "bytes_after": int(out[c].estimated_size()),
             "max_change": plan.get(c, (None, 0.0))[1]} for c in df.columns]
    frame = pl.DataFrame(rows, schema={"column": pl.Utf8, "before": pl.Utf8, "after": pl.Utf8,
                                       "bytes_before": pl.Int64, "bytes_after": pl.Int64, "max_change": pl.Float64})
    return out, DtypeReport(frame, int(df.estimated_size()), int(out.estimated_size()))


_DTYPES = {"Int8": pl.Int8, "Int16": pl.Int16, "Int32": pl.Int32, "Int64": pl.Int64, "UInt8": pl.UInt8,
           "UInt16": pl.UInt16, "UInt32": pl.UInt32, "Float32": pl.Float32, "Categorical": pl.Categorical}


@register_step
class OptimizeDtypes(Step):
    """Pipeline step: learn compact column types on training data, apply them to new data.

    On new data a column is only cast when its values still fit the learned type (an id above
    the training range, say); otherwise it keeps its type and a warning says so, so scoring
    never fails or silently wraps around."""

    name = "optimize_dtypes"

    def __init__(self, *, floats: str | None = "safe", categorical: float | None = 0.5, unsigned: bool = False,
                 exclude=(), fit_only=None):
        super().__init__(fit_only=fit_only, floats=floats, categorical=categorical, unsigned=unsigned,
                         exclude=list(exclude))

    def _fit(self, df):
        p = self.params
        plan = plan_dtypes(df, floats=p["floats"], categorical=p["categorical"], unsigned=p["unsigned"],
                           exclude=set(p["exclude"]) | {getattr(self, "_pipeline_target", None)})
        return {"dtypes": {c: str(t) for c, (t, _) in plan.items()}}

    def _transform(self, df):
        casts, skipped = [], []
        for c, name in self.state_["dtypes"].items():
            if c not in df.columns:
                continue
            t, s = _DTYPES[name], df[c]
            if t.is_integer() and s.dtype.is_integer() and len(s.drop_nulls()):
                bounds = next(((a, b) for tt, a, b in _SIGNED + _UNSIGNED if tt == t), None)
                if bounds and not (bounds[0] <= s.min() and s.max() <= bounds[1]):
                    skipped.append(c)
                    continue
            if t == pl.Float32 and s.dtype.is_float() and len(s.drop_nulls()) and \
                    float(s.abs().max()) > _F32_MAX:
                skipped.append(c)
                continue
            casts.append(pl.col(c).cast(t))
        if skipped:
            warnings.warn(f"OptimizeDtypes: {skipped} have values outside the types learned in training; "
                          "kept their original types", stacklevel=2)
        return df.with_columns(casts) if casts else df


# ── vectors (embeddings) ────────────────────────────────────────────────────
VECTOR_MODES = ("float32", "float16", "int8")


def quantize_vectors(x, mode: str = "int8", *, scale=None) -> tuple:
    """Quantize a 2-D array of vectors (embeddings). Returns ``(quantized, params)``.

    * ``float32`` / ``float16``: 2x / 4x smaller than float64; float16 keeps ~3 significant digits.
    * ``int8``: 8x smaller. Each dimension is scaled symmetrically to -127..127 using its largest
      absolute value (``scale``, learned from ``x`` unless given, e.g. from training data);
      values beyond the training range are clipped. Use ``dequantize_vectors`` to get floats back.
    """
    x = np.asarray(x, dtype=np.float64)
    if mode == "float32":
        return x.astype(np.float32), {"mode": mode}
    if mode == "float16":
        return x.astype(np.float16), {"mode": mode}
    if mode != "int8":
        raise ValueError(f"mode must be one of {VECTOR_MODES}")
    if scale is None:
        scale = np.abs(x).max(axis=0) / 127.0 if x.size else np.ones(x.shape[1] if x.ndim == 2 else 1)
    scale = np.where(np.asarray(scale, dtype=np.float64) > 0, scale, 1.0)
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, {"mode": mode, "scale": [float(v) for v in np.ravel(scale)]}


def dequantize_vectors(q, params: dict) -> np.ndarray:
    """Float vectors back from :func:`quantize_vectors` output."""
    q = np.asarray(q)
    if params["mode"] == "int8":
        return q.astype(np.float64) * np.asarray(params["scale"])
    return q.astype(np.float64)
