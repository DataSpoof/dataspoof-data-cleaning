"""Splitting rows into readable segments ("Pclass = 3", "Age in [20, 28)", "Cabin is missing")."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .._detect import is_num, is_temporal


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def segments(s: pl.Series, *, bins: int = 5, max_categories: int = 20, discrete_max: int = 10):
    """``[(label, mask)]`` for one column, or ``[]`` if it isn't sensibly segmentable.

    Numbers with few distinct values and text/booleans become one segment per value
    (the ``max_categories`` most frequent, the rest grouped as "other"); other numbers are
    cut into ``bins`` quantile bins. Missing values get their own segment. ID-like text
    (mostly unique) and dates are skipped."""
    name, n = s.name, s.len()
    nulls = s.is_null().to_numpy()
    out = []
    values = s.drop_nulls()
    if values.len() == 0:
        return []
    if is_temporal(s.dtype):
        return []
    n_unique = values.n_unique()
    numeric = is_num(s.dtype) and s.dtype != pl.Boolean
    if numeric and n_unique > discrete_max:
        x = s.cast(pl.Float64).to_numpy()
        finite = x[~np.isnan(x)]
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return []
        for b in range(len(edges) - 1):
            last = b == len(edges) - 2
            lo, hi = edges[b], edges[b + 1]
            mask = (x >= lo) & ((x <= hi) if last else (x < hi))
            out.append((f"{name} in [{_fmt(lo)}, {_fmt(hi)}{']' if last else ')'}", mask))
    else:
        if not numeric and n_unique > max(max_categories, 0.5 * values.len()):
            return []                                  # ID-like / free text
        counts = values.value_counts(sort=True)
        top = counts[counts.columns[0]].head(max_categories).to_list()
        for v in top:
            out.append((f"{name} = {_fmt(v)}", (s == v).fill_null(False).to_numpy()))
        if n_unique > max_categories:
            listed = s.to_list()
            keep = set(top)
            out.append((f"{name} = other", np.array([x is not None and x not in keep for x in listed])))
    if nulls.any():
        out.append((f"{name} is missing", nulls))
    return [(label, mask) for label, mask in out if mask.any()]


def normal_sf(z: float) -> float:
    """One-sided p-value P(Z > z)."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def worse_p_value(a: np.ndarray, b: np.ndarray) -> float:
    """p-value that values in ``a`` (a segment) are higher on average than in ``b`` (the rest)."""
    if len(a) < 2 or len(b) < 2:
        return 1.0
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    diff = a.mean() - b.mean()
    if se == 0:
        return 0.0 if diff > 0 else 1.0
    return normal_sf(diff / se)


def bh_adjust(p: list[float]) -> list[float]:
    """Benjamini–Hochberg false-discovery-rate adjusted p-values (q-values)."""
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    q = np.empty(m)
    running = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        running = min(running, p[i] * m / rank)
        q[i] = running
    return q.tolist()
