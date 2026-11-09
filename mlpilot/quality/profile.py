"""Automated EDA: column profiles, semantic types, correlations and plain-English insights."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
from pathlib import Path

import polars as pl

from .. import _rustcore
from .._detect import (CODE_TOKENS, DATE_RE, NUM_RE, PHONE_TOKENS, SENTINELS, URL_RE, floats,
                       is_num, is_temporal, is_text, real_text, tokens)
from .checks import QualityReport, check_quality, jsonable

SEMANTIC_TYPES = ("empty", "constant", "boolean", "binary", "identifier", "numeric", "discrete",
                  "datetime", "numeric_text", "datetime_text", "email", "phone", "url",
                  "categorical", "text", "other")


def _num(v, digits=6):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else round(v, digits)


def semantic_type(series: pl.Series, name: str | None = None) -> str:
    """What a column *means*, beyond its storage type (see ``SEMANTIC_TYPES``)."""
    name = series.name if name is None else name
    dtype, n = series.dtype, series.len()
    nn = series.drop_nulls()
    if nn.len() == 0:
        return "empty"
    u = nn.n_unique()
    if u <= 1:
        return "constant"
    if dtype == pl.Boolean:
        return "boolean"
    if is_temporal(dtype):
        return "datetime"
    if is_num(dtype):
        if u == 2:
            return "binary"
        if dtype.is_integer() and n > 10 and u / n > 0.95:
            return "identifier"
        if dtype.is_integer() and u <= 20:
            return "discrete"
        return "numeric"
    if not is_text(dtype):
        return "other"
    real = real_text(series)
    if real.len() == 0:
        return "empty"
    t = tokens(name)
    if t & PHONE_TOKENS:
        return "phone"
    if {"email", "mail"} & t or real.str.contains("@", literal=True).mean() >= 0.5:
        return "email"
    if real.str.contains(URL_RE).mean() >= 0.8:
        return "url"
    if (not t & CODE_TOKENS and real.str.contains(NUM_RE).mean() >= 0.9
            and real.str.contains(r"^[-+]?0\d").mean() <= 0.1):
        return "numeric_text"
    if real.str.contains(DATE_RE).mean() >= 0.6:
        return "datetime_text"
    if u == 2:
        return "binary"
    avg_len = real.str.len_chars().mean() or 0
    avg_words = real.str.count_matches(r"\S+").mean() or 0
    if n > 10 and u / n > 0.95:
        return "text" if avg_len > 30 else "identifier"
    if avg_len > 50 or (u / real.len() > 0.5 and avg_words > 3):
        return "text"
    return "categorical"


@dataclasses.dataclass
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: str
    count: int
    nulls: int
    null_pct: float
    unique: int
    unique_pct: float
    top_values: list
    numeric: dict | None = None
    text: dict | None = None
    temporal: dict | None = None

    def to_dict(self) -> dict:
        return jsonable(dataclasses.asdict(self))


@dataclasses.dataclass
class Profile:
    """Result of :func:`profile`. Print it for an overview; ``profile["col"]`` for one column."""

    rows: int
    columns: int
    memory_mb: float
    missing_cells: int
    missing_pct: float
    duplicate_rows: int
    created_at: str
    column_profiles: dict
    correlations: list
    insights: list
    quality: QualityReport | None = None

    def __getitem__(self, column: str) -> ColumnProfile:
        return self.column_profiles[column]

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("rows", "columns", "memory_mb", "missing_cells",
                                            "missing_pct", "duplicate_rows", "created_at",
                                            "correlations", "insights")}
        d["column_profiles"] = {c: p.to_dict() for c, p in self.column_profiles.items()}
        d["quality"] = self.quality.to_dict() if self.quality else None
        return jsonable(d)

    def to_json(self, path=None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_frame(self) -> pl.DataFrame:
        """One row per column with the key statistics."""
        rows = []
        for p in self.column_profiles.values():
            num = p.numeric or {}
            txt = p.text or {}
            rows.append({
                "column": p.name, "dtype": p.dtype, "semantic_type": p.semantic_type,
                "count": p.count, "nulls": p.nulls, "null_pct": p.null_pct,
                "unique": p.unique, "unique_pct": p.unique_pct,
                "top_value": str(p.top_values[0][0]) if p.top_values else None,
                "mean": num.get("mean"), "std": num.get("std"), "min": num.get("min"),
                "median": num.get("median"), "max": num.get("max"), "skew": num.get("skew"),
                "mean_length": txt.get("mean_length"),
                "quality_score": self.quality.column_scores.get(p.name) if self.quality else None,
            })
        return pl.DataFrame(rows, infer_schema_length=None)

    def summary(self) -> str:
        lines = [f"Profile: {self.rows:,} rows x {self.columns} columns | "
                 f"{self.memory_mb:.2f} MB | {self.missing_pct:.1f}% missing cells | "
                 f"{self.duplicate_rows:,} duplicate rows"]
        if self.quality:
            q = self.quality
            lines.append(f"Quality score: {q.score:.1f}/100 (grade {q.grade}) | {len(q.issues)} issues")
        lines.append(f"{'column':<24}{'type':<15}{'semantic':<15}{'nulls':>8}{'unique':>9}")
        for p in self.column_profiles.values():
            lines.append(f"{p.name[:23]:<24}{p.dtype[:14]:<15}{p.semantic_type:<15}"
                         f"{p.null_pct:>7.1f}%{p.unique:>9,}")
        if self.insights:
            lines.append("Insights:")
            lines += [f"  - {i}" for i in self.insights]
        return "\n".join(lines)

    __str__ = summary

    def __repr__(self) -> str:
        return self.summary()


def _numeric_stats(nn: pl.Series) -> dict:
    f = nn.cast(pl.Float64)
    vals = f.to_list()
    out = {"min": _num(f.min()), "max": _num(f.max()), "mean": _num(f.mean()),
           "median": _num(f.median()), "std": _num(f.std()),
           "zeros": int((f == 0).sum()), "negatives": int((f < 0).sum())}
    for label, q in (("p01", 0.01), ("p05", 0.05), ("p25", 0.25), ("p75", 0.75),
                     ("p95", 0.95), ("p99", 0.99)):
        out[label] = _num(f.quantile(q))
    out["skew"] = _num(_rustcore.skewness(vals), 4)
    if len(vals) >= 8:
        lo, hi = _rustcore.iqr_bounds(vals)
        out["outliers"] = int(((f < lo) | (f > hi)).sum())
        out["iqr_fences"] = [_num(lo), _num(hi)]
    return out


def _text_stats(nn: pl.Series) -> dict:
    lens = nn.str.len_chars()
    stripped = nn.str.strip_chars()
    cleaned = _rustcore.normalize_whitespace(nn.to_list())
    return {
        "min_length": int(lens.min()), "max_length": int(lens.max()),
        "mean_length": _num(lens.mean(), 2),
        "mean_words": _num(nn.str.count_matches(r"\S+").mean(), 2),
        "empty": int((stripped == "").sum()),
        "placeholders": int(stripped.str.to_lowercase().is_in(SENTINELS).sum()),
        "whitespace_issues": sum(1 for a, b in zip(nn.to_list(), cleaned) if a != b),
    }


def _temporal_stats(nn: pl.Series) -> dict:
    lo, hi = nn.min(), nn.max()
    out = {"min": str(lo), "max": str(hi)}
    try:
        out["range_days"] = (hi - lo).days if isinstance(hi - lo, _dt.timedelta) else None
    except Exception:
        out["range_days"] = None
    return out


def _correlations(df: pl.DataFrame, cols: dict, threshold: float) -> list:
    use = [c for c, p in cols.items()
           if p.numeric is not None and p.semantic_type in ("numeric", "discrete", "binary")]
    if len(use) < 2:
        return []
    matrix = _rustcore.correlation_matrix([floats(df[c]) for c in use])
    pairs = []
    for i in range(len(use)):
        for j in range(i + 1, len(use)):
            r = matrix[i][j]
            if r == r and abs(r) >= threshold:
                pairs.append({"a": use[i], "b": use[j], "r": round(r, 3)})
    return sorted(pairs, key=lambda p: -abs(p["r"]))


def _insights(prof: Profile, limit: int = 12) -> list:
    out = []
    q = prof.quality
    if q is not None:
        high = q.counts["high"]
        out.append(f"Overall data quality is {q.score:.1f}/100 (grade {q.grade}), "
                   f"with {high} high-severity issue{'' if high == 1 else 's'}.")
        weakest = q.weakest_columns()
        if weakest:
            out.append("Weakest columns by quality score: "
                       + ", ".join(f"“{c}” ({s:.0f})" for c, s in weakest) + ".")
    cols = list(prof.column_profiles.values())
    for p in sorted((p for p in cols if p.null_pct >= 20), key=lambda p: -p.null_pct)[:3]:
        out.append(f"“{p.name}” is {p.null_pct:.0f}% empty.")
    if prof.duplicate_rows:
        d = prof.duplicate_rows
        out.append(f"{d:,} row{' is an exact repeat of an earlier row' if d == 1 else 's are exact repeats of earlier rows'}.")
    ids = [p.name for p in cols if p.semantic_type == "identifier"]
    if ids:
        out.append(f"{', '.join(ids[:4])} look{'s' if len(ids) == 1 else ''} like "
                   f"identifier{'' if len(ids) == 1 else 's'} and carr{'ies' if len(ids) == 1 else 'y'} "
                   f"no predictive signal.")
    for p in cols:
        if p.semantic_type == "constant":
            out.append(f"“{p.name}” never changes value.")
        elif p.semantic_type == "numeric_text":
            out.append(f"“{p.name}” holds numbers but is stored as text.")
        elif p.semantic_type == "datetime_text":
            out.append(f"“{p.name}” holds dates but is stored as text.")
    for c in prof.correlations[:3]:
        strength = ("very strongly" if abs(c["r"]) >= 0.8 else
                    "strongly" if abs(c["r"]) >= 0.6 else "moderately")
        direction = "positively" if c["r"] > 0 else "negatively"
        out.append(f"“{c['a']}” and “{c['b']}” are {strength} {direction} correlated (r = {c['r']:.2f}).")
    skewed = sorted((p for p in cols if p.numeric and (p.numeric.get("skew") or 0) > 2),
                    key=lambda p: -p.numeric["skew"])
    for p in skewed[:2]:
        n = p.numeric
        out.append(f"“{p.name}” is heavily right-skewed (skew {n['skew']:.1f}): median "
                   f"{n['median']:g} vs mean {n['mean']:.4g}.")
    for p in cols:
        if (p.semantic_type in ("binary", "boolean", "categorical", "discrete") and p.top_values
                and 2 <= p.unique <= 10 and p.count):
            value, count = p.top_values[0]
            share = count / p.count
            if share >= 0.7:
                out.append(f"“{p.name}” is dominated by {value!r} ({share:.0%} of values).")
    return out[:limit]


def profile(df, *, top_n: int = 10, correlation_threshold: float = 0.5,
            quality: bool = True) -> Profile:
    """Profile a DataFrame: per-column statistics, semantic types, correlations, insights and
    (by default) the full data-quality report from :func:`check_quality`."""
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    rows = df.height
    cols = {}
    for c, dtype in df.schema.items():
        s = df[c]
        nn = s.drop_nulls()
        u = nn.n_unique()
        top = []
        if nn.len():
            counts = nn.value_counts()
            try:
                counts = counts.sort(["count", c], descending=[True, False])
            except Exception:
                counts = counts.sort("count", descending=True)
            top = [[v, int(k)] for v, k in zip(counts[c].head(top_n).to_list(),
                                                counts["count"].head(top_n).to_list())]
        cp = ColumnProfile(name=c, dtype=str(dtype), semantic_type=semantic_type(s, c),
                           count=nn.len(), nulls=s.null_count(),
                           null_pct=round(100 * s.null_count() / max(rows, 1), 2),
                           unique=u, unique_pct=round(100 * u / max(rows, 1), 2), top_values=top)
        if nn.len():
            if is_num(dtype):
                cp.numeric = _numeric_stats(nn)
            elif is_text(dtype):
                cp.text = _text_stats(nn)
            elif is_temporal(dtype):
                cp.temporal = _temporal_stats(nn)
        cols[c] = cp

    try:
        duplicates = rows - df.unique().height
    except Exception:
        duplicates = 0
    missing = sum(p.nulls for p in cols.values())
    prof = Profile(
        rows=rows, columns=df.width, memory_mb=round(df.estimated_size("mb"), 3),
        missing_cells=missing, missing_pct=round(100 * missing / max(rows * df.width, 1), 2),
        duplicate_rows=duplicates,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        column_profiles=cols, correlations=_correlations(df, cols, correlation_threshold),
        insights=[], quality=check_quality(df) if quality else None)
    prof.insights = _insights(prof)
    return prof
