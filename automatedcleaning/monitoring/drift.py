"""Data drift: how far a new batch's distributions have moved from a reference.

A ``Baseline`` keeps only binned distributions and summary statistics of the reference data
(never raw rows), so it can be saved as JSON and shipped with a model. Drift per column is
measured with the Population Stability Index (PSI: < 0.1 stable, 0.1–0.25 moderate,
> 0.25 major), the Jensen–Shannon distance, a chi-square test, missing-rate change and the
share of categories never seen in the reference.
"""

from __future__ import annotations

import dataclasses
import json
import math
import warnings

import numpy as np
import polars as pl

from .._detect import is_num, is_temporal
from ..quality.checks import jsonable

EPS = 1e-4
MAX_CATEGORIES = 50
OTHER, MISSING = "__other__", "__missing__"


def psi(p: np.ndarray, q: np.ndarray) -> float:
    """Population Stability Index between reference proportions ``p`` and current ``q``."""
    p, q = np.clip(np.asarray(p, float), EPS, None), np.clip(np.asarray(q, float), EPS, None)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def js_distance(p, q) -> float:
    p, q = np.clip(np.asarray(p, float), 0, None), np.clip(np.asarray(q, float), 0, None)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return math.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2))


def severity_of(value: float, moderate: float = 0.1, major: float = 0.25) -> str:
    return "major" if value >= major else ("moderate" if value >= moderate else "none")


@dataclasses.dataclass
class ColumnBaseline:
    name: str
    kind: str                         # numeric | categorical
    proportions: list                 # per bin / category (+ missing last)
    labels: list                      # bin labels or category values (+ MISSING last)
    edges: list | None = None         # interior bin edges (numeric)
    n: int = 0
    mean: float | None = None
    std: float | None = None
    missing_rate: float = 0.0

    def counts(self, s: pl.Series) -> np.ndarray:
        """Counts of a new series in this baseline's bins/categories (+ missing)."""
        nulls = int(s.is_null().sum())
        if self.kind == "numeric":
            x = s.cast(pl.Float64, strict=False).drop_nulls().to_numpy()
            x = x[~np.isnan(x)]
            nulls += int(s.len() - nulls - len(x))
            idx = np.searchsorted(np.asarray(self.edges, float), x, side="right")
            counts = np.bincount(idx, minlength=len(self.edges) + 1).astype(float)
        else:
            values = s.drop_nulls().cast(pl.String).value_counts()
            known = {v: i for i, v in enumerate(self.labels[:-1])}
            counts = np.zeros(len(self.labels) - 1)
            other = known.get(OTHER)
            for v, c in values.iter_rows():
                if v in known:
                    counts[known[v]] += c
                elif other is not None:
                    counts[other] += c
                # else: a value never seen in the reference - reported via unseen_share()
        return np.append(counts, nulls)

    def unseen_share(self, s: pl.Series) -> float:
        if self.kind != "categorical":
            return 0.0
        vals = s.drop_nulls().cast(pl.String)
        if vals.len() == 0:
            return 0.0
        known = set(self.labels[:-1]) - {OTHER}
        has_other = OTHER in self.labels
        if has_other:                      # categories beyond the top ones were seen, just pooled
            return 0.0
        return float((~vals.is_in(list(known))).mean())


def fit_column(s: pl.Series, bins: int = 10) -> ColumnBaseline | None:
    """Baseline for one column, or None for columns drift isn't meaningful for (dates, IDs)."""
    if is_temporal(s.dtype) or s.dtype == pl.Null:
        return None
    n = s.len()
    if n == 0:
        return None
    nulls = int(s.is_null().sum())
    values = s.drop_nulls()
    numeric = is_num(s.dtype) and s.dtype != pl.Boolean and values.n_unique() > 10
    if numeric and s.dtype.is_integer() and values.len() > 50 and values.n_unique() == values.len():
        return None                        # every value unique: a row ID / counter, always "drifts"
    if numeric:
        x = values.cast(pl.Float64).to_numpy()
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return None
        edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
        counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1).astype(float)
        labels = ([f"< {edges[0]:.4g}"] + [f"[{a:.4g}, {b:.4g})" for a, b in zip(edges[:-1], edges[1:])]
                  + [f">= {edges[-1]:.4g}"]) if len(edges) else ["all"]
        props = np.append(counts, nulls) / n
        return ColumnBaseline(s.name, "numeric", props.tolist(), labels + [MISSING], edges.tolist(), n,
                              float(x.mean()), float(x.std()), nulls / n)
    text = values.cast(pl.String)
    if s.dtype in (pl.String, pl.Categorical) and text.n_unique() > 0.9 * max(1, text.len()) and text.len() > 50:
        return None                        # ID-like
    vc = text.value_counts(sort=True)
    cats, counts = vc[vc.columns[0]].to_list(), vc[vc.columns[1]].to_numpy().astype(float)
    labels, props = cats[:MAX_CATEGORIES], list(counts[:MAX_CATEGORIES])
    if len(cats) > MAX_CATEGORIES:
        labels.append(OTHER)
        props.append(float(counts[MAX_CATEGORIES:].sum()))
    props = np.append(np.array(props), nulls) / n
    return ColumnBaseline(s.name, "categorical", props.tolist(), labels + [MISSING], None, n, None, None, nulls / n)


@dataclasses.dataclass
class Baseline:
    columns: dict                      # name -> ColumnBaseline
    schema: dict                       # name -> dtype string
    n_rows: int

    @classmethod
    def fit(cls, df: pl.DataFrame, columns=None, bins: int = 10) -> "Baseline":
        cols = columns if columns is not None else df.columns
        fitted = {}
        for c in cols:
            if c in df.columns:
                b = fit_column(df[c], bins)
                if b is not None:
                    fitted[c] = b
        return cls(fitted, {c: str(df.schema[c]) for c in cols if c in df.columns}, df.height)

    def to_dict(self) -> dict:
        return {"columns": {k: dataclasses.asdict(v) for k, v in self.columns.items()},
                "schema": self.schema, "n_rows": self.n_rows}

    @classmethod
    def from_dict(cls, d: dict) -> "Baseline":
        return cls({k: ColumnBaseline(**v) for k, v in d["columns"].items()}, d["schema"], d["n_rows"])


def compare_column(base: ColumnBaseline, s: pl.Series) -> dict:
    counts = base.counts(s)
    n = counts.sum()
    ref = np.asarray(base.proportions, float)
    cur = counts / n if n else np.zeros_like(ref)
    value = psi(ref, cur) if n else 0.0
    row = {"column": base.name, "kind": base.kind, "psi": round(value, 4),
           "js_distance": round(js_distance(ref, cur), 4), "p_value": None,
           "missing_ref": round(base.missing_rate, 4), "missing_cur": round(float(cur[-1]) if n else 0.0, 4),
           "mean_ref": base.mean, "mean_cur": None, "mean_shift_std": None,
           "unseen_share": round(base.unseen_share(s), 4)}
    if n >= 5:
        from scipy.stats import chisquare
        expected = np.clip(ref, EPS, None)
        expected = expected / expected.sum() * n
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row["p_value"] = float(chisquare(counts, expected).pvalue)
    if base.kind == "numeric":
        x = s.cast(pl.Float64, strict=False).drop_nulls()
        if x.len():
            row["mean_cur"] = float(x.mean())
            if base.std:
                row["mean_shift_std"] = round((row["mean_cur"] - base.mean) / base.std, 4)
    sev = severity_of(value)
    if row["unseen_share"] >= 0.05 and sev == "none":
        sev = "moderate"
    if row["missing_cur"] - row["missing_ref"] >= 0.1:
        sev = "major" if row["missing_cur"] - row["missing_ref"] >= 0.25 else max(sev, "moderate", key=_rank)
    row["severity"] = sev
    return row


def _rank(s):
    return {"none": 0, "moderate": 1, "major": 2}[s]


@dataclasses.dataclass
class DriftReport:
    columns: pl.DataFrame
    missing_columns: list
    new_columns: list
    dtype_changes: dict
    n_reference: int
    n_current: int

    @property
    def drifted(self) -> list:
        return self.columns.filter(pl.col("severity") != "none")["column"].to_list()

    @property
    def major(self) -> list:
        return self.columns.filter(pl.col("severity") == "major")["column"].to_list()

    @property
    def drift_share(self) -> float:
        return len(self.major) / self.columns.height if self.columns.height else 0.0

    def summary(self, top: int = 10) -> str:
        lines = [f"Drift: {len(self.major)} of {self.columns.height} column(s) with major drift, "
                 f"{len(self.drifted) - len(self.major)} moderate ({self.n_current:,} rows vs "
                 f"{self.n_reference:,} reference)"]
        for r in self.columns.filter(pl.col("severity") != "none").head(top).iter_rows(named=True):
            extra = ""
            if r["mean_shift_std"] is not None:
                extra = f", mean {r['mean_ref']:.4g} -> {r['mean_cur']:.4g}"
            if r["unseen_share"]:
                extra += f", {r['unseen_share']:.0%} unseen values"
            if r["missing_cur"] - r["missing_ref"] >= 0.05:
                extra += f", missing {r['missing_ref']:.0%} -> {r['missing_cur']:.0%}"
            lines.append(f"  [{r['severity']}] {r['column']}: PSI {r['psi']:.3f}{extra}")
        if self.missing_columns:
            lines.append(f"  Missing columns: {self.missing_columns}")
        if self.dtype_changes:
            lines.append(f"  Type changes: {self.dtype_changes}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return jsonable({"columns": self.columns.to_dicts(), "missing_columns": self.missing_columns,
                         "new_columns": self.new_columns, "dtype_changes": self.dtype_changes,
                         "n_reference": self.n_reference, "n_current": self.n_current,
                         "drift_share": self.drift_share, "drifted": self.drifted})

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __repr__(self) -> str:
        return self.summary()


def compare(baseline: Baseline, current: pl.DataFrame) -> DriftReport:
    rows = [compare_column(b, current[name]) for name, b in baseline.columns.items() if name in current.columns]
    schema = {"column": pl.String, "kind": pl.String, "psi": pl.Float64, "js_distance": pl.Float64,
              "p_value": pl.Float64, "missing_ref": pl.Float64, "missing_cur": pl.Float64, "mean_ref": pl.Float64,
              "mean_cur": pl.Float64, "mean_shift_std": pl.Float64, "unseen_share": pl.Float64,
              "severity": pl.String}
    frame = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    frame = frame.with_columns(pl.col("severity").replace_strict({"major": 0, "moderate": 1, "none": 2},
                                                                 return_dtype=pl.Int8).alias("_o")) \
        .sort(["_o", "psi"], descending=[False, True]).drop("_o")
    missing = [c for c in baseline.schema if c not in current.columns]
    new = [c for c in current.columns if c not in baseline.schema]
    changes = {c: f"{baseline.schema[c]} -> {current.schema[c]}" for c in baseline.schema
               if c in current.columns and str(current.schema[c]) != baseline.schema[c]
               and not (str(current.schema[c]) == "Null" or baseline.schema[c] == "Null")}
    return DriftReport(frame, missing, new, changes, baseline.n_rows, current.height)


def drift_report(reference, current, columns=None, bins: int = 10) -> DriftReport:
    """Compare two datasets column by column (``reference`` may also be a fitted ``Baseline``)."""
    def pl_(df):
        return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    base = reference if isinstance(reference, Baseline) else Baseline.fit(pl_(reference), columns, bins)
    return compare(base, pl_(current))
