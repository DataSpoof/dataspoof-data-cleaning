"""Fairness and bias checks across groups of a sensitive attribute (sex, age band, region …)."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import polars as pl

from ..quality.checks import jsonable
from ._slicing import segments
from .adapter import ModelAdapter, as_polars

# Default tolerances (common rules of thumb; tune for your domain and jurisdiction).
DEFAULT_LIMITS = {
    "disparate_impact_ratio": 0.8,          # four-fifths rule: min/max selection rate ≥ 0.8
    "demographic_parity_difference": 0.1,
    "equal_opportunity_difference": 0.1,    # TPR gap
    "equalized_odds_difference": 0.1,       # max(TPR gap, FPR gap)
    "predictive_parity_difference": 0.1,    # precision gap
    "accuracy_difference": 0.1,
    "error_ratio": 1.25,                    # regression: worst/best group MAE
}

_EXPLAIN = {
    "disparate_impact_ratio": "lowest / highest group selection rate (four-fifths rule)",
    "demographic_parity_difference": "gap in the share of each group predicted positive",
    "equal_opportunity_difference": "gap in true-positive rate (recall) between groups",
    "equalized_odds_difference": "largest gap in true- or false-positive rate between groups",
    "predictive_parity_difference": "gap in precision between groups",
    "accuracy_difference": "gap in accuracy between groups",
    "error_ratio": "worst / best group mean absolute error",
    "mean_prediction_difference": "gap in average prediction between groups",
    "bias_difference": "gap in average over/under-prediction between groups",
}


@dataclasses.dataclass
class FairnessResult:
    """Fairness of a model with respect to one sensitive attribute (or an intersection)."""

    attribute: str
    groups: pl.DataFrame
    disparities: dict
    flags: list
    reference_group: str | None
    excluded_groups: list
    used_by_model: bool
    notes: list

    @property
    def passed(self) -> bool:
        return not self.flags


@dataclasses.dataclass
class FairnessReport:
    task: str
    model: str
    target: str
    n_rows: int
    positive_class: object
    results: list
    proxies: pl.DataFrame
    limits: dict

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def flags(self) -> list:
        return [f for r in self.results for f in r.flags]

    def __getitem__(self, attribute) -> FairnessResult:
        for r in self.results:
            if r.attribute == attribute:
                return r
        raise KeyError(attribute)

    def summary(self) -> str:
        what = f" (positive = {self.positive_class!r})" if self.positive_class is not None else ""
        lines = [f"Fairness of {self.model} on {self.n_rows:,} rows{what}: "
                 + ("no disparities beyond the limits" if self.passed else f"{len(self.flags)} flag(s)")]
        for r in self.results:
            lines.append(f"  {r.attribute}{'  [used as a model feature]' if r.used_by_model else ''}")
            for k, v in r.disparities.items():
                if v is not None:
                    lines.append(f"    {k:<32} {v:.3f}")
            for f in r.flags:
                lines.append(f"    ! {f}")
            for note in r.notes:
                lines.append(f"    - {note}")
        if self.proxies.height:
            flagged = self.proxies.filter(pl.col("flag"))
            if flagged.height:
                lines.append("  Possible proxies (features strongly associated with a sensitive attribute):")
                for row in flagged.head(8).iter_rows(named=True):
                    lines.append(f"    {row['feature']} ~ {row['attribute']}  (Cramér's V {row['cramers_v']:.2f})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        def result(r):
            d = {f.name: getattr(r, f.name) for f in dataclasses.fields(r)}
            d["groups"] = r.groups.to_dicts()
            return d
        return jsonable({"task": self.task, "model": self.model, "target": self.target,
                         "n_rows": self.n_rows, "positive_class": self.positive_class,
                         "passed": self.passed, "limits": self.limits,
                         "results": [result(r) for r in self.results],
                         "proxies": self.proxies.to_dicts()})

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __repr__(self) -> str:
        return self.summary()


# ── group metrics ───────────────────────────────────────────────────────────

def _rate(mask):
    return float(mask.mean()) if mask.size else None


def _binary_group(yb, pb, p):
    pos, neg = yb == 1, yb == 0
    row = {"base_rate": _rate(yb), "selection_rate": _rate(pb),
           "tpr": _rate(pb[pos]) if pos.any() else None, "fpr": _rate(pb[neg]) if neg.any() else None,
           "precision": _rate(yb[pb == 1]) if pb.any() else None, "accuracy": _rate(yb == pb),
           "mean_score": float(p.mean())}
    if pos.any() and neg.any():
        from sklearn.metrics import roc_auc_score
        row["auc"] = float(roc_auc_score(yb, p))
    else:
        row["auc"] = None
    return row


def _spread(values, ratio=False):
    vals = [v for v in values if v is not None and v == v]
    if len(vals) < 2:
        return None
    if ratio:
        hi, lo = max(vals), min(vals)
        return (lo / hi) if hi > 0 else None
    return max(vals) - min(vals)


def _assess(groups: list, task: str, limits: dict, y_std: float):
    disp, flags = {}, []
    col = lambda k: [g[k] for g in groups]          # noqa: E731
    if task == "regression":
        maes = col("mae")
        disp["mean_prediction_difference"] = _spread(col("mean_prediction"))
        disp["bias_difference"] = _spread(col("bias"))
        lo = min(m for m in maes) if maes else None
        disp["error_ratio"] = (max(maes) / lo) if maes and lo and lo > 0 and len(maes) > 1 else None
        if disp["error_ratio"] is not None and disp["error_ratio"] > limits["error_ratio"]:
            flags.append(f"error_ratio {disp['error_ratio']:.2f} > {limits['error_ratio']}: "
                         f"errors are much larger for some groups")
        if disp["bias_difference"] is not None and y_std > 0 and disp["bias_difference"] > 0.25 * y_std:
            flags.append(f"bias_difference {disp['bias_difference']:.4g} is over 0.25 × std of the target: "
                         "the model over-predicts some groups relative to others")
        return disp, flags
    if "selection_rate" in groups[0]:
        disp["demographic_parity_difference"] = _spread(col("selection_rate"))
        disp["disparate_impact_ratio"] = _spread(col("selection_rate"), ratio=True)
        tpr, fpr = _spread(col("tpr")), _spread(col("fpr"))
        disp["equal_opportunity_difference"] = tpr
        disp["equalized_odds_difference"] = max(v for v in (tpr, fpr) if v is not None) \
            if (tpr is not None or fpr is not None) else None
        disp["predictive_parity_difference"] = _spread(col("precision"))
    disp["accuracy_difference"] = _spread(col("accuracy"))
    for k, v in disp.items():
        if v is None:
            continue
        limit = limits.get(k)
        if limit is None:
            continue
        bad = v < limit if k == "disparate_impact_ratio" else v > limit
        if bad:
            op = "<" if k == "disparate_impact_ratio" else ">"
            flags.append(f"{k} {v:.3f} {op} {limit}: {_EXPLAIN[k]}")
    return disp, flags


def _evaluate_attribute(name, seg, ad, y, pred, proba, res, pos, min_group_size, limits,
                        reference, y_std, used):
    rows, excluded = [], []
    for label, mask in seg:
        group = label.split(" = ", 1)[1] if " = " in label else label
        if " & " in name:
            group = label
        n = int(mask.sum())
        row = {"group": group, "size": n, "share": round(mask.mean(), 4)}
        if ad.is_classification:
            if pos is not None:
                yb, pb = (y[mask] == pos).astype(int), (pred[mask] == pos).astype(int)
                row.update(_binary_group(yb, pb, proba[mask, pos]))
            else:
                row["accuracy"] = _rate(y[mask] == pred[mask])
        else:
            row.update({"mean_actual": float(y[mask].mean()), "mean_prediction": float((y + res)[mask].mean()),
                        "mae": float(np.abs(res[mask]).mean()),
                        "rmse": float(np.sqrt((res[mask] ** 2).mean())), "bias": float(res[mask].mean())})
        row["evaluated"] = n >= min_group_size
        rows.append(row)
        if n < min_group_size:
            excluded.append(group)
    assessed = [r for r in rows if r["evaluated"]]
    notes = []
    if len(assessed) < 2:
        disp, flags = {}, []
        notes.append(f"Fewer than two groups have at least {min_group_size} rows; nothing to compare.")
    else:
        disp, flags = _assess(assessed, ad.task, limits, y_std)
    if excluded:
        notes.append(f"Groups with fewer than {min_group_size} rows are reported but not compared: "
                     f"{', '.join(map(str, excluded[:6]))}{' …' if len(excluded) > 6 else ''}")
    ref = None
    if assessed:
        ref = reference if reference is not None else max(assessed, key=lambda r: r["size"])["group"]
        ref_row = next((r for r in rows if str(r["group"]) == str(ref)), None)
        if ref_row is not None and "selection_rate" in ref_row and ref_row["selection_rate"]:
            for r in rows:
                r["selection_rate_ratio"] = (r["selection_rate"] / ref_row["selection_rate"]
                                             if r["selection_rate"] is not None else None)
    if pos is not None and len(assessed) >= 2:
        base = _spread([r["base_rate"] for r in assessed])
        if base is not None and base > 0.1 and disp.get("demographic_parity_difference"):
            notes.append(f"Actual positive rates differ by {base:.2f} between groups, so part of the "
                         "selection-rate gap reflects the data; equal opportunity / equalized odds "
                         "compare groups on equal footing.")
    if used:
        notes.append("This attribute (or an encoding of it) is an input to the model.")
    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.col("group").cast(pl.String))
    floats = [c for c, t in frame.schema.items() if t == pl.Float64]
    frame = frame.with_columns([pl.col(c).round(4) for c in floats])
    return FairnessResult(attribute=name, groups=frame, disparities={k: (round(v, 4) if v is not None else None)
                                                                      for k, v in disp.items()},
                          flags=flags, reference_group=None if ref is None else str(ref),
                          excluded_groups=[str(e) for e in excluded], used_by_model=used, notes=notes)


# ── proxies ─────────────────────────────────────────────────────────────────

def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected Cramér's V between two label arrays (0 = unrelated, 1 = identical)."""
    frame = pl.DataFrame({"a": a, "b": b})
    n = frame.height
    table = frame.group_by("a", "b").len()
    ra = frame.group_by("a").len().rename({"len": "na"})
    rb = frame.group_by("b").len().rename({"len": "nb"})
    r, k = ra.height, rb.height
    if n < 2 or r < 2 or k < 2:
        return 0.0
    t = table.join(ra, on="a").join(rb, on="b")
    observed = t["len"].to_numpy().astype(float)
    expected = (t["na"] * t["nb"]).to_numpy() / n
    # cells with zero observations contribute their expected count
    chi2 = float(((observed - expected) ** 2 / expected).sum() + (n - expected.sum()))
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(math.sqrt(phi2c / denom)) if denom > 0 else 0.0


def _labels(series: pl.Series, bins: int) -> np.ndarray | None:
    seg = segments(series, bins=bins, max_categories=20)
    if len(seg) < 2:
        return None
    out = np.full(series.len(), "", dtype=object)
    for label, mask in seg:
        out[mask & (out == "")] = label
    return out.astype(str)


def find_proxies(df: pl.DataFrame, sensitive: list, *, exclude=(), threshold: float = 0.5,
                 bins: int = 5) -> pl.DataFrame:
    """Features strongly associated with a sensitive attribute (Cramér's V on binned values).

    A proxy lets a model discriminate even when the attribute itself is removed."""
    rows = []
    cache = {}
    for attr in sensitive:
        a = _labels(df[attr], bins)
        if a is None:
            continue
        for c in df.columns:
            if c in sensitive or c in exclude:
                continue
            if c not in cache:
                cache[c] = _labels(df[c], bins)
            b = cache[c]
            if b is None:
                continue
            v = cramers_v(a, b)
            rows.append({"attribute": attr, "feature": c, "cramers_v": round(v, 4), "flag": v >= threshold})
    schema = {"attribute": pl.String, "feature": pl.String, "cramers_v": pl.Float64, "flag": pl.Boolean}
    frame = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    return frame.sort("cramers_v", descending=True)


# ── entry point ─────────────────────────────────────────────────────────────

def fairness_report(model, df, sensitive, *, target: str | None = None, pos_label=None,
                    threshold: float | None = None, min_group_size: int = 20, bins: int = 4,
                    intersectional: bool = True, reference=None, limits: dict | None = None,
                    proxy_threshold: float = 0.5) -> FairnessReport:
    """Check a model for bias across the groups of each ``sensitive`` column.

    Classification (binary, or multiclass with ``pos_label``): per-group selection rate,
    TPR, FPR, precision, accuracy and AUC; flags demographic parity, the four-fifths
    (disparate impact) rule, equal opportunity, equalized odds, predictive parity and
    accuracy gaps beyond ``limits``. Regression: per-group error and bias.
    Numeric attributes (e.g. age) are cut into ``bins`` quantile bands. Groups smaller than
    ``min_group_size`` are reported but not compared. With two or more attributes and
    ``intersectional``, their combinations are checked too (e.g. sex × age band). Also
    lists features that act as **proxies** for a sensitive attribute.
    """
    sensitive = [sensitive] if isinstance(sensitive, str) else list(sensitive)
    ad = ModelAdapter.of(model, target)
    data = ad.labelled(df)
    missing = [s for s in sensitive if s not in data.columns]
    if missing:
        raise ValueError(f"Sensitive column(s) not in the data: {missing}")
    if ad.target in sensitive:
        raise ValueError("The target can't be a sensitive attribute.")
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    X, y = ad.X(data), ad.y(data)
    proba, res, pos = None, None, None
    if ad.is_classification:
        proba = ad.proba(X)
        if ad.task == "binary":
            pos = ad.positive_index(pos_label)
        elif pos_label is not None:
            pos = ad.class_index(pos_label)
        if threshold is not None:
            if ad.task != "binary":
                raise ValueError("threshold only applies to binary classification.")
            assert pos is not None
            pred = np.where(proba[:, pos] >= threshold, pos, 1 - pos)
        else:
            pred = ad.predict(X)
    else:
        pred = ad.predict(X)
        res = pred - y
    y_std = float(np.std(y)) if not ad.is_classification else 0.0
    model_cols = set(X.columns)

    def used(attr):
        return attr in model_cols or any(c.startswith(f"{attr}_") for c in model_cols)

    seg_cache = {s: segments(data[s], bins=bins, max_categories=30) for s in sensitive}
    results = []
    for s in sensitive:
        if not seg_cache[s]:
            continue
        results.append(_evaluate_attribute(s, seg_cache[s], ad, y, pred, proba, res, pos, min_group_size,
                                           limits, reference if len(sensitive) == 1 else None, y_std, used(s)))
    if intersectional and len(sensitive) >= 2:
        for i, a in enumerate(sensitive):
            for b in sensitive[i + 1:]:
                seg = [(f"{la} & {lb}", ma & mb) for la, ma in seg_cache[a] for lb, mb in seg_cache[b]]
                seg = [(lab, m) for lab, m in seg if m.any()]
                if seg:
                    results.append(_evaluate_attribute(f"{a} & {b}", seg, ad, y, pred, proba, res, pos,
                                                       min_group_size, limits, None, y_std,
                                                       used(a) or used(b)))
    proxies = find_proxies(data, sensitive, exclude=[ad.target], threshold=proxy_threshold, bins=bins + 1)
    return FairnessReport(task=ad.task, model=ad.name, target=ad.target, n_rows=data.height,
                          positive_class=ad.classes[pos] if pos is not None else None,
                          results=results, proxies=proxies, limits=limits)
