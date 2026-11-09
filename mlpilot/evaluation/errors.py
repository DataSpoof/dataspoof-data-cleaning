"""Error analysis: which predictions are worst, and which segments of the data the model fails on."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import polars as pl

from ..quality.checks import jsonable
from ._slicing import bh_adjust, segments, worse_p_value
from .adapter import ModelAdapter, as_polars

_ROW = "__row__"


@dataclasses.dataclass
class ErrorReport:
    """Where a model goes wrong.

    ``slices`` ranks data segments by excess error versus the whole dataset; a slice is
    ``significant`` when it is worse than the rest beyond chance (one-sided test, Benjamini–
    Hochberg corrected across all slices tested). ``worst`` lists the individual worst
    predictions (most confident mistakes / largest errors) with their input values.
    """

    task: str
    model: str
    target: str
    n_rows: int
    metric: str                       # error_rate (classification) | mae (regression)
    overall: float
    min_size: int
    slices: pl.DataFrame
    worst: pl.DataFrame
    confusions: pl.DataFrame | None = None

    @property
    def problem_slices(self) -> pl.DataFrame:
        return self.slices.filter(pl.col("significant"))

    def summary(self, top: int = 8) -> str:
        label = "error rate" if self.metric == "error_rate" else "MAE"
        fmt = (lambda v: f"{v:.1%}") if self.metric == "error_rate" else (lambda v: f"{v:.4g}")
        lines = [f"Error analysis of {self.model} on {self.n_rows:,} rows - overall {label} {fmt(self.overall)}"]
        problems = self.problem_slices
        if problems.height:
            lines.append(f"  {problems.height} segment(s) significantly worse than average:")
            for r in problems.head(top).iter_rows(named=True):
                lines.append(f"    {r['slice']:<40} {label} {fmt(r['error'])} ({r['lift']:.1f}x)  "
                             f"n={r['size']}  q={r['q_value']:.3g}")
        else:
            lines.append("  No segment is significantly worse than average.")
            for r in self.slices.head(3).iter_rows(named=True):
                lines.append(f"    weakest: {r['slice']:<33} {label} {fmt(r['error'])} ({r['lift']:.1f}x) n={r['size']}")
        if self.confusions is not None and self.confusions.height:
            pairs = ", ".join(f"{r['actual']} -> {r['predicted']} ({r['count']})"
                              for r in self.confusions.head(3).iter_rows(named=True))
            lines.append(f"  Most common mistakes: {pairs}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        for k, v in out.items():
            if isinstance(v, pl.DataFrame):
                out[k] = v.to_dicts()
        return jsonable(out)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __repr__(self) -> str:
        return self.summary()


def _slice_row(column, label, mask, err, overall, regression_res=None):
    a, b = err[mask], err[~mask]
    mean = float(a.mean())
    row = {"slice": label, "column": column, "size": int(mask.sum()), "share": round(mask.mean(), 4),
           "error": round(mean, 6), "lift": round(mean / overall, 4) if overall > 0 else (math.inf if mean > 0 else 1.0),
           "excess": round((mean - overall) * mask.sum(), 4), "p_value": worse_p_value(a, b)}
    if regression_res is not None:
        row["bias"] = round(float(regression_res[mask].mean()), 6)
    return row


def error_analysis(model, df, *, target: str | None = None, columns=None, top: int = 20,
                   min_size: int | None = None, bins: int = 5, max_categories: int = 20,
                   intersections: bool = True, alpha: float = 0.05) -> ErrorReport:
    """Find the worst predictions and the data segments where the model underperforms.

    Segments come from the *original* columns of ``df`` (``columns`` to restrict them): one
    per category, quantile bins for numbers, and missing values; with ``intersections`` the
    worst segments are also crossed in pairs ("Sex = male & Pclass = 3"). Segments smaller
    than ``min_size`` rows (default: max(10, 1% of rows)) are ignored.
    """
    ad = ModelAdapter.of(model, target)
    data = ad.labelled(as_polars(df).with_row_index(_ROW))
    rows = data[_ROW].to_numpy()
    data = data.drop(_ROW)
    n = data.height
    if n == 0:
        raise ValueError("No labelled rows to analyse.")
    X, y = ad.X(data), ad.y(data)
    min_size = min_size if min_size is not None else max(10, math.ceil(0.01 * n))

    res = None
    if ad.is_classification:
        proba = ad.proba(X)
        pred = ad.predict(X)
        err = (pred != y).astype(float)
        metric = "error_rate"
        names = [ad.classes[i] for i in range(len(ad.classes))]
        p_true = proba[np.arange(n), y]
        order = np.lexsort((p_true, -err))         # mistakes first, most confident first
        worst = pl.DataFrame({"row": rows[order], "actual": [names[i] for i in y[order]],
                              "predicted": [names[i] for i in pred[order]],
                              "confidence": np.round(proba.max(axis=1)[order], 4),
                              "p_actual": np.round(p_true[order], 4)})
        wrong = err.astype(bool)
        confusions = (pl.DataFrame({"actual": [str(names[i]) for i in y[wrong]],
                                    "predicted": [str(names[i]) for i in pred[wrong]]})
                      .group_by("actual", "predicted").len("count")
                      .with_columns((pl.col("count") / max(1, wrong.sum())).round(4).alias("share_of_errors"))
                      .sort("count", "actual", descending=[True, False]))
        worst = worst.head(min(top, int(wrong.sum())))
    else:
        pred = ad.predict(X)
        res = pred - y
        err = np.abs(res)
        metric, confusions = "mae", None
        order = np.argsort(-err)
        worst = pl.DataFrame({"row": rows[order], "actual": y[order], "predicted": pred[order],
                              "error": res[order]}).head(top)
    overall = float(err.mean())
    # attach the input values of the worst rows
    pos = {r: i for i, r in enumerate(rows)}
    context = data.drop(ad.target)[[pos[r] for r in worst["row"].to_list()]]
    context = context.rename({c: f"input_{c}" for c in context.columns if c in worst.columns})
    worst = worst.hstack(context.get_columns()) if context.width else worst

    cols = columns if columns is not None else [c for c in data.columns if c != ad.target]
    tested = []
    for c in cols:
        for label, mask in segments(data[c], bins=bins, max_categories=max_categories):
            if mask.sum() >= min_size and (~mask).sum() >= 2:
                tested.append((c, label, mask))
    found = [_slice_row(c, label, mask, err, overall, res) for c, label, mask in tested]

    if intersections and found:
        ranked = sorted(range(len(found)), key=lambda i: -found[i]["excess"])
        lead = [i for i in ranked if found[i]["lift"] > 1][:8]
        for a_i, a in enumerate(lead):
            for b in lead[a_i + 1:]:
                ca, cb = tested[a], tested[b]
                if ca[0] == cb[0]:
                    continue
                mask = ca[2] & cb[2]
                if mask.sum() >= min_size and (~mask).sum() >= 2:
                    found.append(_slice_row(f"{ca[0]} & {cb[0]}", f"{ca[1]} & {cb[1]}", mask, err, overall, res))

    q = bh_adjust([r["p_value"] for r in found])
    for r, qv in zip(found, q):
        r["p_value"], r["q_value"] = round(r["p_value"], 6), round(qv, 6)
        r["significant"] = bool(qv < alpha and r["error"] > overall)
    schema = {"slice": pl.String, "column": pl.String, "size": pl.Int64, "share": pl.Float64,
              "error": pl.Float64, "lift": pl.Float64, "excess": pl.Float64, "p_value": pl.Float64}
    if res is not None:
        schema["bias"] = pl.Float64
    schema.update({"q_value": pl.Float64, "significant": pl.Boolean})
    slices = pl.DataFrame(found, schema=schema) if found else pl.DataFrame(schema=schema)
    slices = slices.sort(["significant", "excess"], descending=[True, True])
    return ErrorReport(task=ad.task, model=ad.name, target=ad.target, n_rows=n, metric=metric,
                       overall=round(overall, 6), min_size=min_size, slices=slices, worst=worst,
                       confusions=confusions)
