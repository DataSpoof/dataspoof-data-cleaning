"""Automatic insights: statistically supported findings phrased in plain English.

Every insight carries the numbers behind it (segment size, values, p-value), so a report or
an LLM narrative can only restate what the data shows. Findings are associations, not causes.
"""

from __future__ import annotations

import dataclasses
import math
import re

import numpy as np
import polars as pl

from .._detect import is_num, is_temporal
from ..evaluation._slicing import bh_adjust, segments


@dataclasses.dataclass
class Insight:
    kind: str                     # driver | trend | change | correlation | concentration | outliers | quality | model
    headline: str                 # one plain-English sentence
    strength: float               # 0-1, used for ranking
    columns: list
    evidence: dict

    def __str__(self) -> str:
        return self.headline


def _num(v, digits=3) -> str:
    if v is None:
        return "n/a"
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.{digits - 1}f}B"
    if a >= 1e6:
        return f"{v / 1e6:.{digits - 1}f}M"
    if a >= 1e4:
        return f"{v:,.0f}"
    return f"{v:.{digits}g}"


_BIN = re.compile(r"^(?P<col>.+) in \[(?P<lo>[^,]+), (?P<hi>[^\]\)]+)(?P<close>[\)\]])$")


def _phrase(label: str, first: bool, last: bool) -> str:
    """'stock in [0, 20)' -> 'stock < 20'; last bin -> 'stock >= 80'; middle -> 'stock is 20-50'."""
    m = _BIN.match(label)
    if not m:
        return label.replace(" = ", " is ")
    col, lo, hi = m["col"], m["lo"], m["hi"]
    if first and not last:
        return f"{col} < {hi}"
    if last and not first:
        return f"{col} >= {lo}"
    return f"{col} is {lo}-{hi}"


def _segments(df: pl.DataFrame, column: str, bins: int):
    seg = segments(df[column], bins=bins, max_categories=15)
    binned = [i for i, (lab, _) in enumerate(seg) if _BIN.match(lab)]
    out = []
    for i, (label, mask) in enumerate(seg):
        out.append((_phrase(label, binned[:1] == [i], binned[-1:] == [i]), mask, label))
    return out


def _welch(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 1.0
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0:
        return 0.0 if a.mean() != b.mean() else 1.0
    z = abs(a.mean() - b.mean()) / se
    return math.erfc(z / math.sqrt(2))


def _target_kind(s: pl.Series) -> str:
    vals = s.drop_nulls()
    if s.dtype == pl.Boolean or vals.n_unique() == 2:
        return "binary"
    if is_num(s.dtype) and vals.n_unique() > 10:
        return "numeric"
    return "categorical" if vals.n_unique() <= 20 else "none"


def _positive(s: pl.Series):
    vals = sorted(s.drop_nulls().unique().to_list(), key=lambda v: (str(type(v)), v))
    for v in vals:
        if str(v).lower() in ("1", "true", "yes", "y", "positive", "churn", "churned", "default", "fraud"):
            return v
    return vals[-1]


def _drivers(df, target, kind, bins, min_support, alpha):
    n = df.height
    y = df[target]
    if kind == "numeric":
        values = y.cast(pl.Float64).to_numpy()
        measure, pos = "average", None
    else:
        pos = _positive(y) if kind == "binary" else y.drop_nulls().mode().to_list()[0]
        values = (y == pos).fill_null(False).cast(pl.Float64).to_numpy()
        measure = "rate"
    known = ~np.isnan(values) if kind == "numeric" else y.is_not_null().to_numpy()
    found = []
    for col in df.columns:
        if col == target:
            continue
        seen = []
        for phrase, mask, label in _segments(df, col, bins):
            if any(np.array_equal(mask, ~prev) for prev in seen):
                continue                  # the mirror image of a finding already tested (2-valued column)
            seen.append(mask)
            m = mask & known
            rest = ~mask & known
            if m.sum() < max(10, min_support * n) or rest.sum() < 10:
                continue
            a, b = values[m], values[rest]
            seg, other = float(a.mean()), float(b.mean())
            if other == 0 and seg == 0:
                continue
            p = _welch(a, b)
            found.append({"column": col, "segment": phrase, "label": label, "size": int(m.sum()),
                          "share": float(m.mean()), "value": seg, "rest": other, "p_value": p})
    q = bh_adjust([f["p_value"] for f in found])
    out = []
    tname = target if kind == "numeric" else f"{target} = {pos}"
    for f, qv in zip(found, q):
        f["q_value"] = qv
        if qv >= alpha:
            continue
        if kind == "numeric":
            if f["rest"] == 0:
                continue
            change = (f["value"] - f["rest"]) / abs(f["rest"])
            if abs(change) < 0.10:
                continue
            word = "higher" if change > 0 else "lower"
            head = (f"{target} is {abs(change):.0%} {word} when {f['segment']} (average {_num(f['value'])} vs "
                    f"{_num(f['rest'])} otherwise; {f['size']:,} rows).")
            effect = min(1.0, abs(change))
        else:
            diff = f["value"] - f["rest"]
            if abs(diff) < 0.05:
                continue
            ratio = f["value"] / f["rest"] if f["rest"] > 0 else math.inf
            times = f" ({ratio:.1f}x)" if math.isfinite(ratio) and (ratio >= 1.5 or ratio <= 0.67) else ""
            head = (f"The rate of {tname} is {f['value']:.0%} when {f['segment']}, vs {f['rest']:.0%} "
                    f"otherwise{times}; {f['size']:,} rows.")
            change = diff
            effect = min(1.0, abs(diff) * 2)
        strength = effect * (0.5 + 0.5 * math.sqrt(min(1.0, f["share"] * 2)))
        out.append(Insight("driver", head, round(strength, 4), [f["column"], target],
                           {**f, "measure": measure, "change": change, "target": target, "positive": pos}))
    return out


def _time_insights(df, target, time_col):
    out = []
    span = df[time_col].max() - df[time_col].min()
    try:
        days = span.days if hasattr(span, "days") else span / 86_400_000_000
    except Exception:
        return out
    every, unit = ("1mo", "month") if days > 120 else (("1w", "week") if days > 28 else ("1d", "day"))
    t = df.select(pl.col(time_col).cast(pl.Datetime) if df[time_col].dtype == pl.Date else pl.col(time_col),
                  pl.col(target).cast(pl.Float64)).drop_nulls().sort(time_col)
    agg = t.group_by_dynamic(time_col, every=every).agg(pl.col(target).sum().alias("total"),
                                                        pl.col(target).mean().alias("mean"), pl.len().alias("n"))
    agg = agg.filter(pl.col("n") > 0)
    if agg.height < 3:
        return out
    # the last period may be partial: compare averages, not totals
    series = agg["mean"].to_numpy()
    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series, 1)
    start, end = intercept, intercept + slope * (len(series) - 1)
    resid = series - (intercept + slope * x)
    r2 = 1 - resid.var() / series.var() if series.var() > 0 else 0
    fmt = "%b %Y" if unit == "month" else "%Y-%m-%d"
    first, last = agg[time_col][0].strftime(fmt), agg[time_col][-1].strftime(fmt)
    if start != 0 and abs((end - start) / abs(start)) >= 0.05 and r2 >= 0.3:
        change = (end - start) / abs(start)
        out.append(Insight("trend", f"Average {target} {'rose' if change > 0 else 'fell'} about {abs(change):.0%} "
                                    f"from {first} to {last} (steady {unit}ly trend, R² {r2:.2f}).",
                           round(min(1.0, abs(change)) * r2, 4), [target, time_col],
                           {"change": change, "start": start, "end": end, "r2": r2, "periods": len(series),
                            "unit": unit, "from": first, "to": last}))
    prev, cur = series[-2], series[-1]
    if prev != 0 and abs((cur - prev) / abs(prev)) >= 0.10 and len(series) >= 4:
        change = (cur - prev) / abs(prev)
        typical = np.median(np.abs(np.diff(series[:-1]) / np.where(series[:-2] == 0, 1, np.abs(series[:-2]))))
        if abs(change) > 2 * typical:
            out.append(Insight("change", f"Average {target} {'jumped' if change > 0 else 'dropped'} {abs(change):.0%} "
                                         f"in the latest {unit} ({last}) vs the one before ({_num(prev)} -> {_num(cur)}).",
                               round(min(1.0, abs(change)), 4), [target, time_col],
                               {"change": change, "previous": prev, "current": cur, "period": last,
                                "typical_change": float(typical)}))
    return out


def _correlations(df, target, threshold=0.6):
    nums = [c for c, t in df.schema.items() if is_num(t) and t != pl.Boolean and df[c].n_unique() > 2]
    if len(nums) < 2:
        return []
    frame = df.select([pl.col(c).cast(pl.Float64) for c in nums[:40]]).drop_nulls()
    if frame.height < 20:
        return []
    corr = np.corrcoef(frame.to_numpy(), rowvar=False)
    out = []
    for i in range(len(nums[:40])):
        for j in range(i + 1, len(nums[:40])):
            r = corr[i, j]
            if np.isfinite(r) and abs(r) >= threshold:
                a, b = nums[i], nums[j]
                involves = target in (a, b)
                if involves and b == target:
                    a, b = b, a
                text = (f"{a} and {b} move {'together' if r > 0 else 'in opposite directions'} (correlation {r:.2f})"
                        + (f": higher {b} goes with {'higher' if r > 0 else 'lower'} {a}." if involves else "."))
                out.append(Insight("correlation", text, round(abs(r) * (0.9 if involves else 0.5), 4), [a, b],
                                   {"r": float(r), "rows": frame.height}))
    return out


def _concentration(df, target):
    if not is_num(df.schema[target]) or df[target].min() is None or df[target].min() < 0:
        return []
    total = df[target].sum()
    if not total:
        return []
    out = []
    for c, t in df.schema.items():
        if c == target or not (t in (pl.String, pl.Categorical)):
            continue
        k = df[c].n_unique()
        if k < 5 or k > 0.5 * df.height:
            continue
        known = df.filter(pl.col(c).is_not_null())
        total = known[target].sum()
        if not total:
            continue
        sums = known.group_by(c).agg(pl.col(target).sum().alias("s")).sort("s", descending=True)
        top = max(1, round(0.2 * sums.height))
        share = float(sums["s"].head(top).sum() / total)
        if share >= 0.6:
            names = ", ".join(str(v) for v in sums[c].head(min(top, 3)).to_list())
            out.append(Insight("concentration", f"{top} of {sums.height} {c} values ({top / sums.height:.0%}) account for "
                                                f"{share:.0%} of total {target} (led by {names}).",
                               round(min(1.0, (share - 0.2) / 0.8), 4), [c, target],
                               {"top_n": top, "groups": sums.height, "share": share,
                                "leaders": sums[c].head(top).to_list()[:10]}))
    return out


def _outliers(df, target):
    if not is_num(df.schema[target]) or df[target].n_unique() <= 10:
        return []
    x = df[target].cast(pl.Float64).drop_nulls().to_numpy()
    if len(x) < 30:
        return []
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return []
    hi, lo = q3 + 3 * iqr, q1 - 3 * iqr
    n_hi, n_lo = int((x > hi).sum()), int((x < lo).sum())
    out = []
    if n_hi:
        out.append(Insight("outliers", f"{n_hi} row(s) ({n_hi / len(x):.1%}) have unusually high {target} "
                                       f"(above {_num(hi)}; the typical range is {_num(q1)}-{_num(q3)}).",
                           round(min(0.5, n_hi / len(x) * 5), 4), [target], {"count": n_hi, "threshold": hi}))
    if n_lo:
        out.append(Insight("outliers", f"{n_lo} row(s) ({n_lo / len(x):.1%}) have unusually low {target} "
                                       f"(below {_num(lo)}).",
                           round(min(0.5, n_lo / len(x) * 5), 4), [target], {"count": n_lo, "threshold": lo}))
    return out


def find_insights(df, target: str | None = None, *, time_col="auto", model=None, max_insights: int = 12,
                  bins: int = 4, min_support: float = 0.05, alpha: float = 0.01, per_column: int = 2,
                  quality: bool = True) -> list:
    """The most notable findings in ``df``, strongest first.

    With a ``target`` (the metric you care about: sales, churn, price …): which segments push it
    up or down ("sales is 18% lower when stock < 20"), its trend over time, correlations,
    concentration ("20% of regions produce 80% of revenue") and outliers. Segment findings
    must pass a Benjamini–Hochberg-corrected significance test (``alpha``) and cover at least
    ``min_support`` of rows. A fitted ``model`` adds its strongest drivers (SHAP).
    """
    df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    if target is not None and target not in df.columns:
        raise ValueError(f"Target '{target}' is not in the data")
    found = []
    if time_col == "auto":
        time_col = next((c for c, t in df.schema.items() if is_temporal(t) and df[c].n_unique() > 2), None)
    if target is not None:
        kind = _target_kind(df[target])
        if kind != "none":
            found += _drivers(df.drop([time_col] if time_col else []), target, kind, bins, min_support, alpha)
        if kind == "numeric":
            if time_col:
                found += _time_insights(df, target, time_col)
            found += _concentration(df, target)
            found += _outliers(df, target)
    found += _correlations(df, target)
    if model is not None:
        from ..evaluation import explain
        try:
            exp = explain(model, df, rows=min(100, df.height))
            imp = exp.global_importance().head(3)
            for i, r in enumerate(imp.iter_rows(named=True)):
                found.append(Insight("model", f"{r['feature']} is {'the strongest' if i == 0 else 'a leading'} driver "
                                              f"of the model's predicted {getattr(model, 'target', 'outcome')} "
                                              f"({r['share']:.0%} of its attribution).",
                                     round(0.6 * r["share"] + 0.2, 4), [r["feature"]],
                                     {"share": r["share"], "mean_abs_shap": r["mean_abs_shap"]}))
        except Exception:
            pass
    if quality:
        from ..quality import check_quality
        q = check_quality(df)
        if q.score < 85 and q.issues:
            found.append(Insight("quality", f"Data quality is {q.grade} ({q.score:.0f}/100); the biggest issue: "
                                            f"{q.issues[0].title}. Treat findings involving it with care.",
                                 0.3, [], {"score": q.score, "grade": q.grade, "issues": len(q.issues)}))
    found.sort(key=lambda i: -i.strength)
    kept, per = [], {}
    for ins in found:                     # avoid ten findings about the same column
        key = ins.columns[0] if ins.kind == "driver" else None
        if key is not None:
            if per.get(key, 0) >= per_column:
                continue
            per[key] = per.get(key, 0) + 1
        kept.append(ins)
    return kept[:max_insights]
