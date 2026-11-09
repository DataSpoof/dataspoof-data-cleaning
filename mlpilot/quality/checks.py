"""Data-quality checks and a 0–100 quality score.

``check_quality(df)`` runs every check and returns a ``QualityReport``: the list of issues
(each with a severity, the affected column and how many cells it touches) plus scores.

Scoring
-------
Four standard data-quality dimensions, each 0–100:

* **completeness** – share of cells that hold a real value (nulls, text placeholders such as
  "NA"/"?" and numeric codes such as -999 all count as missing).
* **uniqueness**   – share of rows that are not exact repeats of an earlier row.
* **validity**     – share of rule-checked cells that pass: values inside name-implied ranges
  (age, percent, price ...), valid emails / phone numbers / calendar dates, and numbers stored
  as text without non-numeric blockers. 100 if no rule applied.
* **consistency**  – share of text cells free of formatting drift: stray whitespace, the same
  value spelled with different letter case, and minority date formats in mixed-format
  columns. 100 if there is no text.

The overall score is the weighted mean (default weights: completeness 0.35, validity 0.30,
consistency 0.20, uniqueness 0.15) and maps to a grade: A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60,
else F. Each column also gets a score with the same weights over the dimensions that apply
to it. Because the overall score averages over every cell, one broken column in a wide
table barely moves it, so check ``weakest_columns()`` as well. Statistical observations (outliers, skew, negatives, rare categories, constant and
ID-like columns) are reported as issues but don't lower the score, because they aren't
necessarily wrong.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from .. import _rustcore
from .._detect import (CODE_TOKENS, DATE_FORMATS, EMAIL_RE, ISO_DATE_RX, NUM_RE, NUMERIC_SENTINELS,
                       PHONE_TOKENS, SENTINELS, fmt, is_num, is_text, phone_ok, plural, range_rule,
                       real_text, tokens)

SEVERITIES = ("high", "medium", "low")
DIMENSIONS = ("completeness", "uniqueness", "validity", "consistency")
DEFAULT_WEIGHTS = {"completeness": 0.35, "validity": 0.30, "consistency": 0.20, "uniqueness": 0.15}
_SEV_RANK = {"high": 0, "medium": 1, "low": 2}


def jsonable(obj):
    """Convert to plain JSON types (NaN/inf -> None, dates -> ISO strings)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(dataclasses.asdict(obj))
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (_dt.date, _dt.datetime, _dt.time, _dt.timedelta)):
        return obj.isoformat() if hasattr(obj, "isoformat") else str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def grade(score: float) -> str:
    for cutoff, letter in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if score >= cutoff:
            return letter
    return "F"


@dataclasses.dataclass
class Issue:
    """One detected data-quality problem."""

    check: str                  # machine-readable id, e.g. "missing_values"
    severity: str               # high | medium | low
    title: str                  # e.g. "Missing values · Cabin"
    metric: str                 # short badge, e.g. "77.1%"
    detail: str                 # one-sentence explanation with a suggested fix
    column: str | None = None
    affected: int = 0           # cells (or rows) involved
    dimension: str | None = None  # scored dimension, or None if informational

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class QualityReport:
    """Result of :func:`check_quality`. Print it for a readable summary."""

    rows: int
    columns: int
    score: float
    grade: str
    dimensions: dict
    column_scores: dict
    issues: list
    weights: dict

    @property
    def counts(self) -> dict:
        return {s: sum(1 for i in self.issues if i.severity == s) for s in SEVERITIES}

    def weakest_columns(self, k: int = 3, below: float = 90.0) -> list:
        """Up to ``k`` ``(column, score)`` pairs scoring under ``below``, worst first.

        The overall score averages over every cell, so one broken column in a wide table
        barely moves it; check these too."""
        ranked = sorted(self.column_scores.items(), key=lambda kv: kv[1])
        return [(c, s) for c, s in ranked if s < below][:k]

    def filter(self, severity=None, check=None, column=None) -> list:
        return [i for i in self.issues
                if (severity is None or i.severity == severity)
                and (check is None or i.check == check)
                and (column is None or i.column == column)]

    def to_dict(self) -> dict:
        return jsonable({**dataclasses.asdict(self), "counts": self.counts})

    def to_json(self, path=None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_frame(self) -> pl.DataFrame:
        """Issues as a DataFrame (one row per issue)."""
        schema = {"severity": pl.String, "check": pl.String, "column": pl.String,
                  "title": pl.String, "metric": pl.String, "affected": pl.Int64,
                  "dimension": pl.String, "detail": pl.String}
        rows = [{k: getattr(i, k) for k in schema} for i in self.issues]
        return pl.DataFrame(rows, schema=schema)

    def summary(self, top: int = 10) -> str:
        c = self.counts
        lines = [f"Data quality score: {self.score:.1f}/100 (grade {self.grade}) | "
                 f"{self.rows:,} rows x {self.columns} columns | {len(self.issues)} issues "
                 f"({c['high']} high, {c['medium']} medium, {c['low']} low)"]
        for d in DIMENSIONS:
            v = self.dimensions[d]
            bar = "#" * int(round(v / 5)) + "." * (20 - int(round(v / 5)))
            lines.append(f"  {d:<13} {v:5.1f}  {bar}")
        weakest = self.weakest_columns()
        if weakest:
            lines.append("Weakest columns: " + ", ".join(f"{c} {s:.1f}" for c, s in weakest))
        if self.issues:
            lines.append("Top issues:")
            for i in self.issues[:top]:
                lines.append(f"  [{i.severity:<6}] {i.title}  ({i.metric})")
            if len(self.issues) > top:
                lines.append(f"  ... and {len(self.issues) - top} more")
        return "\n".join(lines)

    __str__ = summary

    def __repr__(self) -> str:
        return self.summary()


def _case_variant_cells(nn: pl.Series) -> int:
    """Cells whose value differs only by letter case from a more common spelling."""
    pairs = pl.DataFrame({"v": nn, "l": nn.str.to_lowercase()}).group_by(["l", "v"]).len()
    per = pairs.group_by("l").agg(pl.col("len").sum().alias("total"),
                                  pl.col("len").max().alias("top"),
                                  pl.len().alias("variants"))
    value = per.filter(pl.col("variants") > 1).select((pl.col("total") - pl.col("top")).sum()).item()
    return int(value or 0)


def check_quality(df, *, weights: dict | None = None) -> QualityReport:
    """Run every data-quality check on ``df`` and score it. See the module docstring."""
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    weights = dict(weights or DEFAULT_WEIGHTS)
    if set(weights) - set(DIMENSIONS) or any(v < 0 for v in weights.values()) or not sum(weights.values()):
        raise ValueError(f"weights must be non-negative values for {DIMENSIONS}")

    n = df.height or 1
    num_cols = [c for c, t in df.schema.items() if is_num(t)]
    text_cols = [c for c, t in df.schema.items() if is_text(t)]
    try:
        dup_rows = int(df.is_duplicated().sum())
        dup_extra = df.height - df.unique().height
    except Exception:          # unhashable column types
        dup_rows = dup_extra = 0

    # Scoring accumulators (cell counts per column).
    missing = {c: df[c].null_count() for c in df.columns}
    invalid, checked = {}, {}
    inconsistent = {}
    text_cells = {c: df[c].drop_nulls().len() for c in text_cols}

    def bump(acc, col, k):
        acc[col] = acc.get(col, 0) + k

    issues: list[Issue] = []
    text_flagged = set()   # text columns that really hold numbers or dates
    ranged = set()         # numeric columns covered by a range rule
    placeholders = {}      # numeric column -> placeholder codes found

    def num_series(c):
        """Non-null values of a numeric column, minus detected placeholder codes."""
        s = df[c].drop_nulls()
        if placeholders.get(c):
            codes = pl.Series(placeholders[c]).cast(s.dtype).to_list()
            s = s.filter(~s.is_in(codes))
        return s

    # ── completeness ────────────────────────────────────────────────────────
    for c in df.columns:
        nc = missing[c]
        if nc > 0:
            pct = nc / n
            sev = "high" if pct > 0.3 else ("medium" if pct > 0.05 else "low")
            issues.append(Issue("missing_values", sev, f"Missing values · {c}", f"{pct:.1%}",
                                f"{nc:,} of {n:,} rows are null in “{c}”.", c, nc, "completeness"))

    # Disguised missing values in text columns ("NA", "?", "-", "unknown", "" ...)
    for c in text_cols:
        nn = df[c].drop_nulls()
        if nn.len() == 0:
            continue
        mask = nn.str.strip_chars().str.to_lowercase().is_in(SENTINELS)
        k = int(mask.sum())
        if k:
            bump(missing, c, k)
            vc = nn.filter(mask).value_counts(sort=True)
            toks = ", ".join(f"'{tok}' ×{cnt:,}" for tok, cnt in
                             list(zip(vc[vc.columns[0]].to_list(), vc["count"].to_list()))[:4])
            issues.append(Issue(
                "disguised_missing", "high" if k / n > 0.05 else "medium",
                f"Disguised missing values · {c}", f"{k:,}",
                f"“{c}” has {plural(k, 'missing-data placeholder')} ({toks}), which "
                f"{'is' if k == 1 else 'are'} not counted as null. Convert with "
                f"ac.replace_disguised_missing(df).", c, k, "completeness"))

    # Placeholder numbers in numeric columns (-999, 999, -1 ...)
    for c in num_cols:
        s = df[c].drop_nulls()
        if s.len() < 10 or s.n_unique() <= 3:
            continue
        found = []
        for v in NUMERIC_SENTINELS:
            cnt = int((s == v).sum())
            if cnt < max(2, 0.01 * s.len()):
                continue
            others = s.filter(s != v)
            if others.len() == 0:
                continue
            if v < 0:
                if others.min() < 0:
                    continue
            else:
                mx = others.max()
                if mx >= v or (mx > 0 and v < 1.5 * mx):
                    continue
            found.append((v, cnt))
        if found:
            placeholders[c] = [v for v, _ in found]
            rest = num_series(c)
            total = sum(cnt for _, cnt in found)
            bump(missing, c, total)
            toks = ", ".join(f"{v} ×{cnt:,}" for v, cnt in found)
            issues.append(Issue(
                "placeholder_numbers", "medium", f"Placeholder numbers · {c}", f"{total:,}",
                f"“{c}” contains likely missing-value codes ({toks}) that sit outside "
                f"the rest of the data ({fmt([rest.min()])} to {fmt([rest.max()])}). "
                f"Convert them with ac.replace_disguised_missing(df, numeric_sentinels=[...]).",
                c, total, "completeness"))

    # ── types stored as text ────────────────────────────────────────────────
    # Numbers stored as text (e.g. "1,200", "$50", or digits blocked by a "-")
    for c in text_cols:
        if tokens(c) & CODE_TOKENS:
            continue
        nn = df[c].drop_nulls().str.strip_chars()
        if nn.len() < 5:
            continue
        sent_mask = nn.str.to_lowercase().is_in(SENTINELS)
        real = nn.filter(~sent_mask)
        if real.len() < 5:
            continue
        is_number = real.str.contains(NUM_RE)
        k = int(is_number.sum())
        if k / real.len() < 0.9:
            continue
        nums = real.filter(is_number)
        if nums.str.contains(r"^[-+]?0\d").mean() > 0.1:   # leading zeros -> codes
            continue
        text_flagged.add(c)
        blockers_n = real.len() - k
        bump(invalid, c, blockers_n)
        bump(checked, c, real.len())
        blockers = (real.filter(~is_number).unique().head(3).to_list()
                    + nn.filter(sent_mask).unique().head(3).to_list())
        n_fmt = int(nums.str.contains(r"[,$₹€£¥%()]").sum())
        reasons = []
        if blockers:
            reasons.append("non-numeric values like " + fmt(blockers[:4]))
        if n_fmt:
            reasons.append(f"{plural(n_fmt, 'value')} carrying symbols or commas")
        issues.append(Issue(
            "numbers_as_text", "medium", f"Numbers stored as text · {c}", f"{k / real.len():.0%} numeric",
            f"{k:,} of {real.len():,} non-missing values in “{c}” are numbers, but the "
            f"column is typed as text" + (" because of " + " and ".join(reasons) if reasons else "")
            + ". Clean those values and cast the column to a number.", c, blockers_n, "validity"))

    # Dates stored as text, mixed date formats, impossible calendar dates
    for c in text_cols:
        if c in text_flagged:
            continue
        nn = real_text(df[c])
        if nn.len() < 5:
            continue
        any_mask, used = None, []
        for label, pat in DATE_FORMATS:
            m = nn.str.contains(pat)
            cnt = int(m.sum())
            if cnt:
                used.append((label, cnt))
            any_mask = m if any_mask is None else (any_mask | m)
        matched = int(any_mask.sum())
        if matched / nn.len() < 0.6:
            continue
        text_flagged.add(c)
        impossible = 0
        for v in nn.to_list()[:20000]:
            mm = ISO_DATE_RX.match(v)
            if mm:
                try:
                    _dt.date(int(mm.group(1)), int(mm.group(2)), int(mm.group(3)))
                except ValueError:
                    impossible += 1
        unmatched = nn.len() - matched
        used.sort(key=lambda x: -x[1])
        bump(invalid, c, impossible + unmatched)
        bump(checked, c, nn.len())
        parts = [f"{lbl} ({cnt:,})" for lbl, cnt in used]
        extra = []
        if unmatched:
            extra.append(f"{plural(unmatched, 'value')} not in a recognised date format")
        if impossible:
            extra.append(f"{plural(impossible, 'impossible calendar date')} (e.g. month 13)")
        tail = (" " + "; ".join(extra) + ".") if extra else ""
        if len(used) > 1:
            minority = matched - used[0][1]
            bump(inconsistent, c, minority)
            issues.append(Issue(
                "mixed_date_formats", "medium", f"Mixed date formats · {c}", f"{len(used)} formats",
                f"“{c}” mixes date formats: {', '.join(parts)}.{tail} Standardise it "
                f"and parse it with ac.parse_dates(df, ['{c}']).", c, minority, "consistency"))
        else:
            issues.append(Issue(
                "dates_as_text", "medium" if (impossible or unmatched) else "low",
                f"Dates stored as text · {c}", used[0][0],
                f"“{c}” holds dates ({parts[0]}) but is typed as text.{tail} "
                f"Parse it with ac.parse_dates(df, ['{c}']).", c, impossible + unmatched, "validity"))

    # ── structure (informational) ───────────────────────────────────────────
    if dup_rows > 0:
        sev = "high" if dup_rows / n > 0.05 else "medium"
        issues.append(Issue("duplicate_rows", sev, "Duplicate rows", f"{dup_rows:,}",
                            f"{dup_rows:,} fully duplicated rows — consider dropping them.",
                            None, dup_extra, "uniqueness"))
    for c in df.columns:
        if df[c].n_unique() <= 1:
            issues.append(Issue("constant_column", "medium", f"Constant column · {c}", "1 value",
                                f"“{c}” has a single unique value and carries no information.", c))
    # Near-unique integer/text columns. Continuous floats are naturally unique, and text
    # columns that really hold numbers/dates are reported above.
    for c in df.columns:
        if df[c].dtype in (pl.Float32, pl.Float64) or c in text_flagged:
            continue
        ratio = df[c].n_unique() / n
        if ratio > 0.95 and df.height > 10:
            issues.append(Issue(
                "possible_identifier", "low", f"Possible identifier · {c}", f"{ratio:.0%} unique",
                f"“{c}” is nearly unique per row — likely an ID with no predictive value.", c))

    # ── numeric checks (placeholder codes excluded so each problem is reported once) ──
    for c in num_cols:
        vals = num_series(c).to_list()
        if len(vals) >= 8:
            lo, hi = _rustcore.iqr_bounds([float(v) for v in vals])
            if hi <= lo:     # zero-width fences (most values identical): IQR outliers are meaningless
                continue
            out = sum(1 for v in vals if v < lo or v > hi)
            if out > 0:
                pct = out / len(vals)
                sev = "high" if pct > 0.1 else ("medium" if pct > 0.02 else "low")
                issues.append(Issue("outliers", sev, f"Outliers · {c}", f"{out:,}",
                                    f"{plural(out, 'value')} outside the IQR fences [{lo:.2f}, {hi:.2f}].",
                                    c, out))

    # Values outside a valid range, inferred from the column name
    for c in num_cols:
        rule = range_rule(c)
        if not rule:
            continue
        ranged.add(c)
        lo, hi, kind = rule
        s = num_series(c)
        if s.len() == 0:
            continue
        bad_mask = (s < lo) if hi is None else ((s < lo) | (s > hi))
        k = int(bad_mask.sum())
        bump(checked, c, s.len())
        if k:
            bump(invalid, c, k)
            rng = f"≥ {lo}" if hi is None else f"[{lo}, {hi}]"
            ex = s.filter(bad_mask).unique().sort().head(3).to_list()
            issues.append(Issue(
                "out_of_range", "high" if k / s.len() > 0.05 else "medium",
                f"Out-of-range values · {c}", f"{k:,}",
                f"“{c}” has {plural(k, 'value')} outside the valid range {rng} for "
                f"{kind} (e.g. {fmt(ex)}).", c, k, "validity"))

    for c in num_cols:
        if c in ranged:
            continue   # already covered by the range rule above
        if _rustcore.has_negative([float(v) for v in num_series(c).to_list()]):
            issues.append(Issue("negative_values", "low", f"Negative values · {c}", "present",
                                f"“{c}” contains negative values — verify whether that is expected.", c))
    for c in num_cols:
        sk = _rustcore.skewness([float(v) for v in num_series(c).to_list()])
        if sk == sk and abs(sk) > 1:
            issues.append(Issue("skewed", "low", f"Skewed distribution · {c}", f"skew {sk:.1f}",
                                f"“{c}” is highly skewed — a log transform may help.", c))

    # ── text consistency ────────────────────────────────────────────────────
    for c in text_cols:
        raw = df[c].to_list()
        cleaned = _rustcore.normalize_whitespace([str(v) if v is not None else None for v in raw])
        bad = sum(1 for a, b in zip(raw, cleaned) if a is not None and str(a) != b)
        if bad > 0:
            bump(inconsistent, c, bad)
            issues.append(Issue(
                "whitespace", "low", f"Whitespace issues · {c}", f"{bad:,}",
                f"“{c}” has {plural(bad, 'value')} with leading/trailing or repeated whitespace.",
                c, bad, "consistency"))
    for c in text_cols:
        nn = df[c].drop_nulls()
        u = nn.n_unique()
        lu = nn.str.to_lowercase().n_unique()
        if u > lu:
            cells = _case_variant_cells(nn)
            bump(inconsistent, c, cells)
            issues.append(Issue(
                "inconsistent_casing", "medium", f"Inconsistent casing · {c}", f"{u - lu} merge(s)",
                f"{u} distinct values in “{c}” collapse to {lu} when lowercased "
                f"(values that differ only in letter case).", c, cells, "consistency"))

    # Rare categories (each under 1% of rows) — often typos, but not necessarily wrong.
    for c in text_cols:
        if c in text_flagged:
            continue
        nn = real_text(df[c])
        k = nn.n_unique()
        if nn.len() < 100 or k < 3 or k > 50:
            continue
        vc = nn.value_counts()
        rare = vc.filter(pl.col("count") < 0.01 * nn.len())
        if rare.height:
            ex = rare.sort("count")[vc.columns[0]].head(3).to_list()
            rows = int(rare["count"].sum())
            issues.append(Issue(
                "rare_categories", "low", f"Rare categories · {c}", f"{rare.height} rare",
                f"“{c}” has {plural(rare.height, 'category', 'categories')} (of {k}) that "
                f"each cover under 1% of rows — {plural(rows, 'row')} in total, e.g. "
                f"{fmt(ex)}. They may be typos; otherwise consider grouping them "
                f"into 'Other' with ac.group_rare_categories(df).", c, rows))

    # ── format validity ─────────────────────────────────────────────────────
    # Invalid email addresses (by column name, or when most values contain "@")
    for c in text_cols:
        nn = real_text(df[c])
        if nn.len() == 0:
            continue
        if not ({"email", "mail"} & tokens(c) or nn.str.contains("@", literal=True).mean() >= 0.5):
            continue
        bad = nn.filter(~nn.str.contains(EMAIL_RE))
        bump(checked, c, nn.len())
        if bad.len():
            bump(invalid, c, bad.len())
            issues.append(Issue(
                "invalid_email", "medium" if bad.len() / nn.len() > 0.05 else "low",
                f"Invalid email addresses · {c}", f"{bad.len():,}",
                f"“{c}” has {plural(bad.len(), 'invalid email address', 'invalid email addresses')} "
                f"out of {nn.len():,} (e.g. {fmt(bad.unique().head(3).to_list())}).",
                c, bad.len(), "validity"))

    # Invalid phone numbers (by column name: phone, mobile, tel, cell, fax)
    for c in df.columns:
        if not (tokens(c) & PHONE_TOKENS):
            continue
        nn = real_text(df[c])
        if nn.len() == 0:
            continue
        bad = nn.filter(~phone_ok(nn))
        bump(checked, c, nn.len())
        if bad.len():
            bump(invalid, c, bad.len())
            issues.append(Issue(
                "invalid_phone", "medium" if bad.len() / nn.len() > 0.05 else "low",
                f"Invalid phone numbers · {c}", f"{bad.len():,}",
                f"“{c}” has {plural(bad.len(), 'invalid phone number')} out of "
                f"{nn.len():,} (a valid one has 7–15 digits and only digits, spaces, "
                f"+ - ( ) .), e.g. {fmt(bad.unique().head(3).to_list())}.",
                c, bad.len(), "validity"))

    # ── scoring ─────────────────────────────────────────────────────────────
    def ok_pct(bad, total):
        return 100.0 if total <= 0 else max(0.0, 100.0 * (1.0 - bad / total))

    dims = {
        "completeness": ok_pct(sum(missing.values()), df.height * df.width),
        "uniqueness": ok_pct(dup_extra, df.height),
        "validity": ok_pct(sum(invalid.values()), sum(checked.values())),
        "consistency": ok_pct(sum(inconsistent.values()), sum(text_cells.values())),
    }
    total_w = sum(weights.values())
    score = sum(weights.get(d, 0) * dims[d] for d in DIMENSIONS) / total_w
    column_scores = {}
    for c in df.columns:
        parts = {"completeness": ok_pct(missing[c], df.height)}
        if checked.get(c):
            parts["validity"] = ok_pct(invalid.get(c, 0), checked[c])
        if text_cells.get(c):
            parts["consistency"] = ok_pct(inconsistent.get(c, 0), text_cells[c])
        # Same weights as the overall score, renormalised over the dimensions that apply.
        w = sum(weights.get(d, 0) for d in parts)
        value = (sum(weights.get(d, 0) * v for d, v in parts.items()) / w if w
                 else sum(parts.values()) / len(parts))
        column_scores[c] = round(value, 1)

    issues.sort(key=lambda i: _SEV_RANK[i.severity])
    return QualityReport(rows=df.height, columns=df.width, score=round(score, 1),
                         grade=grade(score), dimensions={d: round(v, 1) for d, v in dims.items()},
                         column_scores=column_scores, issues=issues, weights=weights)
