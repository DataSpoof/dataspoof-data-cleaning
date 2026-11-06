"""Shared detection helpers used by the cleaning and quality modules.

One place for the regexes, column-name heuristics and placeholder lists, so the quality
checks, the profiler, data contracts and the cleaning steps always agree on what counts
as a number stored as text, a date, an email, a placeholder, etc.
"""

from __future__ import annotations

import datetime as _dt
import re

import polars as pl

from .steps import DEFAULT_MISSING_SENTINELS

# Text values that mean "missing" ("NA", "?", "-", "unknown", "" ...), lowercased.
SENTINELS = sorted({s.lower() for s in DEFAULT_MISSING_SENTINELS})
# Numbers commonly used as missing-value codes.
NUMERIC_SENTINELS = [-9999, -999, -99, -1, 999, 9999, 99999]

PHONE_TOKENS = {"phone", "mobile", "tel", "telephone", "cell", "cellphone", "fax"}
CODE_TOKENS = PHONE_TOKENS | {"zip", "zipcode", "postal", "postcode", "pin", "pincode",
                              "code", "id", "sku", "isbn", "account", "acct"}
NONNEG_TOKENS = {"price", "cost", "amount", "salary", "income", "quantity", "qty", "count",
                 "sales", "revenue", "fare", "weight", "height", "distance", "duration",
                 "population", "area", "volume", "length", "width"}

# "1,200", "$50", "-7.5", "(30)", "12%", ".5"
NUM_RE = (r"^\(?\s*[-+]?\s*[$₹€£¥]?\s*[-+]?\s*"
          r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)\s*%?\s*\)?$")

_TIME = r"(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s*[AaPp][Mm])?(?:Z|[+-]\d{2}:?\d{2})?)?$"
DATE_FORMATS = [
    ("YYYY-MM-DD", r"^\d{4}-\d{1,2}-\d{1,2}" + _TIME),
    ("YYYY/MM/DD", r"^\d{4}/\d{1,2}/\d{1,2}" + _TIME),
    ("DD/MM/YYYY or MM/DD/YYYY", r"^\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})" + _TIME),
    ("DD-MM-YYYY or MM-DD-YYYY", r"^\d{1,2}-\d{1,2}-(?:\d{4}|\d{2})" + _TIME),
    ("DD.MM.YYYY", r"^\d{1,2}\.\d{1,2}\.(?:\d{4}|\d{2})" + _TIME),
    ("Month name (e.g. Jan 5, 2021)",
     r"(?i)^(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
     r"[a-z]*\.?\s+(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?\d{4}" + _TIME),
]
DATE_RE = "|".join(f"(?:{pattern})" for _, pattern in DATE_FORMATS)
ISO_DATE_RX = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})")

EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$"
URL_RE = r"(?i)^(?:https?|ftp)://[^\s/$.?#][^\s]*$"
UUID_RE = r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
IPV4_RE = r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$"
PHONE_CHARS_RE = r"(?i)^[+\d\s\-().]+(?:\s*(?:x|ext\.?)\s*\d+)?$"


def tokens(name) -> set:
    """Lowercase word tokens of a column name: 'CompPrice' -> {'comp', 'price'}."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}


def is_text(dtype) -> bool:
    return dtype == pl.String


def is_num(dtype) -> bool:
    return dtype.is_numeric()


def is_temporal(dtype) -> bool:
    return dtype == pl.Date or dtype == pl.Datetime


def floats(series: pl.Series) -> list:
    """Values as floats with NaN for nulls (the Rust backend ignores NaN)."""
    return [float(v) if v is not None else float("nan") for v in series.to_list()]


def real_text(series: pl.Series) -> pl.Series:
    """Non-null, trimmed text values that aren't missing-value placeholders."""
    s = series.drop_nulls().cast(pl.String, strict=False).str.strip_chars()
    return s.filter(~s.str.to_lowercase().is_in(SENTINELS))


def phone_ok(values: pl.Series) -> pl.Series:
    """Boolean mask: 7–15 digits using only digits, spaces and + - ( ) ."""
    text = values.cast(pl.String, strict=False).str.replace(r"\.0$", "")
    digits = text.str.replace_all(r"\D", "").str.len_chars()
    return text.str.contains(PHONE_CHARS_RE) & (digits >= 7) & (digits <= 15)


def range_rule(name, this_year=None):
    """Valid (low, high, description) range implied by a column name, or None."""
    t = tokens(name)
    year = this_year or _dt.date.today().year
    if "age" in t:
        return 0, 120, "an age"
    if t & {"pct", "percent", "percentage"}:
        return 0, 100, "a percentage"
    if t & {"prob", "probability"}:
        return 0, 1, "a probability"
    if t & {"year", "yr"}:
        return 1900, year + 1, "a year"
    if "month" in t:
        return 1, 12, "a month number"
    if "hour" in t:
        return 0, 23, "an hour"
    if t & NONNEG_TOKENS:
        return 0, None, "a non-negative quantity"
    return None


def plural(k, word, plural_form=None) -> str:
    return f"{k:,} {word if k == 1 else (plural_form or word + 's')}"


def fmt(vals) -> str:
    return ", ".join(f"{v:g}" if isinstance(v, float) else
                     f"'{v}'" if isinstance(v, str) else str(v) for v in vals)
