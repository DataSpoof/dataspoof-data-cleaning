"""
Additional data-cleaning steps for AutomatedCleaning.

Every function takes a Polars DataFrame and returns a Polars DataFrame (a few
also return a report dict). Compute-heavy per-cell string work is delegated to
the Rust backend (``automatedcleaning._rustcore``); the rest uses Polars,
scikit-learn and a few optional libraries that are imported lazily so this
module always imports even if they are absent.

Grouped as: column/structure, missing values, string normalization, dtypes,
categorical, outliers, duplicates, scaling, validation, domain, and text.
"""

import re
import math
import unicodedata
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import polars as pl

from . import _rustcore

__all__ = [
    # column & structure
    "standardize_column_names", "drop_empty_rows_and_columns",
    "drop_high_missing_columns", "remove_duplicate_columns", "drop_id_like_columns",
    # missing values
    "replace_disguised_missing", "add_missing_indicators", "impute_missing",
    # string normalization
    "normalize_whitespace", "normalize_unicode", "fix_mojibake", "standardize_casing",
    "standardize_categories",
    # dtypes & parsing
    "parse_dates", "extract_date_parts", "normalize_booleans", "downcast_numeric",
    # categorical
    "group_rare_categories", "fuzzy_cluster_categories", "encode_categorical",
    # outliers
    "remove_outliers_zscore", "remove_outliers_mad", "remove_outliers_isolation_forest",
    "winsorize", "remove_outliers_by_group",
    # duplicates
    "drop_duplicates_subset", "fuzzy_deduplicate",
    # scaling
    "scale_numeric", "bin_numeric",
    # validation
    "apply_range_rules", "check_cross_field", "validate_formats",
    "check_referential_integrity",
    # domain
    "normalize_phone_numbers", "normalize_emails", "canonicalize_urls",
    "standardize_country_region", "convert_currency",
    # text
    "detect_language", "filter_by_language", "correct_spelling_text",
    "redact_profanity", "near_duplicate_text",
]

DEFAULT_MISSING_SENTINELS = [
    "na", "n/a", "n.a.", "null", "none", "nan", "nil", "-", "--", "?", "??",
    "unknown", "unk", "missing", "not available", "not applicable", "", " ",
]


def _header(title):
    print("\n" + "-" * 100)
    print(title.center(100))
    print("-" * 100 + "\n")


def _string_cols(df):
    return [c for c, t in df.schema.items() if t == pl.Utf8]


def _numeric_cols(df):
    return [c for c, t in df.schema.items()
            if t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                     pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8)]


def _to_float_list(series):
    return [float(x) if x is not None else float("nan") for x in series.to_list()]


def _apply_rust_str(df, func, columns=None):
    """Apply a Rust batch string function to each (given/all) string column."""
    cols = columns if columns is not None else _string_cols(df)
    for col in cols:
        if col in df.columns:
            values = df[col].cast(pl.Utf8, strict=False).to_list()
            df = df.with_columns(pl.Series(col, func(values)))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Column & structure
# ─────────────────────────────────────────────────────────────────────────────

def _snake_case(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)      # camelCase -> camel_Case
    name = re.sub(r"[^\w\s]", "", name)                        # drop punctuation
    name = re.sub(r"\s+", "_", name)                          # spaces -> _
    name = re.sub(r"_+", "_", name).strip("_").lower()
    return name or "column"


def standardize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """Trim, lowercase, snake_case and de-duplicate column names."""
    _header("Standardizing column names")
    mapping, seen = {}, {}
    for col in df.columns:
        new = _snake_case(col)
        if new in seen:
            seen[new] += 1
            new = f"{new}_{seen[new]}"
        else:
            seen[new] = 0
        mapping[col] = new
    changed = {k: v for k, v in mapping.items() if k != v}
    if changed:
        print(f"Renamed {len(changed)} columns, e.g. {dict(list(changed.items())[:5])}")
    return df.rename(mapping)


def drop_empty_rows_and_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop columns that are entirely null and rows where every value is null."""
    _header("Dropping fully empty rows and columns")
    keep_cols = [c for c in df.columns if df[c].null_count() < df.height]
    dropped_cols = [c for c in df.columns if c not in keep_cols]
    if dropped_cols:
        print(f"Dropping empty columns: {dropped_cols}")
    df = df.select(keep_cols)
    before = df.height
    if df.width:
        df = df.filter(~pl.all_horizontal(pl.all().is_null()))
    print(f"Dropped {before - df.height} fully empty rows")
    return df


def drop_high_missing_columns(df: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
    """Drop columns whose missing fraction exceeds ``threshold`` (0-1)."""
    _header(f"Dropping columns with > {threshold:.0%} missing")
    n = df.height or 1
    drop = [c for c in df.columns if df[c].null_count() / n > threshold]
    if drop:
        print(f"Dropping high-missing columns: {drop}")
    return df.drop(drop) if drop else df


def remove_duplicate_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Remove columns that hold identical values under different names."""
    _header("Removing duplicate columns")
    seen, drop = {}, []
    for col in df.columns:
        key = (str(df[col].dtype), tuple(df[col].to_list()))
        if key in seen:
            drop.append(col)
        else:
            seen[key] = col
    if drop:
        print(f"Dropping duplicate columns: {drop}")
    return df.drop(drop) if drop else df


def drop_id_like_columns(df: pl.DataFrame, uniqueness: float = 0.99) -> pl.DataFrame:
    """Drop near-unique identifier columns (unique ratio >= ``uniqueness``)."""
    _header("Dropping ID-like columns")
    n = df.height or 1
    drop = []
    for col in df.columns:
        ratio = df[col].n_unique() / n
        looks_id = ratio >= uniqueness or re.fullmatch(r"(?i)(id|uuid|guid|index|_id)", col.replace(" ", "_"))
        if ratio >= uniqueness or (looks_id and ratio > 0.9):
            drop.append(col)
    if drop:
        print(f"Dropping ID-like columns: {drop}")
    return df.drop(drop) if drop else df


# ─────────────────────────────────────────────────────────────────────────────
# Missing values
# ─────────────────────────────────────────────────────────────────────────────

def replace_disguised_missing(df: pl.DataFrame, extra_sentinels=None,
                              numeric_sentinels=None) -> pl.DataFrame:
    """Convert disguised-missing tokens (``"NA"``, ``"?"``, ``999`` ...) to null.

    String columns are handled in the Rust backend; ``numeric_sentinels`` (e.g.
    ``[999, -1]``) are nulled out in numeric columns when provided.
    """
    _header("Recognizing disguised missing values")
    sentinels = list(DEFAULT_MISSING_SENTINELS)
    if extra_sentinels:
        sentinels += [str(s).lower() for s in extra_sentinels]

    for col in _string_cols(df):
        values = df[col].to_list()
        df = df.with_columns(pl.Series(col, _rustcore.replace_disguised_missing(values, sentinels)))

    if numeric_sentinels:
        for col in _numeric_cols(df):
            df = df.with_columns(
                pl.when(pl.col(col).is_in(list(numeric_sentinels)))
                .then(None).otherwise(pl.col(col)).alias(col)
            )
    print(f"Applied {len(sentinels)} sentinel patterns")
    return df


def add_missing_indicators(df: pl.DataFrame, suffix: str = "_missing") -> pl.DataFrame:
    """Add a boolean ``<col><suffix>`` flag for every column that has nulls."""
    _header("Adding missing-value indicator columns")
    added = []
    for col in df.columns:
        if df[col].null_count() > 0:
            df = df.with_columns(pl.col(col).is_null().alias(f"{col}{suffix}"))
            added.append(f"{col}{suffix}")
    print(f"Added indicators: {added}" if added else "No columns needed indicators")
    return df


def impute_missing(df: pl.DataFrame, numeric_strategy: str = "median",
                   categorical_strategy: str = "mode", constant_value=0,
                   knn_neighbors: int = 5, columns=None) -> pl.DataFrame:
    """Impute missing values with a selectable strategy.

    numeric_strategy: ``mean|median|knn|mice|interpolate|ffill|bfill|constant``
    categorical_strategy: ``mode|ffill|bfill|constant``
    """
    _header(f"Imputing missing values (num={numeric_strategy}, cat={categorical_strategy})")
    num_cols = [c for c in _numeric_cols(df) if columns is None or c in columns]
    cat_cols = [c for c in _string_cols(df) if columns is None or c in columns]

    if num_cols and numeric_strategy in ("knn", "mice"):
        arr = df.select(num_cols).to_numpy()
        if numeric_strategy == "knn":
            from sklearn.impute import KNNImputer
            imp = KNNImputer(n_neighbors=knn_neighbors)
        else:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            imp = IterativeImputer(random_state=42)
        filled = imp.fit_transform(arr)
        df = df.with_columns([pl.Series(num_cols[i], filled[:, i]) for i in range(len(num_cols))])
    else:
        for col in num_cols:
            if df[col].null_count() == 0:
                continue
            if numeric_strategy == "mean":
                df = df.with_columns(pl.col(col).fill_null(df[col].mean()))
            elif numeric_strategy == "median":
                df = df.with_columns(pl.col(col).fill_null(df[col].median()))
            elif numeric_strategy == "interpolate":
                df = df.with_columns(pl.col(col).interpolate())
            elif numeric_strategy == "ffill":
                df = df.with_columns(pl.col(col).forward_fill())
            elif numeric_strategy == "bfill":
                df = df.with_columns(pl.col(col).backward_fill())
            elif numeric_strategy == "constant":
                df = df.with_columns(pl.col(col).fill_null(constant_value))

    for col in cat_cols:
        if df[col].null_count() == 0:
            continue
        if categorical_strategy == "mode":
            modes = df[col].drop_nulls().mode().to_list()
            if modes:
                df = df.with_columns(pl.col(col).fill_null(modes[0]))
        elif categorical_strategy == "ffill":
            df = df.with_columns(pl.col(col).forward_fill())
        elif categorical_strategy == "bfill":
            df = df.with_columns(pl.col(col).backward_fill())
        elif categorical_strategy == "constant":
            df = df.with_columns(pl.col(col).fill_null(str(constant_value)))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# String normalization (Rust-backed)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_whitespace(df: pl.DataFrame, columns=None) -> pl.DataFrame:
    """Trim and collapse internal whitespace in string columns (Rust)."""
    _header("Normalizing whitespace")
    return _apply_rust_str(df, _rustcore.normalize_whitespace, columns)


def normalize_unicode(df: pl.DataFrame, columns=None) -> pl.DataFrame:
    """NFKC-normalize and strip control/non-printable characters (Rust)."""
    _header("Normalizing Unicode (NFKC) + removing control chars")
    return _apply_rust_str(df, _rustcore.normalize_unicode, columns)


def fix_mojibake(df: pl.DataFrame, columns=None) -> pl.DataFrame:
    """Repair UTF-8-as-Latin-1 mojibake (e.g. ``Ã©`` -> ``é``) (Rust)."""
    _header("Fixing mojibake / encoding artifacts")
    return _apply_rust_str(df, _rustcore.fix_mojibake, columns)


def standardize_casing(df: pl.DataFrame, mapping: dict) -> pl.DataFrame:
    """Standardize casing per column. ``mapping``: col -> ``title|lower|upper``."""
    _header("Standardizing text casing")
    for col, mode in mapping.items():
        if col not in df.columns:
            continue
        expr = pl.col(col).cast(pl.Utf8, strict=False)
        if mode == "title":
            expr = expr.str.to_titlecase()
        elif mode == "lower":
            expr = expr.str.to_lowercase()
        elif mode == "upper":
            expr = expr.str.to_uppercase()
        df = df.with_columns(expr.alias(col))
    return df


def standardize_categories(df: pl.DataFrame, columns=None) -> pl.DataFrame:
    """Trim + lowercase categorical values so duplicates actually collapse (Rust)."""
    _header("Standardizing categorical values before dedup")
    df = _apply_rust_str(df, _rustcore.normalize_whitespace, columns)
    cols = columns if columns is not None else _string_cols(df)
    for col in cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).str.to_lowercase().alias(col))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Data types & parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_dates(df: pl.DataFrame, columns=None, min_success: float = 0.7) -> pl.DataFrame:
    """Detect and parse date/datetime string columns to native dtypes.

    A column is converted only if at least ``min_success`` of its non-null
    values parse successfully.
    """
    _header("Parsing dates to ISO 8601")
    candidates = columns if columns is not None else _string_cols(df)
    for col in candidates:
        if col not in df.columns or df[col].dtype != pl.Utf8:
            continue
        s = df[col]
        non_null = s.drop_nulls().len() or 1
        parsed = None
        # Fast path: let Polars infer a datetime/date format.
        for caster in (lambda x: x.str.to_datetime(strict=False),
                       lambda x: x.str.to_date(strict=False)):
            try:
                p = caster(s)
                if p.drop_nulls().len() / non_null >= min_success:
                    parsed = p
                    break
            except Exception:
                continue
        # Fallback: per-value parsing with dateutil (handles varied formats).
        if parsed is None:
            from dateutil import parser as dparser

            def _pv(v):
                try:
                    return dparser.parse(str(v))
                except Exception:
                    return None
            vals = [_pv(v) if v is not None else None for v in s.to_list()]
            if sum(v is not None for v in vals) / non_null >= min_success:
                parsed = pl.Series(col, vals)
        if parsed is not None:
            df = df.with_columns(parsed.alias(col))
            print(f"Parsed '{col}' as date/datetime")
    return df


def extract_date_parts(df: pl.DataFrame, columns=None) -> pl.DataFrame:
    """Add year/month/day/weekday/quarter columns for each date/datetime column."""
    _header("Extracting date parts")
    date_types = (pl.Date, pl.Datetime)
    cols = columns if columns is not None else [c for c, t in df.schema.items() if t in date_types]
    for col in cols:
        if col in df.columns and df[col].dtype in date_types:
            df = df.with_columns([
                pl.col(col).dt.year().alias(f"{col}_year"),
                pl.col(col).dt.month().alias(f"{col}_month"),
                pl.col(col).dt.day().alias(f"{col}_day"),
                pl.col(col).dt.weekday().alias(f"{col}_weekday"),
                pl.col(col).dt.quarter().alias(f"{col}_quarter"),
            ])
            print(f"Extracted parts from '{col}'")
    return df


def normalize_booleans(df: pl.DataFrame, columns=None) -> pl.DataFrame:
    """Convert boolean-like string columns (yes/no, Y/N, 1/0 ...) to real booleans (Rust)."""
    _header("Normalizing booleans")
    truthy = {"true", "t", "yes", "y", "1", "on", "false", "f", "no", "n", "0", "off"}
    if columns is None:
        columns = []
        for col in _string_cols(df):
            uniq = {str(v).strip().lower() for v in df[col].drop_nulls().unique().to_list()}
            if uniq and uniq.issubset(truthy):
                columns.append(col)
    for col in columns:
        if col in df.columns:
            mapped = _rustcore.normalize_booleans(df[col].cast(pl.Utf8, strict=False).to_list())
            as_bool = [True if m == "true" else False if m == "false" else None for m in mapped]
            df = df.with_columns(pl.Series(col, as_bool, dtype=pl.Boolean))
            print(f"Converted '{col}' to boolean")
    return df


def downcast_numeric(df: pl.DataFrame) -> pl.DataFrame:
    """Shrink numeric columns to the smallest dtype that fits (int64->int32 ...)."""
    _header("Downcasting numeric dtypes")
    num = _numeric_cols(df)
    if num:
        df = df.with_columns([pl.col(c).shrink_dtype().alias(c) for c in num])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Categorical
# ─────────────────────────────────────────────────────────────────────────────

def group_rare_categories(df: pl.DataFrame, threshold: float = 0.01,
                          other_label: str = "Other", columns=None) -> pl.DataFrame:
    """Replace categories rarer than ``threshold`` (fraction of rows) with ``other_label``."""
    _header(f"Grouping rare categories (< {threshold:.1%}) into '{other_label}'")
    n = df.height or 1
    cols = columns if columns is not None else _string_cols(df)
    for col in cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        val_col = vc.columns[0]
        rare = vc.filter(pl.col("count") / n < threshold)[val_col].to_list()
        if rare:
            df = df.with_columns(
                pl.when(pl.col(col).is_in(rare)).then(pl.lit(other_label))
                .otherwise(pl.col(col)).alias(col)
            )
            print(f"'{col}': grouped {len(rare)} rare categories")
    return df


def fuzzy_cluster_categories(df: pl.DataFrame, columns=None, threshold: int = 90) -> pl.DataFrame:
    """Merge near-identical category spellings to a canonical (most frequent) value.

    Uses rapidfuzz token-set ratio; values >= ``threshold`` similar are merged.
    """
    _header(f"Fuzzy-clustering near-identical categories (>= {threshold})")
    from rapidfuzz import fuzz
    cols = columns if columns is not None else _string_cols(df)
    for col in cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(sort=True)
        val_col = vc.columns[0]
        values = [v for v in vc[val_col].to_list() if v is not None]
        canonical, mapping = [], {}
        for val in values:                       # values already sorted by frequency desc
            match = next((c for c in canonical
                          if fuzz.token_sort_ratio(str(val).lower(), str(c).lower()) >= threshold), None)
            if match is None:
                canonical.append(val)
            else:
                mapping[val] = match
        if mapping:
            df = df.with_columns(pl.col(col).replace(mapping).alias(col))
            print(f"'{col}': merged {len(mapping)} spelling variants")
    return df


def encode_categorical(df: pl.DataFrame, method: str = "onehot", columns=None,
                       target: str = None, max_onehot: int = 20) -> pl.DataFrame:
    """Encode categorical columns.

    method: ``onehot|label|frequency|ordinal|target`` (``target`` requires ``target``).
    """
    _header(f"Encoding categorical columns ({method})")
    cols = columns if columns is not None else [c for c in _string_cols(df) if c != target]
    if method == "onehot":
        small = [c for c in cols if df[c].n_unique() <= max_onehot]
        if small:
            df = df.to_dummies(columns=small)
    elif method == "label":
        for col in cols:
            cats = df[col].drop_nulls().unique().to_list()
            df = df.with_columns(pl.col(col).replace_strict(
                {c: i for i, c in enumerate(cats)}, default=None).alias(col))
    elif method == "frequency":
        for col in cols:
            vc = df[col].value_counts()
            freq = dict(zip(vc[vc.columns[0]].to_list(), vc["count"].to_list()))
            df = df.with_columns(pl.col(col).replace(freq).cast(pl.Int64, strict=False).alias(col))
    elif method == "ordinal":
        for col in cols:
            cats = sorted(str(x) for x in df[col].drop_nulls().unique().to_list())
            df = df.with_columns(pl.col(col).cast(pl.Utf8).replace_strict(
                {c: i for i, c in enumerate(cats)}, default=None).alias(col))
    elif method == "target":
        if target is None or target not in df.columns:
            raise ValueError("target encoding requires a valid `target` column")
        for col in cols:
            means = df.group_by(col).agg(pl.col(target).mean().alias("_m"))
            mapping = dict(zip(means[col].to_list(), means["_m"].to_list()))
            df = df.with_columns(pl.col(col).replace(mapping).cast(pl.Float64, strict=False).alias(col))
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Outliers
# ─────────────────────────────────────────────────────────────────────────────

def remove_outliers_zscore(df: pl.DataFrame, threshold: float = 3.0, columns=None) -> pl.DataFrame:
    """Remove rows where any numeric column's z-score exceeds ``threshold``."""
    _header(f"Removing outliers (z-score > {threshold})")
    cols = columns if columns is not None else _numeric_cols(df)
    for col in cols:
        mean, std = df[col].mean(), df[col].std()
        if std and std > 0:
            df = df.filter(((pl.col(col) - mean) / std).abs() <= threshold)
    return df


def remove_outliers_mad(df: pl.DataFrame, threshold: float = 3.5, columns=None) -> pl.DataFrame:
    """Remove rows using the modified z-score (median absolute deviation)."""
    _header(f"Removing outliers (modified z-score / MAD > {threshold})")
    cols = columns if columns is not None else _numeric_cols(df)
    for col in cols:
        vals = np.array([v for v in df[col].to_list() if v is not None], dtype=float)
        if len(vals) == 0:
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        if mad == 0:
            continue
        df = df.filter((0.6745 * (pl.col(col) - med) / mad).abs() <= threshold)
    return df


def remove_outliers_isolation_forest(df: pl.DataFrame, contamination: float = 0.05,
                                     columns=None) -> pl.DataFrame:
    """Remove multivariate outliers with an Isolation Forest over numeric columns."""
    _header(f"Removing multivariate outliers (Isolation Forest, contamination={contamination})")
    from sklearn.ensemble import IsolationForest
    cols = columns if columns is not None else _numeric_cols(df)
    if not cols:
        return df
    arr = df.select(cols).fill_null(strategy="mean").to_numpy()
    labels = IsolationForest(contamination=contamination, random_state=42).fit_predict(arr)
    keep = (labels == 1).tolist()
    print(f"Removed {keep.count(False)} rows")
    return df.filter(pl.Series(keep))


def winsorize(df: pl.DataFrame, lower: float = 0.05, upper: float = 0.05, columns=None) -> pl.DataFrame:
    """Cap (rather than drop) numeric values at the given lower/upper quantiles."""
    _header(f"Winsorizing numeric columns ({lower:.0%} / {upper:.0%})")
    cols = columns if columns is not None else _numeric_cols(df)
    for col in cols:
        lo, hi = df[col].quantile(lower), df[col].quantile(1 - upper)
        if lo is not None and hi is not None:
            df = df.with_columns(pl.col(col).clip(lo, hi).alias(col))
    return df


def remove_outliers_by_group(df: pl.DataFrame, group_col: str, columns=None,
                             k: float = 1.5) -> pl.DataFrame:
    """Remove IQR outliers computed *within* each group of ``group_col``."""
    _header(f"Removing outliers per group of '{group_col}'")
    cols = columns if columns is not None else _numeric_cols(df)
    for col in cols:
        q1 = pl.col(col).quantile(0.25).over(group_col)
        q3 = pl.col(col).quantile(0.75).over(group_col)
        iqr = q3 - q1
        df = df.filter((pl.col(col) >= q1 - k * iqr) & (pl.col(col) <= q3 + k * iqr))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Duplicates
# ─────────────────────────────────────────────────────────────────────────────

def drop_duplicates_subset(df: pl.DataFrame, subset=None, keep: str = "first") -> pl.DataFrame:
    """Drop duplicate rows on an optional ``subset`` of key columns."""
    _header(f"Dropping duplicates (subset={subset}, keep={keep})")
    before = df.height
    df = df.unique(subset=subset, keep=keep, maintain_order=True)
    print(f"Removed {before - df.height} duplicate rows")
    return df


def fuzzy_deduplicate(df: pl.DataFrame, key_columns, threshold: int = 90,
                      keep: str = "first") -> pl.DataFrame:
    """Remove near-duplicate rows by fuzzy-matching a concatenated key.

    Rows whose key is >= ``threshold`` similar to an already-kept row are dropped.
    O(n²) — intended for moderate row counts / after blocking.
    """
    _header(f"Fuzzy de-duplication on {key_columns} (>= {threshold})")
    from rapidfuzz import fuzz
    keys = (df.select([pl.col(c).cast(pl.Utf8, strict=False).fill_null("") for c in key_columns])
              .to_numpy().tolist())
    keys = [" ".join(row).strip().lower() for row in keys]
    kept_keys, keep_mask = [], []
    for k in keys:
        dup = any(fuzz.token_sort_ratio(k, kk) >= threshold for kk in kept_keys)
        keep_mask.append(not dup)
        if not dup:
            kept_keys.append(k)
    if keep == "last":
        keep_mask = keep_mask[::-1]
    print(f"Removed {keep_mask.count(False)} near-duplicate rows")
    return df.filter(pl.Series(keep_mask))


# ─────────────────────────────────────────────────────────────────────────────
# Numeric scaling
# ─────────────────────────────────────────────────────────────────────────────

def scale_numeric(df: pl.DataFrame, method: str = "standard", columns=None) -> pl.DataFrame:
    """Scale numeric columns: ``minmax|standard|robust``."""
    _header(f"Scaling numeric columns ({method})")
    cols = columns if columns is not None else _numeric_cols(df)
    for col in cols:
        s = df[col]
        if method == "minmax":
            lo, hi = s.min(), s.max()
            if hi is not None and lo is not None and hi != lo:
                df = df.with_columns(((pl.col(col) - lo) / (hi - lo)).alias(col))
        elif method == "standard":
            mean, std = s.mean(), s.std()
            if std and std > 0:
                df = df.with_columns(((pl.col(col) - mean) / std).alias(col))
        elif method == "robust":
            med, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
            iqr = (q3 - q1) if (q1 is not None and q3 is not None) else 0
            if iqr:
                df = df.with_columns(((pl.col(col) - med) / iqr).alias(col))
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    return df


def bin_numeric(df: pl.DataFrame, columns, bins: int = 5, strategy: str = "quantile") -> pl.DataFrame:
    """Discretize numeric columns into ``bins`` buckets: ``quantile|uniform``.

    Adds a ``<col>_bin`` categorical column.
    """
    _header(f"Binning numeric columns into {bins} buckets ({strategy})")
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        if strategy == "quantile":
            qs = [i / bins for i in range(1, bins)]
            breaks = sorted(set(v for v in (s.quantile(q) for q in qs) if v is not None))
        else:
            lo, hi = s.min(), s.max()
            if lo is None or hi is None or lo == hi:
                continue
            step = (hi - lo) / bins
            breaks = [lo + step * i for i in range(1, bins)]
        if breaks:
            df = df.with_columns(s.cut(breaks).alias(f"{col}_bin"))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Validation & consistency
# ─────────────────────────────────────────────────────────────────────────────

def apply_range_rules(df: pl.DataFrame, rules: dict, action: str = "null") -> pl.DataFrame:
    """Enforce numeric ranges. ``rules``: col -> (min, max). ``action``: ``null|clip|drop``."""
    _header(f"Applying range rules ({action})")
    for col, (lo, hi) in rules.items():
        if col not in df.columns:
            continue
        in_range = (pl.col(col) >= lo) & (pl.col(col) <= hi)
        n_bad = df.filter(~in_range & pl.col(col).is_not_null()).height
        if action == "null":
            df = df.with_columns(pl.when(in_range).then(pl.col(col)).otherwise(None).alias(col))
        elif action == "clip":
            df = df.with_columns(pl.col(col).clip(lo, hi).alias(col))
        elif action == "drop":
            df = df.filter(in_range | pl.col(col).is_null())
        print(f"'{col}': {n_bad} values outside [{lo}, {hi}]")
    return df


def check_cross_field(df: pl.DataFrame, checks: dict, drop_violations: bool = False) -> pl.DataFrame:
    """Check cross-field consistency rules.

    ``checks``: name -> Polars boolean expression that is True for VALID rows,
    e.g. ``{"date_order": pl.col("start") <= pl.col("end")}``. Reports violation
    counts; optionally drops violating rows.
    """
    _header("Checking cross-field consistency")
    for name, valid_expr in checks.items():
        violations = df.filter(~valid_expr.fill_null(True)).height
        print(f"Rule '{name}': {violations} violating rows")
        if drop_violations and violations:
            df = df.filter(valid_expr.fill_null(True))
    return df


_FORMAT_PATTERNS = {
    "email": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    "url": r"^https?://[^\s]+$",
    "ip": r"^(\d{1,3}\.){3}\d{1,3}$",
    "phone": r"^\+?[\d\s\-().]{7,}$",
    "postal": r"^[A-Za-z0-9][A-Za-z0-9\s\-]{2,9}$",
}


def validate_formats(df: pl.DataFrame, specs: dict, add_flag: bool = True) -> pl.DataFrame:
    """Validate string columns against a format. ``specs``: col -> ``email|url|ip|phone|postal``.

    Adds a boolean ``<col>_valid`` column (when ``add_flag``) and reports invalid counts.
    """
    _header("Validating formats")
    for col, kind in specs.items():
        if col not in df.columns or kind not in _FORMAT_PATTERNS:
            continue
        valid = pl.col(col).cast(pl.Utf8, strict=False).str.contains(_FORMAT_PATTERNS[kind])
        invalid = df.filter(~valid.fill_null(False) & pl.col(col).is_not_null()).height
        if add_flag:
            df = df.with_columns(valid.alias(f"{col}_valid"))
        print(f"'{col}' ({kind}): {invalid} invalid values")
    return df


def check_referential_integrity(df: pl.DataFrame, column: str, valid_values,
                                drop_invalid: bool = False) -> pl.DataFrame:
    """Report (optionally drop) rows whose ``column`` value is not in ``valid_values``."""
    _header(f"Checking referential integrity of '{column}'")
    valid_values = list(valid_values)
    bad = df.filter(~pl.col(column).is_in(valid_values) & pl.col(column).is_not_null()).height
    print(f"'{column}': {bad} rows reference values not in the allowed set")
    if drop_invalid and bad:
        df = df.filter(pl.col(column).is_in(valid_values) | pl.col(column).is_null())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific standardization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_phone_numbers(df: pl.DataFrame, columns, region: str = "US") -> pl.DataFrame:
    """Normalize phone numbers to E.164 (e.g. ``+14155552671``) using ``phonenumbers``."""
    _header(f"Normalizing phone numbers to E.164 (region={region})")
    import phonenumbers

    def norm(v):
        if v is None:
            return None
        try:
            p = phonenumbers.parse(str(v), region)
            if phonenumbers.is_valid_number(p):
                return phonenumbers.format_number(p, phonenumbers.PhoneNumberFormat.E164)
        except Exception:
            pass
        return None
    for col in columns:
        if col in df.columns:
            df = df.with_columns(pl.Series(col, [norm(v) for v in df[col].to_list()]))
    return df


def normalize_emails(df: pl.DataFrame, columns, add_flag: bool = True) -> pl.DataFrame:
    """Lowercase/trim emails and flag validity (``email_validator`` if available)."""
    _header("Normalizing emails")
    try:
        from email_validator import validate_email, EmailNotValidError
        have_validator = True
    except Exception:
        have_validator = False

    def norm(v):
        if v is None:
            return (None, None)
        s = str(v).strip().lower()
        if have_validator:
            try:
                validate_email(s, check_deliverability=False)
                return (s, True)
            except Exception:
                return (s, False)
        return (s, bool(re.match(_FORMAT_PATTERNS["email"], s)))
    for col in columns:
        if col in df.columns:
            pairs = [norm(v) for v in df[col].to_list()]
            df = df.with_columns(pl.Series(col, [p[0] for p in pairs]))
            if add_flag:
                df = df.with_columns(pl.Series(f"{col}_valid", [p[1] for p in pairs]))
    return df


def canonicalize_urls(df: pl.DataFrame, columns) -> pl.DataFrame:
    """Canonicalize URLs: add scheme, lowercase host, strip fragments/trailing slash."""
    _header("Canonicalizing URLs")

    def canon(v):
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        if "://" not in s:
            s = "http://" + s
        parts = urlsplit(s)
        path = parts.path.rstrip("/")
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))
    for col in columns:
        if col in df.columns:
            df = df.with_columns(pl.Series(col, [canon(v) for v in df[col].to_list()]))
    return df


def standardize_country_region(df: pl.DataFrame, columns, mapping: dict = None) -> pl.DataFrame:
    """Standardize country names/codes to a canonical form using a mapping table."""
    _header("Standardizing country / region names")
    default_map = {
        "usa": "United States", "us": "United States", "u.s.a.": "United States",
        "united states of america": "United States", "uk": "United Kingdom",
        "u.k.": "United Kingdom", "england": "United Kingdom", "uae": "United Arab Emirates",
        "india": "India", "in": "India", "bharat": "India",
    }
    table = {**default_map, **{k.lower(): v for k, v in (mapping or {}).items()}}
    for col in columns:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase()
                .replace(table).alias(col)
            )
    return df


def convert_currency(df: pl.DataFrame, columns, rates: dict, base: str = "USD") -> pl.DataFrame:
    """Convert numeric currency columns to ``base`` using ``rates`` (col -> multiplier)."""
    _header(f"Converting currency columns to {base}")
    for col in columns:
        if col in df.columns and col in rates:
            df = df.with_columns((pl.col(col).cast(pl.Float64, strict=False) * rates[col]).alias(col))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Text (beyond core cleaning)
# ─────────────────────────────────────────────────────────────────────────────

def detect_language(df: pl.DataFrame, column: str, out_col: str = None) -> pl.DataFrame:
    """Add a detected-language column (ISO code) using ``langdetect`` (optional)."""
    _header(f"Detecting language of '{column}'")
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    out_col = out_col or f"{column}_lang"

    def lang(v):
        try:
            return detect(str(v)) if v and str(v).strip() else None
        except Exception:
            return None
    return df.with_columns(pl.Series(out_col, [lang(v) for v in df[column].to_list()]))


def filter_by_language(df: pl.DataFrame, column: str, keep: str = "en") -> pl.DataFrame:
    """Keep only rows whose ``column`` text is detected as language ``keep``."""
    _header(f"Filtering rows to language '{keep}'")
    tmp = detect_language(df, column, out_col="__lang__")
    kept = tmp.filter(pl.col("__lang__") == keep).drop("__lang__")
    print(f"Kept {kept.height} / {df.height} rows")
    return kept


def correct_spelling_text(df: pl.DataFrame, columns) -> pl.DataFrame:
    """Correct spelling in free-text columns using ``pyspellchecker`` (optional)."""
    _header("Correcting spelling in text columns")
    from spellchecker import SpellChecker
    sp = SpellChecker()

    def fix(v):
        if v is None or not str(v).strip():
            return v
        out = []
        for w in str(v).split():
            corrected = sp.correction(w)
            out.append(corrected if corrected else w)
        return " ".join(out)
    for col in columns:
        if col in df.columns:
            df = df.with_columns(pl.Series(col, [fix(v) for v in df[col].to_list()]))
    return df


def redact_profanity(df: pl.DataFrame, columns, replacement: str = "****") -> pl.DataFrame:
    """Redact profanity in text columns using ``better_profanity`` (optional)."""
    _header("Redacting profanity")
    from better_profanity import profanity
    profanity.load_censor_words()
    for col in columns:
        if col in df.columns:
            df = df.with_columns(pl.Series(
                col, [profanity.censor(str(v), replacement[0]) if v is not None else None
                      for v in df[col].to_list()]))
    return df


def near_duplicate_text(df: pl.DataFrame, column: str, threshold: int = 90,
                        drop: bool = False) -> pl.DataFrame:
    """Flag (or drop) rows whose ``column`` text is a near-duplicate of an earlier row.

    Adds a boolean ``<column>_near_dup`` flag. O(n²) — use on moderate sizes.
    """
    _header(f"Detecting near-duplicate text in '{column}' (>= {threshold})")
    from rapidfuzz import fuzz
    texts = [str(v).strip().lower() if v is not None else "" for v in df[column].to_list()]
    seen, flags = [], []
    for t in texts:
        dup = bool(t) and any(fuzz.token_sort_ratio(t, s) >= threshold for s in seen)
        flags.append(dup)
        if not dup and t:
            seen.append(t)
    print(f"Found {sum(flags)} near-duplicate rows")
    if drop:
        return df.filter(~pl.Series(flags))
    return df.with_columns(pl.Series(f"{column}_near_dup", flags))
