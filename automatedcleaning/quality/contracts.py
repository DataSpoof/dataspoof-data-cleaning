"""Data contracts: declare what valid data looks like, then validate every batch against it.

A contract is plain YAML/JSON, so it can live in version control and be reviewed like code::

    name: customers
    extra_columns: warn          # allow | warn | error
    no_duplicate_rows: true
    unique: [customer_id]        # key columns (a list, or a list of lists)
    row_count: {min: 100}
    columns:
      customer_id: {type: integer, required: true, unique: true}
      email:       {type: string, format: email, max_null_fraction: 0.01}
      age:         {type: number, min: 0, max: 120, warn_only: [min, max]}
      plan:        {type: string, allowed: [basic, pro, enterprise]}

Column rules: ``type`` (integer | number | string | boolean | date | datetime | any),
``required`` (no nulls), ``max_null_fraction``, ``unique``, ``min``, ``max``, ``allowed``,
``pattern`` (regex), ``format`` (email | url | uuid | ipv4 | phone), ``min_length``,
``max_length``. Every rule is an ``error`` by default; ``severity: warn`` softens a whole
column and ``warn_only: [rule, ...]`` softens individual rules.

``Contract.infer(df)`` writes a starting contract from data you trust; structural rules
(columns, types, required, unique) are errors and statistical ones (ranges, allowed values,
lengths, null fractions) are warnings, because new data legitimately drifts. Review and
tighten it before enforcing.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import difflib
import json
import re
from pathlib import Path

import polars as pl

from .._detect import (EMAIL_RE, IPV4_RE, URL_RE, UUID_RE, is_num, is_temporal, is_text,
                       phone_ok)
from .checks import jsonable

TYPES = ("integer", "number", "string", "boolean", "date", "datetime", "any")
FORMATS = {"email": EMAIL_RE, "url": URL_RE, "uuid": UUID_RE, "ipv4": IPV4_RE, "phone": None}
COLUMN_RULES = ("column_present", "type", "required", "max_null_fraction", "unique", "min", "max",
                "allowed", "pattern", "format", "min_length", "max_length")
_COLUMN_KEYS = set(COLUMN_RULES[1:]) | {"severity", "warn_only", "description"}
_TABLE_KEYS = {"name", "version", "description", "columns", "row_count", "unique",
               "no_duplicate_rows", "extra_columns", "severity"}
_SEVERITIES = ("error", "warn")


class ContractError(ValueError):
    """The contract definition itself is invalid."""


class ContractViolation(Exception):
    """Data failed a contract. ``.result`` holds the full :class:`ContractResult`."""

    def __init__(self, result: "ContractResult"):
        self.result = result
        super().__init__(result.summary())


@dataclasses.dataclass
class RuleResult:
    rule: str
    column: str | None
    passed: bool
    severity: str
    failed: int
    checked: int
    message: str
    examples: list


@dataclasses.dataclass
class ContractResult:
    """Outcome of validating one DataFrame. ``passed`` is False only if an *error* rule failed."""

    contract: str
    rows: int
    results: list

    @property
    def failures(self) -> list:
        return [r for r in self.results if not r.passed]

    @property
    def errors(self) -> list:
        return [r for r in self.failures if r.severity == "error"]

    @property
    def warnings(self) -> list:
        return [r for r in self.failures if r.severity == "warn"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> "ContractResult":
        if not self.passed:
            raise ContractViolation(self)
        return self

    def to_dict(self) -> dict:
        return jsonable({"contract": self.contract, "rows": self.rows, "passed": self.passed,
                         "errors": len(self.errors), "warnings": len(self.warnings),
                         "results": [dataclasses.asdict(r) for r in self.results]})

    def to_json(self, path=None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_frame(self) -> pl.DataFrame:
        schema = {"rule": pl.String, "column": pl.String, "passed": pl.Boolean,
                  "severity": pl.String, "failed": pl.Int64, "checked": pl.Int64,
                  "message": pl.String}
        return pl.DataFrame([{k: getattr(r, k) for k in schema} for r in self.results], schema=schema)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Contract '{self.contract}': {status} | {self.rows:,} rows | "
                 f"{len(self.results)} rules | {len(self.errors)} errors, {len(self.warnings)} warnings"]
        for r in self.failures:
            where = f"{r.column}." if r.column else ""
            lines.append(f"  [{'ERROR' if r.severity == 'error' else 'warn '}] {where}{r.rule}: {r.message}")
        return "\n".join(lines)

    __str__ = summary

    def __repr__(self) -> str:
        return self.summary()


# ── helpers ──────────────────────────────────────────────────────────────────

def _type_of(dtype) -> str:
    if dtype == pl.Boolean:
        return "boolean"
    if dtype.is_integer():
        return "integer"
    if is_num(dtype):
        return "number"
    if dtype == pl.Date:
        return "date"
    if dtype == pl.Datetime:
        return "datetime"
    if is_text(dtype):
        return "string"
    return "any"


def _type_matches(expected: str, dtype) -> bool:
    return {
        "integer": dtype.is_integer(),
        "number": is_num(dtype),
        "string": is_text(dtype),
        "boolean": dtype == pl.Boolean,
        "date": dtype == pl.Date,
        "datetime": is_temporal(dtype),
        "any": True,
    }[expected]


def _examples(series: pl.Series, k: int = 5) -> list:
    try:
        return jsonable(series.unique(maintain_order=True).head(k).to_list())
    except Exception:
        return jsonable(series.head(k).to_list())


def _bound(value, dtype):
    """Convert a contract min/max to something comparable with ``dtype``."""
    if is_temporal(dtype) and isinstance(value, str):
        parsed = _dt.datetime.fromisoformat(value)
        return parsed.date() if dtype == pl.Date else parsed
    return value


def _closest(key, options) -> str:
    match = difflib.get_close_matches(key, sorted(options), n=1)
    return f" Did you mean '{match[0]}'?" if match else ""


# ── contract ─────────────────────────────────────────────────────────────────

class Contract:
    """A data contract. Build it from a dict/YAML/JSON, or infer one with ``Contract.infer``."""

    def __init__(self, columns: dict | None = None, *, name: str = "contract", version=1,
                 description: str = "", row_count: dict | None = None, unique=None,
                 no_duplicate_rows: bool = False, extra_columns: str = "warn",
                 severity: str = "error"):
        self.name, self.version, self.description = name, version, description
        self.row_count, self.no_duplicate_rows = row_count, bool(no_duplicate_rows)
        self.extra_columns, self.severity = extra_columns, severity
        if unique and isinstance(unique, (list, tuple)) and all(isinstance(u, str) for u in unique):
            unique = [list(unique)]
        self.unique = [list(k) for k in (unique or [])]
        self.columns = {str(c): dict(spec or {}) for c, spec in (columns or {}).items()}
        self._check_definition()

    def _check_definition(self):
        if self.extra_columns not in ("allow", "warn", "error"):
            raise ContractError("extra_columns must be 'allow', 'warn' or 'error'")
        if self.severity not in _SEVERITIES:
            raise ContractError("severity must be 'error' or 'warn'")
        if self.row_count and set(self.row_count) - {"min", "max"}:
            raise ContractError("row_count accepts only 'min' and 'max'")
        for col, spec in self.columns.items():
            for key in spec:
                if key not in _COLUMN_KEYS:
                    raise ContractError(f"Column '{col}': unknown rule '{key}'.{_closest(key, _COLUMN_KEYS)}")
            if "type" in spec and spec["type"] not in TYPES:
                raise ContractError(f"Column '{col}': type must be one of {TYPES}")
            if "format" in spec and spec["format"] not in FORMATS:
                raise ContractError(f"Column '{col}': format must be one of {sorted(FORMATS)}")
            if spec.get("severity", "error") not in _SEVERITIES:
                raise ContractError(f"Column '{col}': severity must be 'error' or 'warn'")
            for rule in spec.get("warn_only", []):
                if rule not in COLUMN_RULES:
                    raise ContractError(f"Column '{col}': warn_only has unknown rule '{rule}'."
                                        f"{_closest(rule, COLUMN_RULES)}")
            if "pattern" in spec:
                try:
                    re.compile(spec["pattern"])
                except re.error as exc:
                    raise ContractError(f"Column '{col}': invalid pattern: {exc}") from None
            if "allowed" in spec and not isinstance(spec["allowed"], (list, tuple)):
                raise ContractError(f"Column '{col}': allowed must be a list")

    # -- (de)serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        d = {"name": self.name, "version": self.version}
        if self.description:
            d["description"] = self.description
        if self.severity != "error":
            d["severity"] = self.severity
        d["extra_columns"] = self.extra_columns
        if self.row_count:
            d["row_count"] = self.row_count
        if self.no_duplicate_rows:
            d["no_duplicate_rows"] = True
        if self.unique:
            d["unique"] = self.unique
        d["columns"] = self.columns
        return jsonable(d)

    @classmethod
    def from_dict(cls, data: dict) -> "Contract":
        unknown = set(data) - _TABLE_KEYS
        if unknown:
            key = sorted(unknown)[0]
            raise ContractError(f"Unknown contract key '{key}'.{_closest(key, _TABLE_KEYS)}")
        return cls(data.get("columns"), **{k: v for k, v in data.items() if k != "columns"})

    @classmethod
    def from_yaml(cls, text: str) -> "Contract":
        import yaml
        return cls.from_dict(yaml.safe_load(text) or {})

    @classmethod
    def from_file(cls, path) -> "Contract":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            return cls.from_yaml(text)
        return cls.from_dict(json.loads(text))

    def to_yaml(self, path=None) -> str:
        import yaml
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_json(self, path=None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def __repr__(self) -> str:
        return f"Contract({self.name!r}, {len(self.columns)} columns)"

    # -- inference -----------------------------------------------------------
    @classmethod
    def infer(cls, df, *, name: str = "inferred-contract", null_slack: float = 0.05,
              max_allowed_values: int = 20, range_margin: float = 0.0,
              extra_columns: str = "warn") -> "Contract":
        """Write a starting contract from data you trust (review it before enforcing).

        ``null_slack`` is added to the observed null fraction; ``range_margin`` widens
        numeric ranges by that share of the observed range.
        """
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)
        rows = df.height or 1
        columns = {}
        for c, dtype in df.schema.items():
            s, spec, soft = df[c], {}, []
            nn = s.drop_nulls()
            spec["type"] = _type_of(dtype)
            null_fraction = s.null_count() / rows
            if null_fraction == 0:
                spec["required"] = True
            else:
                spec["max_null_fraction"] = round(min(1.0, null_fraction + null_slack), 4)
                soft.append("max_null_fraction")
            if nn.len():
                n_unique = nn.n_unique()
                if (n_unique == nn.len() and df.height > 10 and not dtype.is_float()
                        and dtype != pl.Boolean):
                    spec["unique"] = True
                if is_num(dtype) and dtype != pl.Boolean:
                    lo, hi = nn.min(), nn.max()
                    margin = (hi - lo) * range_margin
                    if dtype.is_integer() and margin:
                        margin = int(round(margin))
                    spec["min"], spec["max"] = lo - margin, hi + margin
                    soft += ["min", "max"]
                elif is_text(dtype):
                    if n_unique <= max_allowed_values and n_unique < nn.len():
                        spec["allowed"] = sorted(nn.unique().to_list())
                        soft.append("allowed")
                    else:
                        lengths = nn.str.len_chars()
                        spec["min_length"], spec["max_length"] = int(lengths.min()), int(lengths.max())
                        soft += ["min_length", "max_length"]
                    if nn.str.contains(EMAIL_RE).all():
                        spec["format"] = "email"
            if soft:
                spec["warn_only"] = soft
            columns[c] = spec
        return cls(columns, name=name, extra_columns=extra_columns)

    # -- validation ----------------------------------------------------------
    def validate(self, df) -> ContractResult:
        """Check ``df`` against every rule and return a :class:`ContractResult`."""
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)
        results: list[RuleResult] = []
        rows = df.height

        def add(rule, column, passed, severity, failed=0, checked=0, message="", examples=()):
            results.append(RuleResult(rule, column, bool(passed), severity, int(failed), int(checked),
                                      message, list(examples)))

        for col, spec in self.columns.items():
            soft = set(spec.get("warn_only", []))
            col_severity = spec.get("severity", self.severity)

            def sev(rule, _soft=soft, _sev=col_severity):
                return "warn" if rule in _soft else _sev

            if col not in df.columns:
                add("column_present", col, False, sev("column_present"), 1, 1, "column is missing")
                continue
            s = df[col]
            dtype, nn = s.dtype, s.drop_nulls()
            nulls = s.null_count()

            if spec.get("type", "any") != "any":
                ok = _type_matches(spec["type"], dtype)
                add("type", col, ok, sev("type"), 0 if ok else nn.len(), nn.len(),
                    f"expected {spec['type']}, found {dtype}" if not ok else "ok")
            if spec.get("required"):
                add("required", col, nulls == 0, sev("required"), nulls, rows,
                    f"{nulls:,} null values" if nulls else "ok")
            if "max_null_fraction" in spec:
                fraction = nulls / rows if rows else 0.0
                ok = fraction <= spec["max_null_fraction"]
                add("max_null_fraction", col, ok, sev("max_null_fraction"), nulls, rows,
                    f"{fraction:.1%} null (limit {spec['max_null_fraction']:.1%})")
            if spec.get("unique"):
                dups = nn.filter(nn.is_duplicated())
                count = nn.len() - nn.n_unique()
                add("unique", col, count == 0, sev("unique"), count, nn.len(),
                    f"{count:,} duplicate values" if count else "ok", _examples(dups))

            for rule in ("min", "max"):
                if rule not in spec:
                    continue
                if not (is_num(dtype) or is_temporal(dtype)):
                    add(rule, col, False, sev(rule), nn.len(), nn.len(),
                        f"can't compare {dtype} values with a {rule}")
                    continue
                bound = _bound(spec[rule], dtype)
                bad = nn.filter(nn < bound) if rule == "min" else nn.filter(nn > bound)
                word = "below" if rule == "min" else "above"
                add(rule, col, bad.len() == 0, sev(rule), bad.len(), nn.len(),
                    f"{bad.len():,} values {word} {spec[rule]}" if bad.len() else "ok",
                    _examples(bad.sort()))

            if "allowed" in spec:
                allowed = pl.Series(list(spec["allowed"])).cast(dtype, strict=False).drop_nulls().to_list()
                bad = nn.filter(~nn.is_in(allowed)) if allowed else nn
                add("allowed", col, bad.len() == 0, sev("allowed"), bad.len(), nn.len(),
                    f"{bad.len():,} values outside the allowed set" if bad.len() else "ok",
                    _examples(bad))

            text = nn.cast(pl.String, strict=False) if nn.len() else nn
            if "pattern" in spec:
                bad = text.filter(~text.str.contains(spec["pattern"]))
                add("pattern", col, bad.len() == 0, sev("pattern"), bad.len(), nn.len(),
                    f"{bad.len():,} values don't match {spec['pattern']}" if bad.len() else "ok",
                    _examples(bad))
            if "format" in spec:
                fmt_name = spec["format"]
                ok_mask = phone_ok(text) if fmt_name == "phone" else text.str.contains(FORMATS[fmt_name])
                bad = text.filter(~ok_mask)
                add("format", col, bad.len() == 0, sev("format"), bad.len(), nn.len(),
                    f"{bad.len():,} values aren't valid {fmt_name}" if bad.len() else "ok",
                    _examples(bad))
            for rule in ("min_length", "max_length"):
                if rule in spec:
                    lengths = text.str.len_chars()
                    mask = lengths < spec[rule] if rule == "min_length" else lengths > spec[rule]
                    bad = text.filter(mask)
                    word = "shorter" if rule == "min_length" else "longer"
                    add(rule, col, bad.len() == 0, sev(rule), bad.len(), nn.len(),
                        f"{bad.len():,} values {word} than {spec[rule]} characters" if bad.len() else "ok",
                        _examples(bad))

        if self.row_count:
            lo, hi = self.row_count.get("min"), self.row_count.get("max")
            ok = (lo is None or rows >= lo) and (hi is None or rows <= hi)
            add("row_count", None, ok, self.severity, 0 if ok else 1, 1,
                f"{rows:,} rows (expected {lo if lo is not None else '-'}..{hi if hi is not None else '-'})")
        if self.no_duplicate_rows:
            try:
                extra = rows - df.unique().height
            except Exception:
                extra = 0
            add("no_duplicate_rows", None, extra == 0, self.severity, extra, rows,
                f"{extra:,} duplicate rows" if extra else "ok")
        for key in self.unique:
            missing = [k for k in key if k not in df.columns]
            label = "+".join(key)
            if missing:
                add("unique_key", label, False, self.severity, 1, 1, f"key columns missing: {missing}")
                continue
            dups = int(df.select(key).is_duplicated().sum())
            add("unique_key", label, dups == 0, self.severity, dups, rows,
                f"{dups:,} rows share a key with another row" if dups else "ok")
        if self.extra_columns != "allow":
            extra_cols = [c for c in df.columns if c not in self.columns]
            add("extra_columns", None, not extra_cols,
                "error" if self.extra_columns == "error" else "warn", len(extra_cols), df.width,
                f"unexpected columns: {extra_cols}" if extra_cols else "ok")
        return ContractResult(self.name, rows, results)
