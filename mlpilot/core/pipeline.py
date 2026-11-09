"""``Pipeline``: an ordered, savable, auditable sequence of steps.

* ``fit`` / ``fit_transform`` learn every step's rules on training data.
* ``transform`` replays exactly those rules on new data (test sets, production batches).
* ``save`` / ``load`` persist a fitted pipeline as plain JSON (never pickle), so it is safe to
  load and easy to review or diff.
* Every run produces a ``RunReport`` (rows, columns and nulls per step, timings, data
  fingerprints, config hash, versions), optionally appended to an audit log.
"""

from __future__ import annotations

from typing import Any

import contextlib
import dataclasses
import functools
import datetime as _dt
import difflib
import hashlib
import io
import json
import math
import platform
import re
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from . import registry
from .step import FunctionStep, NotFittedError, Step

FORMAT = "dataspoof.pipeline"
FORMAT_VERSION = 1
SCHEMA_POLICIES = ("warn", "raise", "ignore")
_LOG_LIMIT = 2000


class SchemaMismatchError(ValueError):
    """Input to ``transform`` doesn't match the training schema (policy ``"raise"``)."""


class MLPilotWarning(UserWarning):
    """Base category for every warning this library issues (filter them all with this class)."""


AutomatedCleaningWarning = MLPilotWarning      # the name before 2.1.0 (AutomatedCleaning); same class


class ServiceWarning(MLPilotWarning, RuntimeWarning):
    """An optional external service (LLM, alert webhook) failed; a fallback was used."""


class PipelineWarning(MLPilotWarning):
    """Warning category for pipeline problems such as schema drift."""


class PipelineStepError(RuntimeError):
    """A step failed; the message names the step and the phase."""


# ── helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _schema(df: pl.DataFrame) -> dict:
    return {c: str(t) for c, t in df.schema.items()}


def _nulls(df: pl.DataFrame) -> int:
    return int(sum(df.null_count().row(0))) if df.width else 0


def _fingerprint(df: pl.DataFrame) -> dict:
    schema = _schema(df)
    fp = {
        "rows": df.height,
        "columns": df.width,
        "nulls": _nulls(df),
        "schema_hash": hashlib.sha256(json.dumps(list(schema.items())).encode()).hexdigest()[:16],
        "content_hash": None,
    }
    try:  # order-independent; stable for a given Polars version
        hashes = df.hash_rows().to_numpy()
        fp["content_hash"] = f"{int(np.sum(hashes, dtype=np.uint64)):016x}"
    except Exception:
        pass
    return fp


def _jsonable(obj):
    """Convert to plain JSON types (NaN/inf -> None, dates -> ISO strings)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (_dt.date, _dt.datetime, _dt.time)):
        return obj.isoformat()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _versions() -> dict:
    return dict(_versions_cached())


@functools.lru_cache(maxsize=1)
def _versions_cached() -> dict:
    from importlib import metadata

    out = {"format": FORMAT_VERSION, "python": platform.python_version(),
           "polars": pl.__version__}
    try:
        out["mlpilot"] = metadata.version("mlpilot")
    except metadata.PackageNotFoundError:
        out["mlpilot"] = "source"
    return out


def _parse_dtype(name: str):
    """Polars dtype from its string form ('Float64', "Datetime(time_unit='us', time_zone=None)")."""
    simple = getattr(pl, name, None)
    if isinstance(simple, type) and issubclass(simple, pl.DataType):
        return simple
    match = re.fullmatch(r"Datetime\(time_unit='(\w+)', time_zone=(?:None|'([^']+)')\)", name)
    if match:
        return pl.Datetime(match.group(1), match.group(2))  # type: ignore[arg-type]
    return None


def _read_config(path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Reading YAML configs needs PyYAML: pip install pyyaml") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def make_step(spec) -> Step:
    """Build a step from a ``Step``, a name, ``(name, params)`` or ``{"step", "params", ...}``.

    Registered stateful steps win over functions of the same name; prefix a name with
    ``fn:`` to force the plain-function version.
    """
    if isinstance(spec, Step):
        return spec
    fit_only = None
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, (tuple, list)) and len(spec) == 2:
        name, params = spec[0], dict(spec[1] or {})
    elif isinstance(spec, dict):
        unknown = set(spec) - {"step", "params", "fit_only", "state"}
        if unknown or "step" not in spec:
            raise ValueError(f"Invalid step spec {spec!r}: expected keys 'step' and optionally "
                             f"'params', 'fit_only' (unknown: {sorted(unknown)}).")
        name, params, fit_only = spec["step"], dict(spec.get("params") or {}), spec.get("fit_only")
    else:
        raise TypeError(f"Cannot build a step from {type(spec).__name__}: {spec!r}")

    if name.startswith("fn:"):
        return FunctionStep(name[3:], fit_only=fit_only, **params)
    cls = registry.get_step_class(name)
    if cls is not None:
        try:
            return cls(fit_only=fit_only, **params)
        except TypeError as exc:
            raise TypeError(f"Invalid parameters for step '{name}': {exc}") from None
    if registry.resolve_function(name) is not None:
        return FunctionStep(name, fit_only=fit_only, **params)
    close = difflib.get_close_matches(name, list(registry.available_steps()), n=3)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ValueError(f"Unknown step '{name}'.{hint} See available_steps(); custom functions "
                     f"must be registered with register_function().")


# ── reports ──────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class StepReport:
    index: int
    step: str
    skipped: bool
    rows_in: int
    rows_out: int
    cols_in: int
    cols_out: int
    nulls_in: int
    nulls_out: int
    added: list
    removed: list
    seconds: float
    log: str = ""


@dataclasses.dataclass
class RunReport:
    """What a single ``fit``/``transform`` run did. Print it for a readable summary."""

    run_id: str
    pipeline: str
    mode: str
    started_at: str
    finished_at: str
    seconds: float
    config_hash: str
    versions: dict
    input: dict
    output: dict
    steps: list
    warnings: list

    def to_dict(self) -> dict:
        return _jsonable(dataclasses.asdict(self))

    def to_json(self, path=None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_frame(self) -> pl.DataFrame:
        rows = [{k: v for k, v in dataclasses.asdict(s).items() if k not in ("added", "removed", "log")}
                for s in self.steps]
        return pl.DataFrame(rows)

    def summary(self) -> str:
        i, o = self.input, self.output
        lines = [
            f"Pipeline '{self.pipeline}' · {self.mode} · {len(self.steps)} steps · {self.seconds:.2f}s"
            f" · rows {i['rows']:,}→{o['rows']:,} · cols {i['columns']}→{o['columns']}"
            f" · nulls {i['nulls']:,}→{o['nulls']:,}",
            f"{'#':>3}  {'step':<28}{'rows':>15}{'cols':>10}{'nulls':>15}{'time':>9}",
        ]
        for s in self.steps:
            if s.skipped:
                lines.append(f"{s.index:>3}  {s.step:<28}{'(skipped: fit only)':>49}")
                continue
            lines.append(f"{s.index:>3}  {s.step:<28}{f'{s.rows_in:,}→{s.rows_out:,}':>15}"
                         f"{f'{s.cols_in}→{s.cols_out}':>10}{f'{s.nulls_in:,}→{s.nulls_out:,}':>15}"
                         f"{s.seconds * 1000:>7.0f}ms")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)

    __str__ = summary

    def __repr__(self) -> str:
        return self.summary()


# ── pipeline ─────────────────────────────────────────────────────────────────

class Pipeline:
    """An ordered sequence of steps that is fitted once and replayed on any new data.

    Parameters
    ----------
    steps : list of Step instances, step names, ``(name, params)`` tuples or
        ``{"step": name, "params": {...}}`` dicts.
    name : label used in reports and saved files.
    on_schema_mismatch : ``"warn"`` (default), ``"raise"`` or ``"ignore"`` when ``transform``
        receives columns/dtypes that differ from training.
    verbose : show the steps' own printed output (otherwise it is captured in the report).
    audit_log : optional path; every run appends one JSON line with its ``RunReport``.
    target : optional label column. It is hidden from ordinary steps (so they can't impute,
        clip, encode or drop it), passed to target-aware steps, and kept aligned when rows are
        removed. It may be absent when transforming new data for prediction.
    """

    def __init__(self, steps=(), name: str = "pipeline", *, on_schema_mismatch: str = "warn",
                 verbose: bool = False, audit_log=None, target: str | None = None):
        if on_schema_mismatch not in SCHEMA_POLICIES:
            raise ValueError(f"on_schema_mismatch must be one of {SCHEMA_POLICIES}")
        self.name = name
        self.target = target
        self.steps = [make_step(s) for s in steps]
        self.on_schema_mismatch = on_schema_mismatch
        self.verbose = verbose
        self.audit_log = audit_log
        self.input_schema_: dict | None = None
        self.output_schema_: dict | None = None
        self.fitted_at_: str | None = None
        self.report_: RunReport | None = None

    # -- building -----------------------------------------------------------
    def add(self, step, **params) -> "Pipeline":
        """Append a step (``pipe.add("impute", numeric="mean")``). Resets the fitted state."""
        self.steps.append(make_step({"step": step, "params": params} if isinstance(step, str) else step))
        self.fitted_at_ = None
        return self

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __getitem__(self, i):
        return self.steps[i]

    @property
    def is_fitted(self) -> bool:
        return self.fitted_at_ is not None and all(s.is_fitted for s in self.steps)

    # -- running ------------------------------------------------------------
    def fit(self, df) -> "Pipeline":
        self.fit_transform(df)
        return self

    def fit_transform(self, df) -> pl.DataFrame:
        return self._run(df, "fit")

    def transform(self, df) -> pl.DataFrame:
        if not self.is_fitted:
            raise NotFittedError(f"Pipeline '{self.name}' is not fitted. Call fit()/fit_transform() "
                                 f"or load a fitted pipeline with Pipeline.load().")
        return self._run(df, "transform")

    def _run(self, df, mode: str) -> pl.DataFrame:
        if not isinstance(df, pl.DataFrame):
            try:
                df = pl.from_pandas(df)
            except Exception as exc:
                raise TypeError(f"Expected a polars or pandas DataFrame, got {type(df).__name__}") from exc

        if mode == "fit" and self.target and self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' is not in the training data.")
        if mode == "transform":
            df = self._restore_null_columns(df)
        run_warnings = self._check_schema(df) if mode == "transform" else []
        input_schema, input_fp = _schema(df), _fingerprint(df)
        started_at, t_start = _now(), time.perf_counter()
        target = self.target if self.target and self.target in df.columns else None
        reports = []

        for i, step in enumerate(self.steps, 1):
            rows, cols, nulls = df.height, df.columns, _nulls(df)
            if mode == "transform" and step.fit_only:
                reports.append(StepReport(i, step.label, True, rows, rows, len(cols), len(cols),
                                          nulls, nulls, [], [], 0.0))
                continue
            buf = io.StringIO()
            t0 = time.perf_counter()
            step._pipeline_target = self.target  # type: ignore[attr-defined]
            try:
                with (contextlib.nullcontext() if self.verbose else contextlib.redirect_stdout(buf)):
                    df = self._apply(step, df, mode, target)
            except PipelineStepError:
                raise
            except Exception as exc:
                raise PipelineStepError(f"Step {i} '{step.label}' failed during {mode}: {exc}") from exc
            before, after = set(cols), set(df.columns)
            reports.append(StepReport(
                i, step.label, False, rows, df.height, len(cols), df.width, nulls, _nulls(df),
                [c for c in df.columns if c not in before], [c for c in cols if c not in after],
                time.perf_counter() - t0, buf.getvalue()[-_LOG_LIMIT:]))

        if mode == "fit":
            self.input_schema_, self.output_schema_ = input_schema, _schema(df)
            self.fitted_at_ = _now()

        self.report_ = RunReport(
            run_id=uuid.uuid4().hex, pipeline=self.name, mode=mode, started_at=started_at,
            finished_at=_now(), seconds=time.perf_counter() - t_start,
            config_hash=self.config_hash(), versions=_versions(), input=input_fp,
            output=_fingerprint(df), steps=reports, warnings=run_warnings)
        if self.audit_log:
            with open(self.audit_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(self.report_.to_dict()) + "\n")
        return df

    def _restore_null_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """A column that is entirely null in a batch arrives with dtype Null; give it back the
        dtype it had in training so steps and the schema check see the expected type."""
        fixes = []
        for c, dtype in df.schema.items():
            if dtype == pl.Null and c in (self.input_schema_ or {}):
                wanted = _parse_dtype((self.input_schema_ or {})[c])
                if wanted is not None:
                    fixes.append(pl.col(c).cast(wanted))
        return df.with_columns(fixes) if fixes else df

    @staticmethod
    def _apply(step: Step, df: pl.DataFrame, mode: str, target: str | None) -> pl.DataFrame:
        """Run one step, hiding ``target`` from steps that don't use it."""
        fitting = mode == "fit"
        if target is None or step.uses_target:
            return step.fit_transform(df) if fitting else step.transform(df)
        y, X = df.get_column(target), df.drop(target)
        if fitting:
            step.fit(X)
        elif step.state_ is None:
            raise NotFittedError(f"Step '{step.label}' must be fitted before transform().")
        mask = step.row_mask(X)
        if mask is not None:                       # row filter: drop the same rows from y
            return X.filter(mask).with_columns(y.filter(mask))
        out = step.transform(X)
        if out.width == 0:      # every feature dropped; Polars 1.x reports height 0 for such frames
            return y.to_frame()
        if out.height != X.height:
            raise PipelineStepError(
                f"Step '{step.label}' changed the number of rows, so the target '{target}' can't "
                f"stay aligned. Steps that remove rows must implement row_mask().")
        return out.with_columns(y)

    def _check_schema(self, df: pl.DataFrame) -> list:
        expected, actual = dict(self.input_schema_ or {}), _schema(df)
        if self.target and self.target not in actual:
            expected.pop(self.target, None)        # predicting: the label is legitimately absent
        problems = []
        missing = [c for c in expected if c not in actual]
        extra = [c for c in actual if c not in expected]
        changed = [f"{c} ({expected[c]} → {actual[c]})" for c in expected
                   if c in actual and expected[c] != actual[c]]
        if missing:
            problems.append(f"missing columns: {missing}")
        if extra:
            problems.append(f"unexpected columns: {extra}")
        if changed:
            problems.append(f"dtype changes: {changed}")
        if not problems or self.on_schema_mismatch == "ignore":
            return problems if self.on_schema_mismatch != "ignore" else []
        message = f"Input schema differs from training for pipeline '{self.name}': " + "; ".join(problems)
        if self.on_schema_mismatch == "raise":
            raise SchemaMismatchError(message)
        with warnings.catch_warnings():   # show even if the host silenced warnings
            warnings.simplefilter("always", PipelineWarning)
            warnings.warn(message, PipelineWarning, stacklevel=4)
        return problems

    # -- config & persistence ----------------------------------------------
    def to_config(self) -> dict:
        """The pipeline's recipe (steps + params), without learned state."""
        cfg: dict[str, Any] = {"name": self.name, "on_schema_mismatch": self.on_schema_mismatch}
        if self.target:
            cfg["target"] = self.target
        cfg["steps"] = [s.to_dict(include_state=False) for s in self.steps]
        return _jsonable(cfg)

    def config_hash(self) -> str:
        blob = json.dumps(self.to_config(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @classmethod
    def from_config(cls, config, **settings) -> "Pipeline":
        """Build an unfitted pipeline from a dict or a ``.json`` / ``.yaml`` file."""
        if not isinstance(config, dict):
            config = _read_config(config)
        opts = {"name": config.get("name", "pipeline"),
                "on_schema_mismatch": config.get("on_schema_mismatch", "warn"),
                "target": config.get("target")}
        opts.update(settings)
        return cls(config.get("steps", []), **opts)

    def save_config(self, path) -> Path:
        """Write the recipe (no learned state) as JSON, or YAML for ``.yaml``/``.yml`` paths."""
        path = Path(path)
        cfg = self.to_config()
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        else:
            path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return path

    def save(self, path) -> Path:
        """Save the pipeline, including everything it learned, as JSON."""
        payload = {
            "format": FORMAT, "format_version": FORMAT_VERSION, **self.to_config(),
            "steps": [s.to_dict(include_state=True) for s in self.steps],
            "input_schema": self.input_schema_, "output_schema": self.output_schema_,
            "fitted_at": self.fitted_at_, "config_hash": self.config_hash(), "versions": _versions(),
        }
        path = Path(path)
        path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path, **settings) -> "Pipeline":
        """Load a pipeline saved with :meth:`save`, ready to ``transform`` new data."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f"{path} is not a saved pipeline. Use Pipeline.from_config() for config files.")
        if data.get("format_version", 0) > FORMAT_VERSION:
            raise ValueError(f"{path} was saved by a newer version (format {data['format_version']}); "
                             f"upgrade mlpilot to load it.")
        pipe = cls.from_config(data, **settings)
        for step, saved in zip(pipe.steps, data["steps"]):
            step.state_ = saved.get("state")
        pipe.input_schema_ = data.get("input_schema")
        pipe.output_schema_ = data.get("output_schema")
        pipe.fitted_at_ = data.get("fitted_at")
        return pipe

    def __repr__(self) -> str:
        target = f", target={self.target!r}" if self.target else ""
        lines = [f"Pipeline({self.name!r}{target}, fitted={self.is_fitted}, {len(self.steps)} steps)"]
        for i, s in enumerate(self.steps, 1):
            lines.append(f"  {i:>2}. {s.label}{'  [fit only]' if s.fit_only else ''}")
        return "\n".join(lines)
