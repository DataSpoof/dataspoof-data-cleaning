"""Model registry: versioned models moving through dev → qa → staging → production.

Layout::

    <root>/<model name>/v<N>/model/        the saved model (checksummed by AutoML.save)
    <root>/<model name>/v<N>/version.json  metrics, run, dataset, stage, who/when
    <root>/<model name>/history.jsonl      every stage change, with who, when and why
"""

from __future__ import annotations

import dataclasses
import operator
import shutil
from pathlib import Path

import polars as pl

from ._store import (FileLock, append_jsonl, check_name, dataset_fingerprint, now, read_json, read_jsonl, user,
                     write_json)

STAGES = ("none", "dev", "qa", "staging", "production", "archived")
_OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq}


class PromotionError(Exception):
    """A model version failed the checks required to move to a stage."""


@dataclasses.dataclass
class ModelVersion:
    name: str
    version: int
    stage: str
    path: Path
    created_at: str
    created_by: str
    metrics: dict
    tags: dict
    description: str | None
    run_id: str | None
    dataset: dict | None
    task: str | None
    target: str | None
    model_sha256: str | None

    @property
    def model_path(self) -> Path:
        return self.path / "model"

    def load(self):
        from ..automl import load_model
        return load_model(self.model_path)

    def __repr__(self) -> str:
        return f"ModelVersion({self.name} v{self.version}, {self.stage})"


def _version_from(path: Path) -> ModelVersion:
    d = read_json(path / "version.json")
    fields = {f.name for f in dataclasses.fields(ModelVersion)} - {"path"}
    return ModelVersion(path=path, **{k: d.get(k) for k in fields})


def _parse_requirement(spec):
    """``0.8`` → (>=, 0.8); ``("<=", 12.5)`` or ``"<= 12.5"`` → (<=, 12.5)."""
    if isinstance(spec, (int, float)):
        return ">=", float(spec)
    if isinstance(spec, str):
        op, _, value = spec.strip().partition(" ")
        return op, float(value)
    op, value = spec
    return op, float(value)


class ModelRegistry:
    """A model registry in a directory (default ``./ac_registry``).

    >>> reg = ModelRegistry("ac_registry")
    >>> v = reg.register(model, "churn", run=run)                  # -> v1 in "dev"
    >>> reg.promote("churn", v.version, "staging")
    >>> reg.promote("churn", v.version, "production", require={"test_roc_auc": 0.85})
    >>> model = reg.load("churn")                                   # current production model
    """

    def __init__(self, root="ac_registry"):
        self.root = Path(root)

    def _dir(self, name: str) -> Path:
        return self.root / check_name(name, "model name")

    def _lock(self, name: str) -> FileLock:
        return FileLock(self._dir(name) / ".lock")

    # ── registering ─────────────────────────────────────────────────────────
    def register(self, model, name: str, *, run=None, stage: str = "dev", metrics: dict | None = None,
                 tags: dict | None = None, description: str | None = None, dataset=None,
                 sign_with=None, sign_password: str | None = None) -> ModelVersion:
        """Add a new version of model ``name``.

        ``model`` is a fitted AutoML / AutoCluster object or a saved model directory.
        Metrics come from ``run`` (a tracked Run) and/or ``metrics``; an AutoML model also
        contributes its cross-validation score. ``dataset`` (a DataFrame) records the training
        data fingerprint (the run's ``train`` dataset is used otherwise). ``sign_with`` (an
        Ed25519 private key, see ``automatedcleaning.security``) signs the stored model so
        deployments can verify it with ``AC_MODEL_PUBLIC_KEY``."""
        if stage not in STAGES or stage == "archived":
            raise ValueError(f"stage must be one of {STAGES[:-1]}")
        base = self._dir(name)
        with self._lock(name):
            existing = [int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit()]
            version = max(existing, default=0) + 1
            vdir = base / f"v{version}"
            vdir.mkdir(parents=True)
            try:
                if isinstance(model, (str, Path)):
                    shutil.copytree(model, vdir / "model")
                else:
                    model.save(vdir / "model")
            except Exception:
                shutil.rmtree(vdir, ignore_errors=True)
                raise
        if sign_with is not None:
            from ..security import sign_model
            sign_model(vdir / "model", sign_with, password=sign_password, signer=user())
        manifest = read_json(vdir / "model" / "manifest.json")
        all_metrics = {}
        if manifest.get("kind") == "supervised":
            from ..automl.automl import _metric_label, _shown
            all_metrics[f"cv_{_metric_label(manifest['metric'])}"] = _shown(manifest["metric"],
                                                                           manifest["best"]["score"])
        ds = None
        if run is not None:
            all_metrics.update(run.metrics)
            ds = next((d for d in run.datasets if d.get("role") == "train" or d.get("name") == "train"), None)
        all_metrics.update(metrics or {})
        if dataset is not None:
            ds = dataset_fingerprint(dataset, "train")
        task = manifest["task"]["task"] if manifest.get("kind") == "supervised" else manifest.get("kind")
        record = {"name": name, "version": version, "stage": stage, "created_at": now(), "created_by": user(),
                  "metrics": all_metrics, "tags": dict(tags or {}), "description": description,
                  "run_id": run.id if run is not None else None, "dataset": ds, "task": task,
                  "target": manifest.get("target"), "model_sha256": manifest.get("model_sha256")}
        write_json(vdir / "version.json", record)
        append_jsonl(base / "history.jsonl", {"time": now(), "user": user(), "version": version,
                                              "from": None, "to": stage, "reason": "registered"})
        return _version_from(vdir)

    # ── reading ─────────────────────────────────────────────────────────────
    def models(self) -> pl.DataFrame:
        """Every registered model: latest version and which version is in each stage."""
        rows = []
        if self.root.exists():
            for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
                vs = self._versions(d.name)
                if not vs:
                    continue
                stage_of = lambda s: next((v.version for v in reversed(vs) if v.stage == s), None)  # noqa: E731
                rows.append({"name": d.name, "versions": len(vs), "latest": vs[-1].version,
                             "production": stage_of("production"), "staging": stage_of("staging"),
                             "qa": stage_of("qa"),
                             "dev": stage_of("dev"), "task": vs[-1].task, "target": vs[-1].target})
        schema = {"name": pl.String, "versions": pl.Int64, "latest": pl.Int64, "production": pl.Int64,
                  "staging": pl.Int64, "qa": pl.Int64, "dev": pl.Int64, "task": pl.String, "target": pl.String}
        return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)

    def _versions(self, name: str) -> list:
        base = self._dir(name)
        dirs = sorted((p for p in base.glob("v*") if p.name[1:].isdigit() and (p / "version.json").exists()),
                      key=lambda p: int(p.name[1:]))
        return [_version_from(p) for p in dirs]

    def versions(self, name: str) -> pl.DataFrame:
        """All versions of a model with their stage and metrics (``metric.*`` columns)."""
        vs = self._versions(name)
        if not vs:
            raise KeyError(f"No model named {name!r}")
        rows = [{"version": v.version, "stage": v.stage, "created_at": v.created_at, "created_by": v.created_by,
                 "run_id": v.run_id, "description": v.description,
                 **{f"metric.{k}": val for k, val in v.metrics.items()}} for v in vs]
        cols = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        return pl.DataFrame([{c: r.get(c) for c in cols} for r in rows], infer_schema_length=None)

    def get(self, name: str, version: int | None = None, stage: str | None = None) -> ModelVersion:
        """A specific version, the newest version in ``stage``, or (neither given) the production
        version if there is one, else the newest version."""
        vs = self._versions(name)
        if not vs:
            raise KeyError(f"No model named {name!r}")
        if version is not None:
            for v in vs:
                if v.version == int(version):
                    return v
            raise KeyError(f"{name} has no version {version}")
        if stage is not None:
            for v in reversed(vs):
                if v.stage == stage:
                    return v
            raise KeyError(f"{name} has no version in stage {stage!r}")
        prod = [v for v in vs if v.stage == "production"]
        return prod[-1] if prod else vs[-1]

    def load(self, name: str, version: int | None = None, stage: str | None = None):
        """Load a model (default: the production version). The checksum is verified on load."""
        return self.get(name, version, stage).load()

    def history(self, name: str) -> pl.DataFrame:
        rows = read_jsonl(self._dir(name) / "history.jsonl")
        schema = {"time": pl.String, "user": pl.String, "version": pl.Int64, "from": pl.String,
                  "to": pl.String, "reason": pl.String}
        return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)

    # ── promotion ───────────────────────────────────────────────────────────
    def check(self, name: str, version: int, require: dict) -> list:
        """Failed requirements for a version, e.g. ``{"test_roc_auc": 0.85, "test_rmse": ("<=", 12)}``."""
        v = self.get(name, version)
        failures = []
        for metric, spec in (require or {}).items():
            op, limit = _parse_requirement(spec)
            if op not in _OPS:
                raise ValueError(f"Unknown comparison {op!r}; use one of {list(_OPS)}")
            value = v.metrics.get(metric)
            if value is None:
                failures.append(f"{metric} was not recorded")
            elif not _OPS[op](value, limit):
                failures.append(f"{metric} = {value:.4g} (needs {op} {limit:g})")
        return failures

    def promote(self, name: str, version: int, stage: str, *, reason: str | None = None,
                require: dict | None = None, archive_existing: bool = True,
                must_beat_production: str | None = None) -> ModelVersion:
        """Move a version to ``stage``. Moving to production archives the current production
        version (``archive_existing``). ``require`` sets metric thresholds, and
        ``must_beat_production="test_roc_auc"`` requires the candidate to be at least as good as
        the current production model on that metric (use a ``-`` prefix, e.g. ``"-test_rmse"``,
        for lower-is-better). Failing checks raise ``PromotionError`` and nothing changes."""
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        with self._lock(name):
            v = self.get(name, version)
            failures = self.check(name, version, require or {})
            if must_beat_production and stage == "production":
                metric = must_beat_production.lstrip("-")
                lower = must_beat_production.startswith("-")
                current = [p for p in self._versions(name) if p.stage == "production" and p.version != v.version]
                if current and metric in current[-1].metrics:
                    mine, theirs = v.metrics.get(metric), current[-1].metrics[metric]
                    if mine is None:
                        failures.append(f"{metric} was not recorded")
                    elif (mine > theirs) if lower else (mine < theirs):
                        failures.append(f"{metric} = {mine:.4g} is worse than production v{current[-1].version} "
                                        f"({theirs:.4g})")
            if failures:
                raise PromotionError(f"{name} v{version} can't move to {stage}: " + "; ".join(failures))
            changes = [(v, stage)]
            if stage == "production" and archive_existing:
                changes += [(p, "archived") for p in self._versions(name)
                            if p.stage == "production" and p.version != v.version]
            for target, new_stage in changes:
                record = read_json(target.path / "version.json")
                old = record["stage"]
                if old == new_stage:
                    continue
                record["stage"] = new_stage
                write_json(target.path / "version.json", record)
                append_jsonl(self._dir(name) / "history.jsonl",
                             {"time": now(), "user": user(), "version": target.version, "from": old,
                              "to": new_stage, "reason": reason if target is v else
                              f"replaced by v{v.version}"})
        return self.get(name, version)

    def rollback(self, name: str, reason: str = "rollback") -> ModelVersion:
        """Put the previous production version back into production."""
        hist = self.history(name)
        prev = hist.filter((pl.col("from") == "production") & (pl.col("to") == "archived"))
        if prev.height == 0:
            raise KeyError(f"{name} has no earlier production version to roll back to")
        return self.promote(name, int(prev["version"][-1]), "production", reason=reason)
