"""Experiment tracking: every training run's parameters, metrics, datasets and artifacts.

Layout (plain files, easy to back up, diff or sync to object storage)::

    <root>/<experiment>/<run_id>/run.json        params, tags, datasets, status, git, timings
    <root>/<experiment>/<run_id>/metrics.jsonl   one line per logged metric value (with step)
    <root>/<experiment>/<run_id>/artifacts/      files, saved models, model cards
"""

from __future__ import annotations

from typing import Any

import shutil
import time
import uuid
from pathlib import Path

import polars as pl

from ._store import (append_jsonl, check_name, dataset_fingerprint, git_info, host, now, read_json,
                     read_jsonl, user, write_json)

RUN_FILE, METRICS_FILE, ARTIFACTS = "run.json", "metrics.jsonl", "artifacts"


def _scalar(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


class Run:
    """One tracked run. Use as a context manager: it ends as ``finished`` or, if an
    exception escapes, as ``failed`` with the error recorded."""

    def __init__(self, path: Path, meta: dict):
        self.path, self._meta = Path(path), meta

    # ── identity ────────────────────────────────────────────────────────────
    @property
    def id(self) -> str:
        return self._meta["run_id"]

    @property
    def experiment(self) -> str:
        return self._meta["experiment"]

    @property
    def name(self) -> str:
        return self._meta["name"]

    @property
    def status(self) -> str:
        return self._meta["status"]

    @property
    def params(self) -> dict:
        return dict(self._meta["params"])

    @property
    def tags(self) -> dict:
        return dict(self._meta["tags"])

    @property
    def datasets(self) -> list:
        return list(self._meta["datasets"])

    @property
    def artifacts_dir(self) -> Path:
        return self.path / ARTIFACTS

    @property
    def metrics(self) -> dict:
        """Latest value of every metric."""
        latest = {}
        for rec in read_jsonl(self.path / METRICS_FILE):
            latest[rec["key"]] = rec["value"]
        return latest

    def metric_history(self, key: str | None = None) -> pl.DataFrame:
        rows = [r for r in read_jsonl(self.path / METRICS_FILE) if key is None or r["key"] == key]
        schema = {"key": pl.String, "value": pl.Float64, "step": pl.Int64, "time": pl.String}
        return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)

    def to_dict(self) -> dict:
        return {**self._meta, "metrics": self.metrics}

    # ── logging ─────────────────────────────────────────────────────────────
    def _save(self):
        write_json(self.path / RUN_FILE, self._meta)

    def _check_open(self):
        if self._meta["status"] != "running":
            raise RuntimeError(f"Run {self.id} has ended ({self.status}); start a new run to log more.")

    def log_param(self, key: str, value) -> "Run":
        return self.log_params({key: value})

    def log_params(self, params: dict) -> "Run":
        self._check_open()
        self._meta["params"].update({str(k): _scalar(v) for k, v in params.items()})
        self._save()
        return self

    def log_metric(self, key: str, value: float, step: int | None = None) -> "Run":
        return self.log_metrics({key: value}, step=step)

    def log_metrics(self, metrics: dict, step: int | None = None) -> "Run":
        self._check_open()
        for k, v in metrics.items():
            if v is None:
                continue
            append_jsonl(self.path / METRICS_FILE, {"key": str(k), "value": float(v), "step": step, "time": now()})
        return self

    def set_tag(self, key: str, value) -> "Run":
        self._meta["tags"][str(key)] = _scalar(value)
        self._save()
        return self

    def log_dataset(self, df, name: str = "train", role: str | None = None) -> dict:
        """Record which data the run used (shape, schema and content hash, not the data)."""
        self._check_open()
        fp = {**dataset_fingerprint(df, name), "role": role or name}
        self._meta["datasets"] = [d for d in self._meta["datasets"] if d["name"] != name] + [fp]
        self._save()
        return fp

    def log_artifact(self, source, name: str | None = None) -> Path:
        """Copy a file or directory into the run's artifacts."""
        self._check_open()
        source = Path(source)
        dest = self.artifacts_dir / (name or source.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(source, dest)
        self._meta["artifacts"] = sorted(set(self._meta["artifacts"]) | {dest.name})
        self._save()
        return dest

    def log_text(self, text: str, name: str) -> Path:
        self._check_open()
        check_name(name, "artifact name")
        dest = self.artifacts_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        self._meta["artifacts"] = sorted(set(self._meta["artifacts"]) | {name})
        self._save()
        return dest

    def log_model(self, model, name: str = "model") -> Path:
        """Save a fitted AutoML / AutoCluster model into the run's artifacts."""
        self._check_open()
        check_name(name, "artifact name")
        path = model.save(self.artifacts_dir / name)
        self._meta["artifacts"] = sorted(set(self._meta["artifacts"]) | {name})
        self._meta["model_artifact"] = name
        self._save()
        return Path(path)

    def log_automl(self, model, train_df=None, test_df=None, *, card: bool = True, **card_kw) -> "Run":
        """Log everything about a fitted AutoML model in one call: settings, the CV score and
        leaderboard, datasets, held-out metrics (with ``test_df``), the model and a model card."""
        from ..automl.automl import _metric_label, _shown
        label = _metric_label(model.metric_)
        self.log_params({"target": model.target, "task": model.task_.task, "metric": label,
                         "algorithm": model.best_["model"], "cv": model.cv_description_,
                         "time_budget": model.time_budget, "n_trials": len(model.trials_),
                         "n_features": model.n_features_, "train_rows": model.n_rows_,
                         "dropped_leaky": ",".join(model.dropped_leaky_) or None,
                         **{f"hp.{k}": v for k, v in (model.best_["params"] or {}).items()
                            if not isinstance(v, (list, dict))}})
        self.log_metrics({f"cv_{label}": _shown(model.metric_, model.best_["score"]),
                          f"cv_{label}_std": model.best_["std"], "fit_seconds": model.fit_seconds_})
        if train_df is not None:
            self.log_dataset(train_df, "train")
        self.log_text(model.leaderboard_.write_csv(), "leaderboard.csv")
        if test_df is not None:
            from ..evaluation import evaluate_model
            self.log_dataset(test_df, "test")
            report = evaluate_model(model, test_df)
            self.log_metrics({f"test_{k}": v for k, v in report.metrics.items() if v is not None})
            self.log_text(report.to_json(), "evaluation.json")
        self.log_model(model)
        if card:
            from ..evaluation import model_card
            c = model_card(model, test_df, **card_kw)
            self.log_text(c.to_html(), "model_card.html")
            self.log_text(c.to_markdown(), "model_card.md")
        return self

    def end(self, status: str = "finished", error: str | None = None) -> "Run":
        if self._meta["status"] != "running":
            return self
        self._meta.update(status=status, ended_at=now(),
                          duration_seconds=round(time.time() - self._meta["_t0"], 3), error=error)
        self._save()
        return self

    def load_model(self, name: str | None = None):
        from ..automl import load_model
        return load_model(self.artifacts_dir / (name or self._meta.get("model_artifact") or "model"))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.end("finished")
        else:
            self.end("failed", error=f"{exc_type.__name__}: {exc}"[:1000])
        return False

    def __repr__(self) -> str:
        return f"Run({self.experiment}/{self.name} {self.id[:8]}, {self.status})"


class Tracker:
    """Tracks experiment runs in a directory (default ``./ac_runs``).

    >>> tracker = Tracker("ac_runs", experiment="churn")
    >>> with tracker.start_run("lgbm-v2") as run:
    ...     run.log_automl(model, train, test)
    >>> tracker.runs()          # every run as a table: params, latest metrics, status
    """

    def __init__(self, root="ac_runs", experiment: str = "default"):
        self.root = Path(root)
        self.experiment = check_name(experiment, "experiment name")

    def start_run(self, name: str | None = None, *, experiment: str | None = None, tags: dict | None = None,
                  params: dict | None = None) -> Run:
        exp = check_name(experiment or self.experiment, "experiment name")
        run_id = uuid.uuid4().hex
        meta: dict[str, Any] = {"run_id": run_id, "experiment": exp, "name": name or f"run-{run_id[:8]}", "status": "running",
                "started_at": now(), "ended_at": None, "duration_seconds": None, "error": None,
                "user": user(), "host": host(), "git": git_info(), "params": {}, "tags": dict(tags or {}),
                "datasets": [], "artifacts": [], "_t0": time.time()}
        run = Run(self.root / exp / run_id, meta)
        run._save()
        if params:
            run.log_params(params)
        return run

    def _run_dirs(self, experiment: str | None = None):
        exps = [experiment] if experiment else ([p.name for p in self.root.iterdir() if p.is_dir()]
                                                if self.root.exists() else [])
        for exp in exps:
            base = self.root / exp
            if base.exists():
                yield from (p for p in base.iterdir() if (p / RUN_FILE).exists())

    def get_run(self, run_id: str) -> Run:
        for d in self._run_dirs():
            if d.name == run_id or d.name.startswith(run_id):
                return Run(d, read_json(d / RUN_FILE))
        raise KeyError(f"No run {run_id!r}")

    def experiments(self) -> list:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir()) if self.root.exists() else []

    def runs(self, experiment: str | None = None, status: str | None = None) -> pl.DataFrame:
        """All runs (newest first): id, name, status, timings, ``param.*`` and latest ``metric.*``."""
        rows = []
        for d in self._run_dirs(experiment):
            run = Run(d, read_json(d / RUN_FILE))
            if status and run.status != status:
                continue
            m = run._meta
            row = {"run_id": run.id, "experiment": run.experiment, "name": run.name, "status": run.status,
                   "started_at": m["started_at"], "duration_seconds": m["duration_seconds"], "user": m["user"]}
            row.update({f"param.{k}": (str(v) if v is not None else None) for k, v in m["params"].items()})
            row.update({f"metric.{k}": v for k, v in run.metrics.items()})
            rows.append(row)
        if not rows:
            return pl.DataFrame(schema={"run_id": pl.String, "experiment": pl.String, "name": pl.String,
                                        "status": pl.String, "started_at": pl.String})
        cols = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        frame = pl.DataFrame([{c: r.get(c) for c in cols} for r in rows], infer_schema_length=None)
        return frame.sort("started_at", descending=True)

    def best_run(self, metric: str, experiment: str | None = None, mode: str = "max") -> Run:
        """The finished run with the best latest value of ``metric`` (``mode`` max or min)."""
        frame = self.runs(experiment, status="finished")
        col = f"metric.{metric}"
        if col not in frame.columns or frame[col].drop_nulls().len() == 0:
            raise KeyError(f"No finished run logged '{metric}'")
        best = frame.filter(pl.col(col).is_not_null()).sort(col, descending=(mode == "max"))["run_id"][0]
        return self.get_run(best)

    def compare(self, run_ids: list) -> pl.DataFrame:
        """Params and metrics of several runs side by side (one column per run)."""
        runs = [self.get_run(r) for r in run_ids]
        keys = []
        for r in runs:
            keys += [f"param.{k}" for k in r.params if f"param.{k}" not in keys]
        for r in runs:
            keys += [f"metric.{k}" for k in r.metrics if f"metric.{k}" not in keys]
        data: dict[str, list] = {"field": keys}
        for r in runs:
            vals = {**{f"param.{k}": v for k, v in r.params.items()}, **{f"metric.{k}": v for k, v in r.metrics.items()}}
            data[f"{r.name} ({r.id[:6]})"] = [None if vals.get(k) is None else str(vals.get(k)) for k in keys]
        return pl.DataFrame(data)
