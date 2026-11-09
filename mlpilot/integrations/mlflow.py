"""MLflow integration (needs ``mlflow``, the ``[mlflow]`` extra).

Keep using MLflow's UI, model registry and serving if your company runs MLflow:

>>> from mlpilot.integrations import mlflow as acm
>>> acm.export_run(run, tracking_uri="http://mlflow:5000")          # params, metric history, artifacts
>>> acm.export_tracker(tracker)                                     # every run (re-running skips copies)
>>> info = acm.log_automl(model, test_df, experiment="churn", register_as="churn")
>>> acm.export_registry(registry, "churn")                          # versions + stages -> aliases
>>> model = acm.load_automl("models:/churn@production")             # back to an AutoML object

Models are logged as MLflow **pyfunc** models with a signature and an input example, so all of
MLflow works on them::

    mlflow.pyfunc.load_model("models:/churn/1").predict(pandas_df)
    mlflow models serve -m models:/churn@production --env-manager local

The pyfunc output is a DataFrame: ``<target>_pred`` plus one ``proba_<class>`` column per class
for classifiers. The saved AutoML directory is stored inside the MLflow model (its checksum is
verified again when it is loaded).
"""

from __future__ import annotations

import datetime as _dt
import re
import tempfile
from pathlib import Path

try:
    import mlflow
    import mlflow.pyfunc
    from mlflow.entities import Metric, Param, RunTag
    from mlflow.tracking import MlflowClient
except ImportError as exc:                                   # pragma: no cover
    raise ImportError('The MLflow integration needs mlflow: pip install "dataspoof-mlpilot[mlflow]"') from exc

SOURCE_TAG = "mlpilot.run_id"
_STATUS = {"finished": "FINISHED", "failed": "FAILED", "running": "RUNNING", "killed": "KILLED"}


# ── helpers ─────────────────────────────────────────────────────────────────
def _client(tracking_uri: str | None) -> MlflowClient:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def _experiment_id(client: MlflowClient, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    return exp.experiment_id if exp is not None else client.create_experiment(name)


def _key(k) -> str:
    """MLflow keys allow letters, digits, ``_ - . / space`` (and ``:``)."""
    return re.sub(r"[^\w\-./ :]", "_", str(k))[:250] or "_"


def _ms(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        t = _dt.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return int(t.timestamp() * 1000)


def _batched(client: MlflowClient, run_id: str, *, metrics=(), params=(), tags=()):
    """``log_batch`` within MLflow's per-request limits (1000 items, 100 params, 100 tags)."""
    metrics, params, tags = list(metrics), list(params), list(tags)
    for i in range(0, len(params), 100):
        client.log_batch(run_id, params=params[i:i + 100])
    for i in range(0, len(tags), 100):
        client.log_batch(run_id, tags=tags[i:i + 100])
    for i in range(0, len(metrics), 1000):
        client.log_batch(run_id, metrics=metrics[i:i + 1000])


def find_exported(run_id: str, *, tracking_uri: str | None = None, experiment: str | None = None) -> str | None:
    """The MLflow run id a tracked run was already exported to, or ``None``."""
    client = _client(tracking_uri)
    exps = [client.get_experiment_by_name(experiment)] if experiment else client.search_experiments()
    ids = [e.experiment_id for e in exps if e is not None]
    if not ids:
        return None
    found = client.search_runs(ids, filter_string=f"tags.`{SOURCE_TAG}` = '{run_id}'", max_results=1)
    return found[0].info.run_id if found else None


# ── runs ────────────────────────────────────────────────────────────────────
def export_run(run, *, tracking_uri: str | None = None, experiment: str | None = None,
               skip_existing: bool = True) -> str:
    """Copy a tracked :class:`~mlpilot.registry.Run` into MLflow: params, every metric
    value with its step and time, tags, dataset fingerprints, start/end time, status and all
    artifacts. Returns the MLflow run id.

    With ``skip_existing`` (default), a run that was already exported (matched by the
    ``mlpilot.run_id`` tag) is not copied again, so syncing is safe to repeat."""
    client = _client(tracking_uri)
    exp_name = experiment or run.experiment
    if skip_existing:
        prev = find_exported(run.id, tracking_uri=tracking_uri, experiment=exp_name)
        if prev:
            return prev
    meta = run.to_dict()
    tags = {**{_key(k): str(v) for k, v in run.tags.items()}, SOURCE_TAG: run.id,
            "mlflow.user": str(meta.get("user") or ""), "mlflow.source.name": "mlpilot"}
    git = meta.get("git") or {}
    if git.get("commit"):
        tags["mlflow.source.git.commit"] = str(git["commit"])
    for d in run.datasets:
        tags[_key(f"dataset.{d['name']}")] = f"{d.get('rows')}x{d.get('columns')} {d.get('content_hash')}"
    created = client.create_run(_experiment_id(client, exp_name), start_time=_ms(meta.get("started_at")),
                                run_name=run.name, tags={k: v[:5000] for k, v in tags.items()})
    rid = created.info.run_id
    try:
        default_ts = _ms(meta.get("started_at")) or int(_dt.datetime.now().timestamp() * 1000)
        metrics = [Metric(_key(r["key"]), float(r["value"]), _ms(r["time"]) or default_ts, int(r["step"] or 0))
                   for r in run.metric_history().iter_rows(named=True)]
        params = [Param(_key(k), ("" if v is None else str(v))[:6000]) for k, v in run.params.items()]
        _batched(client, rid, metrics=metrics, params=params)
        if run.artifacts_dir.exists() and any(run.artifacts_dir.iterdir()):
            client.log_artifacts(rid, str(run.artifacts_dir))
        if meta.get("error"):
            client.set_tag(rid, "error", str(meta["error"])[:5000])
        status = _STATUS.get(run.status, "FINISHED")
        if status != "RUNNING":
            client.set_terminated(rid, status=status, end_time=_ms(meta.get("ended_at")))
    except Exception:
        client.set_terminated(rid, status="FAILED")
        raise
    return rid


def export_tracker(tracker, *, experiment: str | None = None, tracking_uri: str | None = None,
                   mlflow_experiment: str | None = None) -> dict:
    """Export every run of a :class:`~mlpilot.registry.Tracker` (or one experiment).
    Returns ``{our run id: mlflow run id}``; runs already exported are skipped."""
    out = {}
    runs = tracker.runs(experiment)
    for rid in (runs["run_id"].to_list() if runs.height else []):
        r = tracker.get_run(rid)
        out[r.id] = export_run(r, tracking_uri=tracking_uri, experiment=mlflow_experiment)
    return out


# ── models ──────────────────────────────────────────────────────────────────
class AutoMLPyfunc(mlflow.pyfunc.PythonModel):
    """Wraps a saved AutoML / AutoCluster / Forecaster directory as an MLflow pyfunc model."""

    def load_context(self, context):
        from ..automl import load_model
        self.model = load_model(context.artifacts["automl"])

    def predict(self, context, model_input, params=None):
        import pandas as pd
        import polars as pl
        df = model_input if isinstance(model_input, pl.DataFrame) else pl.from_pandas(pd.DataFrame(model_input))
        if hasattr(self.model, "predict_with_proba"):
            labels, proba = self.model.predict_with_proba(df)
            out = pl.DataFrame([labels]) if proba is None else pl.concat([pl.DataFrame([labels]), proba],
                                                                           how="horizontal")
        else:
            out = pl.DataFrame([self.model.predict(df)])
        return out.to_pandas()


def _requirements() -> list:
    from importlib import metadata
    reqs = []
    for dist, name in (("dataspoof-mlpilot", "dataspoof-mlpilot"), ("polars", "polars"),
                       ("scikit-learn", "scikit-learn"), ("pandas", "pandas"), ("pyarrow", "pyarrow"),
                       ("mlflow", "mlflow")):
        try:
            reqs.append(f"{name}=={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            continue
    return reqs


def _log_model_dir(model_dir: Path, example=None, *, register_as: str | None = None, name: str = "model"):
    from mlflow.models import infer_signature
    signature = None
    if example is not None:
        out = AutoMLPyfunc()
        out.model = _load_dir(model_dir)
        signature = infer_signature(example, out.predict(None, example))
    return mlflow.pyfunc.log_model(name=name, python_model=AutoMLPyfunc(), artifacts={"automl": str(model_dir)},
                                   signature=signature, input_example=example, pip_requirements=_requirements(),
                                   registered_model_name=register_as)


def _load_dir(path):
    from ..automl import load_model
    return load_model(path)


def _example(model, df, rows: int = 5):
    if df is None:
        return None
    import polars as pl
    df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    target = getattr(model, "target", None)
    if target and target in df.columns:
        df = df.drop(target)
    return df.head(rows).to_pandas()


def log_automl(model, test_df=None, *, tracking_uri: str | None = None, experiment: str | None = None,
               register_as: str | None = None, run_name: str | None = None, tags: dict | None = None):
    """Log a fitted AutoML model to MLflow as a pyfunc model: settings, the CV score, test
    metrics (when ``test_df`` is given; its first rows also become the input example and
    signature), and the model itself. ``register_as`` also creates a registered model version.

    Returns MLflow's ``ModelInfo`` (``.run_id``, ``.model_uri``, ``.registered_model_version``)."""
    from ..automl.automl import _metric_label, _shown
    _client(tracking_uri)
    if experiment:
        mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags):
        label = _metric_label(model.metric_)
        mlflow.log_params({"target": model.target, "task": model.task_.task, "algorithm": model.best_["model"],
                           "metric": label, "cv": model.cv_description_})
        mlflow.log_metric(f"cv_{_key(label)}", _shown(model.metric_, model.best_["score"]))
        if test_df is not None:
            from ..evaluation import evaluate_model
            mlflow.log_metrics({f"test_{_key(k)}": float(v) for k, v in
                                evaluate_model(model, test_df, bootstrap=0).metrics.items()
                                if isinstance(v, (int, float)) and v == v})
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp) / "automl")
            return _log_model_dir(Path(tmp) / "automl", _example(model, test_df), register_as=register_as)


def load_automl(model_uri: str, *, tracking_uri: str | None = None):
    """Load a model logged by :func:`log_automl` / :func:`export_registry` back as the original
    AutoML object (``runs:/<id>/model``, ``models:/<name>/<version>``, ``models:/<name>@<alias>``)."""
    _client(tracking_uri)
    return mlflow.pyfunc.load_model(model_uri).unwrap_python_model().model


def export_registry(registry, name: str, *, register_as: str | None = None, tracking_uri: str | None = None,
                    experiment: str | None = None, example=None) -> dict:
    """Copy every version of model ``name`` from a :class:`~mlpilot.registry.ModelRegistry`
    into the MLflow model registry. Each version keeps its metrics and description, the stage
    becomes an MLflow **alias** (``production``, ``staging``, ``dev``) and a ``stage`` tag.
    Versions already exported are skipped. Returns ``{our version: mlflow version}``."""
    client = _client(tracking_uri)
    target = register_as or name
    mlflow.set_experiment(experiment or f"registry/{name}")
    try:
        client.get_registered_model(target)
    except Exception:
        client.create_registered_model(target, description=f"Exported from mlpilot registry '{name}'")
    done = {}
    for mv in client.search_model_versions(f"name='{target}'"):
        src = (mv.tags or {}).get("mlpilot.version")
        if src:
            done[int(src)] = int(mv.version)
    versions = registry.versions(name)["version"].to_list()
    out = {}
    for v in versions:
        ours = registry.get(name, version=v)
        if v not in done:
            with mlflow.start_run(run_name=f"{name}-v{v}"):
                mlflow.set_tags({"mlpilot.registry": name, "mlpilot.version": str(v),
                                 **({SOURCE_TAG: ours.run_id} if ours.run_id else {})})
                metrics = {f"{_key(k)}": float(x) for k, x in (ours.metrics or {}).items()
                           if isinstance(x, (int, float)) and x == x}
                if metrics:
                    mlflow.log_metrics(metrics)
                info = _log_model_dir(ours.model_path, example)
            # register_model resolves MLflow 3 logged-model URIs (models:/m-...) to their model id
            mv = mlflow.register_model(info.model_uri, target,
                                       tags={"mlpilot.version": str(v), "stage": ours.stage,
                                             "created_by": str(ours.created_by)})
            if ours.description:
                client.update_model_version(target, mv.version, description=ours.description)
            done[v] = int(mv.version)
        else:
            client.set_model_version_tag(target, str(done[v]), "stage", ours.stage)
        out[v] = done[v]
    # stages -> aliases: the newest version in each stage holds the alias
    for stage in ("dev", "staging", "production"):
        holders = [v for v in versions if registry.get(name, version=v).stage == stage]
        if holders:
            client.set_registered_model_alias(target, stage, str(out[holders[-1]]))
        else:
            try:
                client.delete_registered_model_alias(target, stage)
            except Exception:
                pass
    return out
