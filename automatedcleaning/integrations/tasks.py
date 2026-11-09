"""The units of work every orchestrator integration runs (plain functions, JSON-able results).

They are what an Airflow operator / Prefect task / Dagster op calls, and can be used directly
from any scheduler (cron, Argo, Step Functions …)."""

from __future__ import annotations

from pathlib import Path


def train_and_register(source, target: str, *, registry: str, name: str, tracker: str | None = None,
                       experiment: str | None = None, test_fraction: float = 0.2, promote_to: str | None = None,
                       require: dict | None = None, read_kwargs: dict | None = None, automl_kwargs: dict | None = None,
                       sign_with=None) -> dict:
    """Read data, train AutoML on a train split, evaluate on the held-out split, track the run,
    register the model and (optionally) promote it if it passes ``require``."""
    from ..automl import AutoML
    from ..ingestion import read
    from ..registry import ModelRegistry, PromotionError, Tracker
    df = read(source, **(read_kwargs or {}))
    df = df.sample(fraction=1.0, shuffle=True, seed=42) if not (automl_kwargs or {}).get("time_col") else df
    cut = int(df.height * (1 - test_fraction))
    train, test = df.head(cut), df.tail(df.height - cut)
    run = None
    if tracker:
        run = Tracker(tracker, experiment=experiment or name).start_run(f"{name}-train")
    try:
        model = AutoML(target, verbose=False, **(automl_kwargs or {})).fit(train)
        if run is not None:
            run.log_automl(model, train, test)
        reg = ModelRegistry(registry)
        metrics = {}
        if run is None:
            from ..evaluation import evaluate_model
            metrics = {f"test_{k}": v for k, v in evaluate_model(model, test, bootstrap=0).metrics.items() if v is not None}
        version = reg.register(model, name, run=run, metrics=metrics, dataset=train, sign_with=sign_with)
        promoted = None
        if promote_to:
            try:
                reg.promote(name, version.version, promote_to, require=require, reason="automated training run")
                promoted = promote_to
            except PromotionError as exc:
                promoted = f"blocked: {exc}"
        if run is not None:
            run.end()
        return {"name": name, "version": version.version, "promoted": promoted, "metrics": version.metrics,
                "run_id": run.id if run else None}
    except Exception as exc:
        if run is not None:
            run.end("failed", error=f"{type(exc).__name__}: {exc}"[:1000])
        raise


def batch_score_from_registry(registry: str, name: str, source, output, *, stage: str = "production",
                              chunk_size: int = 100_000, keep="all") -> dict:
    """Score a file with the model currently in ``stage``."""
    from ..deploy import batch_score
    from ..registry import ModelRegistry
    version = ModelRegistry(registry).get(name, stage=stage)
    info = batch_score(version.model_path, source, output, chunk_size=chunk_size, keep=keep)
    return {**info, "model": name, "version": version.version}


def monitor_and_retrain(registry: str, name: str, batch, monitor_path, *, train_source=None, holdout_source=None,
                        stage: str = "production", promote_to: str = "staging", alert_handlers=()) -> dict:
    """Check a batch against the saved monitor; if retraining is recommended and training data is
    given, train a challenger, register it and promote it only if it beats the champion."""
    from ..ingestion import read
    from ..monitoring import Monitor, retrain
    from ..registry import ModelRegistry
    reg = ModelRegistry(registry)
    version = reg.get(name, stage=stage)
    model = version.load()
    mon = Monitor.load(monitor_path, model)
    for h in alert_handlers:
        mon.on_alert(h)
    report = mon.check(read(batch))
    out = {"retrain_recommended": report.retrain_recommended, "alerts": [a.message for a in report.alerts],
           "retrained": None}
    if report.retrain_recommended and train_source is not None and holdout_source is not None:
        result = retrain(model, read(train_source), read(holdout_source), registry=reg, name=name,
                         promote_to=promote_to)
        out["retrained"] = {"version": result.version.version if result.version else None, "improved": result.improved,
                            "promoted_to": result.promoted_to, "metric": result.metric,
                            "challenger": result.challenger_score, "champion": result.champion_score}
    return out


def save_monitor(registry: str, name: str, reference, path, *, stage: str = "production") -> str:
    """Fit a monitor for the model in ``stage`` on reference data (e.g. the holdout) and save it."""
    from ..ingestion import read
    from ..monitoring import Monitor
    from ..registry import ModelRegistry
    model = ModelRegistry(registry).load(name, stage=stage)
    return str(Monitor(model, read(reference), name=name).save(Path(path)))
