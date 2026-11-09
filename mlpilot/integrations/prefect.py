"""Prefect tasks and a ready-made flow (needs ``prefect``).

>>> from mlpilot.integrations.prefect import ml_lifecycle_flow
>>> ml_lifecycle_flow(source="data/churn.parquet", target="churned", registry="/models", name="churn",
...                   score_source="data/new.parquet", score_output="out/scored.parquet")
"""

from __future__ import annotations

try:
    from prefect import flow, task
except ImportError as exc:                                   # pragma: no cover
    raise ImportError("These tasks need Prefect: pip install prefect") from exc

from . import tasks

train_task = task(tasks.train_and_register, name="automl-train-and-register", retries=1)
score_task = task(tasks.batch_score_from_registry, name="batch-score", retries=2, retry_delay_seconds=30)
monitor_task = task(tasks.monitor_and_retrain, name="monitor-and-retrain")


@flow(name="mlpilot-ml-lifecycle")
def ml_lifecycle_flow(source, target: str, registry: str, name: str, score_source=None, score_output=None,
                      promote_to: str = "production", require: dict | None = None):
    trained = train_task(source, target, registry=registry, name=name, promote_to=promote_to, require=require)
    scored = None
    if score_source is not None and score_output is not None:
        scored = score_task(registry, name, score_source, score_output, stage=promote_to)
    return {"trained": trained, "scored": scored}
