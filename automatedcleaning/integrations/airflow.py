"""Apache Airflow operators (needs ``apache-airflow``).

>>> from automatedcleaning.integrations.airflow import AutoMLTrainOperator, BatchScoreOperator, MonitorOperator
>>> train = AutoMLTrainOperator(task_id="train", source="s3://bucket/churn/{{ ds }}.parquet", target="churned",
...                             registry="/models", name="churn", promote_to="staging",
...                             require={"test_roc_auc": 0.8})

Each operator returns a JSON-able dict, pushed to XCom.
"""

from __future__ import annotations

try:
    from airflow.models import BaseOperator
except ImportError as exc:                                   # pragma: no cover - exercised without airflow
    raise ImportError('These operators need Apache Airflow: pip install apache-airflow') from exc

from . import tasks


class AutoMLTrainOperator(BaseOperator):
    template_fields = ("source", "registry", "name", "tracker")

    def __init__(self, *, source, target: str, registry: str, name: str, tracker: str | None = None,
                 promote_to: str | None = None, require: dict | None = None, automl_kwargs: dict | None = None,
                 read_kwargs: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.source, self.target, self.registry, self.name, self.tracker = source, target, registry, name, tracker
        self.promote_to, self.require = promote_to, require
        self.automl_kwargs, self.read_kwargs = automl_kwargs, read_kwargs

    def execute(self, context):
        return tasks.train_and_register(self.source, self.target, registry=self.registry, name=self.name,
                                        tracker=self.tracker, promote_to=self.promote_to, require=self.require,
                                        automl_kwargs=self.automl_kwargs, read_kwargs=self.read_kwargs)


class BatchScoreOperator(BaseOperator):
    template_fields = ("source", "output", "registry", "name")

    def __init__(self, *, registry: str, name: str, source, output, stage: str = "production", **kwargs):
        super().__init__(**kwargs)
        self.registry, self.name, self.source, self.output, self.stage = registry, name, source, output, stage

    def execute(self, context):
        return tasks.batch_score_from_registry(self.registry, self.name, self.source, self.output, stage=self.stage)


class MonitorOperator(BaseOperator):
    template_fields = ("batch", "monitor_path", "train_source", "holdout_source")

    def __init__(self, *, registry: str, name: str, batch, monitor_path, train_source=None, holdout_source=None,
                 fail_on_retrain: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.registry, self.name, self.batch, self.monitor_path = registry, name, batch, monitor_path
        self.train_source, self.holdout_source, self.fail_on_retrain = train_source, holdout_source, fail_on_retrain

    def execute(self, context):
        out = tasks.monitor_and_retrain(self.registry, self.name, self.batch, self.monitor_path,
                                        train_source=self.train_source, holdout_source=self.holdout_source)
        if self.fail_on_retrain and out["retrain_recommended"] and not out["retrained"]:
            raise RuntimeError(f"Model '{self.name}' needs retraining: {out['alerts']}")
        return out
