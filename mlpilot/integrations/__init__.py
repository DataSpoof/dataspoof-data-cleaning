"""Integrations with the rest of the data stack.

* ``integrations.tasks``    plain functions: train & register, batch score, monitor & retrain
* ``integrations.airflow``  Airflow operators        (needs apache-airflow)
* ``integrations.prefect``  Prefect tasks + a flow   (needs prefect)
* ``integrations.dagster``  Dagster assets           (needs dagster)
* ``integrations.mlflow``   runs, pyfunc models and registry stages to MLflow (needs mlflow)
* ``integrations.dvc``      version datasets with git + DVC, read any revision (needs dvc, git)
* ``integrations.feast``    export the feature store as a Feast repository (needs feast)
* ``integrations.kafka``    stream scoring with Kafka (needs confluent-kafka)
* ``integrations.evidently`` Evidently drift / performance reports (needs evidently)
* ``integrations.aws_glue`` Glue Data Catalog tables + Glue jobs (needs boto3)
* ``integrations.azure_data_factory``  trigger and wait for ADF pipelines (REST; no SDK needed)

Import the submodule you need; only that one requires its third-party package.
"""

from . import tasks

__all__ = ["tasks"]
