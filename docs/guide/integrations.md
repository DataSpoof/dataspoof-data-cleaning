# Integrations & scale


| | |
|---|---|
| **Airflow** | `AutoMLTrainOperator`, `BatchScoreOperator`, `MonitorOperator` (`integrations.airflow`) |
| **Prefect / Dagster** | tasks and a lifecycle flow; software-defined assets |
| **MLflow** | export runs, models and registry stages to an MLflow server (see below) |
| **DVC** | version datasets in a git + DVC repository and read back any old revision (see below) |
| **Feast** | export the feature store as a ready-to-apply Feast repository (see below) |
| **Kafka** | `score_stream(model, KafkaSource(...), KafkaSink(...))`: offsets are committed only after delivery, so every record is processed at least once |
| **Spark / Ray / Dask** | `ac.scale.score_spark(model_path, sdf)`, `score_ray`, `score_dask` |
| **Files of any size** | `ac.transform_file(pipeline, "big.csv", "clean.parquet")` (memory-bounded chunks) |
| **SQL pushdown** | `ac.to_sql(pipeline, "raw.orders", dialect="snowflake")`: a fitted cleaning pipeline becomes one `SELECT`, so the warehouse does the work and the data never leaves it. In testing, results matched Python exactly. Any step that has no exact SQL equivalent raises an error. |

The orchestrator wrappers all call `integrations.tasks` (train & register, batch score, monitor
& retrain). Those are plain functions, so they also work from cron, Argo or Step Functions.

### Already running MLflow, DVC or Feast?

The built-in tracker, data repository and feature store need no servers. If your team already
runs one of these tools, these bridges move the same objects into it:

```python
from automatedcleaning.integrations import mlflow as acm
acm.export_tracker(tracker, tracking_uri="http://mlflow:5000")       # every run: params, metric history, artifacts
info = acm.log_automl(model, test, experiment="churn", register_as="churn")   # a pyfunc model with a signature
acm.export_registry(registry, "churn")                                # versions; stages become aliases
model = acm.load_automl("models:/churn@production")                   # back to an AutoML object
#   $ mlflow models serve -m models:/churn@production --env-manager local

from automatedcleaning.integrations.dvc import DVCRepo, export_repository
repo = DVCRepo.init("ml-data")                                        # git init + dvc init
repo.add_remote("storage", "s3://bucket/dvc")
repo.track(df, "data/customers.parquet", message="Q2 refresh", tag="customers-v2", push=True)
old = repo.read("data/customers.parquet", rev="customers-v1")         # any commit, tag or branch
export_repository(data_repo, "customers", repo)                       # DataRepository history -> git commits + tags

from automatedcleaning.integrations.feast import export_to_feast, feast_features
fs = export_to_feast(store, "feast_repo")             # feature_store.yaml + features.py, applied and materialized
fs.get_historical_features(entity_df, feast_features(store, ["customer_stats:*"])).to_df()
```

**MLflow** (`[mlflow]`):
- Models are logged as **pyfunc** models with a signature and an input example, so
  `mlflow.pyfunc.load_model` and `mlflow models serve` work on them directly.
- Exports are batched. Running an export again skips runs and versions already copied, so a
  scheduled sync is safe.

**DVC** (`[dvc]`): it calls the real `git` and `dvc` commands, so the result is an ordinary DVC
repository that `dvc pull`, `dvc diff` and CI already understand.
- Re-committing unchanged data creates no commit.
- A version that is missing from the local cache is fetched from the remote.
- `log_to_run` records in a run exactly which data version it used.

**Feast** (`[feast]`):
- The export writes `feature_store.yaml`, generated Entity / FileSource / FeatureView
  definitions (types, TTL, owner, tags) and each view's history.
- To use your production online store (Redis, DynamoDB, Postgres…), change one block of
  `feature_store.yaml`.
- An entity with several join keys becomes one Feast entity per key.

**Tested against the real tools, not stand-ins:**

| Tool | Setup | What the tests check |
|---|---|---|
| MLflow 3.18 | a real `mlflow server`, plus `mlflow models serve` answering HTTP | predictions over HTTP match predictions in Python |
| DVC 3.67 | git, plus a DVC remote | push, then delete the cache and pull, then read old revisions |
| Feast 0.67 | `apply` and `materialize` | point-in-time results and online lookups match ours value for value |

There is one behaviour difference. For entity rows that have no value inside the TTL window,
Feast's file offline store can leave the row out. `FeatureStore` keeps the row, with null
features.
