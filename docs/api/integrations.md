# Integrations: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Integrations guide](../guide/integrations.md).

**On this page:**

- [`mlpilot.integrations`](#mlpilotintegrations)
- [`mlpilot.integrations.airflow`](#mlpilotintegrationsairflow): `AutoMLTrainOperator`, `BatchScoreOperator`, `MonitorOperator`
- [`mlpilot.integrations.aws_glue`](#mlpilotintegrationsaws_glue): `hive_type`, `polars_type`, `register_table`, `write_table`, `describe_table`, `table_schema`, `read_table`, `run_job`
- [`mlpilot.integrations.azure_data_factory`](#mlpilotintegrationsazure_data_factory): `DataFactoryError`, `DataFactory`
- [`mlpilot.integrations.dagster`](#mlpilotintegrationsdagster): `ml_assets`
- [`mlpilot.integrations.dvc`](#mlpilotintegrationsdvc): `DVCError`, `DVCVersion`, `DVCRepo`, `export_repository`
- [`mlpilot.integrations.evidently`](#mlpilotintegrationsevidently): `EvidentlyResult`, `evidently_report`
- [`mlpilot.integrations.feast`](#mlpilotintegrationsfeast): `feast_features`, `export_to_feast`, `to_feast_entity_df`
- [`mlpilot.integrations.kafka`](#mlpilotintegrationskafka): `KafkaSource`, `KafkaSink`, `score_stream`
- [`mlpilot.integrations.mlflow`](#mlpilotintegrationsmlflow): `find_exported`, `export_run`, `export_tracker`, `AutoMLPyfunc`, `log_automl`, `load_automl`, `export_registry`
- [`mlpilot.integrations.prefect`](#mlpilotintegrationsprefect): `ml_lifecycle_flow`
- [`mlpilot.integrations.tasks`](#mlpilotintegrationstasks): `train_and_register`, `batch_score_from_registry`, `monitor_and_retrain`, `save_monitor`

---

## `mlpilot.integrations`

<sub>Source: [`mlpilot/integrations/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/integrations/__init__.py)</sub>

Integrations with the rest of the data stack.

* `integrations.tasks`    plain functions: train & register, batch score, monitor & retrain
* `integrations.airflow`  Airflow operators        (needs apache-airflow)
* `integrations.prefect`  Prefect tasks + a flow   (needs prefect)
* `integrations.dagster`  Dagster assets           (needs dagster)
* `integrations.mlflow`   runs, pyfunc models and registry stages to MLflow (needs mlflow)
* `integrations.dvc`      version datasets with git + DVC, read any revision (needs dvc, git)
* `integrations.feast`    export the feature store as a Feast repository (needs feast)
* `integrations.kafka`    stream scoring with Kafka (needs confluent-kafka)
* `integrations.evidently` Evidently drift / performance reports (needs evidently)
* `integrations.aws_glue` Glue Data Catalog tables + Glue jobs (needs boto3)
* `integrations.azure_data_factory`  trigger and wait for ADF pipelines (REST; no SDK needed)

Import the submodule you need; only that one requires its third-party package.

---

## `mlpilot.integrations.airflow`

<sub>Source: [`mlpilot/integrations/airflow.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/integrations/airflow.py)</sub>

Apache Airflow operators (needs `apache-airflow`).

```python
from mlpilot.integrations.airflow import AutoMLTrainOperator, BatchScoreOperator, MonitorOperator
train = AutoMLTrainOperator(task_id="train", source="s3://bucket/churn/{{ ds }}.parquet", target="churned",
                            registry="/models", name="churn", promote_to="staging",
                            require={"test_roc_auc": 0.8})
```

Each operator returns a JSON-able dict, pushed to XCom.

### class `AutoMLTrainOperator`

```python
class AutoMLTrainOperator(BaseOperator):
```

**Constructor**

```python
AutoMLTrainOperator(*, source, target: str, registry: str, name: str, tracker: str | None=None, promote_to: str | None=None, require: dict | None=None, automl_kwargs: dict | None=None, read_kwargs: dict | None=None, **kwargs)
```

#### `AutoMLTrainOperator.execute()`

```python
execute(context)
```

### class `BatchScoreOperator`

```python
class BatchScoreOperator(BaseOperator):
```

**Constructor**

```python
BatchScoreOperator(*, registry: str, name: str, source, output, stage: str='production', **kwargs)
```

#### `BatchScoreOperator.execute()`

```python
execute(context)
```

### class `MonitorOperator`

```python
class MonitorOperator(BaseOperator):
```

**Constructor**

```python
MonitorOperator(*, registry: str, name: str, batch, monitor_path, train_source=None, holdout_source=None, fail_on_retrain: bool=False, **kwargs)
```

#### `MonitorOperator.execute()`

```python
execute(context)
```

---

## `mlpilot.integrations.aws_glue`

<sub>Source: [`mlpilot/integrations/aws_glue.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/integrations/aws_glue.py)</sub>

AWS Glue: the Data Catalog and Glue jobs (needs `boto3`; the `[aws]` extra).

```python
from mlpilot.integrations import aws_glue as glue
glue.write_table(clean_df, "analytics", "customers", "s3://lake/analytics/customers/",
                 partition_by=["country"], description="Cleaned customer master",
                 parameters={"quality_score": "97.5"})        # Parquet on S3 + a catalog table
glue.register_table(df, "analytics", "orders", "s3://lake/orders/")   # catalog only (data already there)
glue.table_schema("analytics", "customers")                 # -> Polars schema from the catalog
df = glue.read_table("analytics", "customers")              # location + format from the catalog
glue.run_job("nightly-orders-etl", {"--date": "2026-11-08"})  # start a Glue job and wait
```

Athena, Redshift Spectrum, EMR and Lake Formation all read the same catalog. Every function takes
`glue=` / `storage_options=` for a custom client or endpoint.

### `hive_type()`

```python
def hive_type(dtype) -> str
```

Polars dtype -> Glue / Hive type string.

### `polars_type()`

```python
def polars_type(hive: str)
```

Glue / Hive type string -> Polars dtype (complex types become String).

### `register_table()`

```python
def register_table(df_or_schema, database: str, table: str, location: str, *, fmt: str='parquet', partition_by: list | None=None, description: str | None=None, parameters: dict | None=None, column_comments: dict | None=None, create_database: bool=True, glue=None) -> dict
```

Create or update a catalog table describing data at `location`. `df_or_schema`: a
DataFrame or a `{column: dtype}` schema. Returns the Glue `TableInput` that was written.

### `write_table()`

```python
def write_table(df: pl.DataFrame, database: str, table: str, location: str, *, partition_by: list | None=None, storage_options: dict | None=None, glue=None, add_partitions: bool=True, **register_kw) -> dict
```

Write `df` as Parquet under `location` (Hive-style `key=value` folders when
`partition_by`), register the table, and add its partitions to the catalog.

### `describe_table()`

```python
def describe_table(database: str, table: str, *, glue=None) -> dict
```

### `table_schema()`

```python
def table_schema(database: str, table: str, *, glue=None) -> dict
```

`{column: polars dtype}` from the catalog (partition keys included, last).

### `read_table()`

```python
def read_table(database: str, table: str, *, storage_options: dict | None=None, lazy: bool=False, glue=None)
```

Read a catalog table from its S3 location in the format the catalog records.

### `run_job()`

```python
def run_job(name: str, arguments: dict | None=None, *, wait: bool=True, poll: float=15.0, timeout: float=6 * 3600, glue=None) -> dict
```

Start a Glue job (`arguments` like `{"--date": "2026-11-08"}`) and, with `wait`, poll
until it finishes. Raises `RuntimeError` if it fails, stops or times out.

---

## `mlpilot.integrations.azure_data_factory`

<sub>Source: [`mlpilot/integrations/azure_data_factory.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/integrations/azure_data_factory.py)</sub>

Azure Data Factory: trigger pipelines and wait for them (Azure Resource Manager REST API).

```python
from mlpilot.integrations.azure_data_factory import DataFactory
adf = DataFactory(subscription="…", resource_group="data-rg", factory="corp-adf")
run = adf.run_pipeline("CopyOrdersToLake", {"date": "2026-11-08"})     # waits by default
run["status"], run["duration_ms"]
adf.activity_runs(run["run_id"])                                          # per-activity results
```

Authentication: `token=` (a bearer token or a secret reference such as
`vault://secret/azure#token`), else `AZURE_ACCESS_TOKEN`, else `azure-identity`'s
`DefaultAzureCredential` (managed identity, workload identity, `az login`...).
No Azure SDK is needed for the API calls themselves.

### class `DataFactoryError`

```python
class DataFactoryError(RuntimeError):
```

### class `DataFactory`

```python
class DataFactory:
```

**Constructor**

```python
DataFactory(subscription: str, resource_group: str, factory: str, *, token: str | None=None, base_url: str='https://management.azure.com', timeout: float=30.0)
```

#### `DataFactory.start_pipeline()`

```python
start_pipeline(pipeline: str, parameters: dict | None=None, *, reference_run_id: str | None=None) -> str
```

#### `DataFactory.pipeline_run()`

```python
pipeline_run(run_id: str) -> dict
```

#### `DataFactory.run_pipeline()`

```python
run_pipeline(pipeline: str, parameters: dict | None=None, *, wait: bool=True, poll: float=15.0, timeout: float=6 * 3600) -> dict
```

Trigger `pipeline`; with `wait`, poll until it finishes and raise if it didn't succeed.

#### `DataFactory.cancel()`

```python
cancel(run_id: str) -> None
```

#### `DataFactory.activity_runs()`

```python
activity_runs(run_id: str, *, days: int=7) -> list
```

---

## `mlpilot.integrations.dagster`

<sub>Source: [`mlpilot/integrations/dagster.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/integrations/dagster.py)</sub>

Dagster software-defined assets (needs `dagster`).

```python
from mlpilot.integrations.dagster import ml_assets
defs = dagster.Definitions(assets=ml_assets(source="data/churn.parquet", target="churned",
                                            registry="/models", name="churn"))
```

### `ml_assets()`

```python
def ml_assets(*, source, target: str, registry: str, name: str, score_source=None, score_output=None, promote_to: str='production', require: dict | None=None, group: str='ml')
```

A `trained_model` asset and, with `score_source`/`score_output`, a `scored_batch`
asset downstream of it.

---

## `mlpilot.integrations.dvc`

<sub>Source: [`mlpilot/integrations/dvc.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/integrations/dvc.py)</sub>

DVC integration (needs `dvc` and `git`; the `[dvc]` extra).

Version datasets with DVC when your team already uses it: the data goes to DVC's cache and
remote (S3, GCS, Azure, SSH, a shared drive), a small `.dvc` pointer goes into git, and any
past version can be read back by git revision.

```python
from mlpilot.integrations.dvc import DVCRepo, export_repository
repo = DVCRepo.init("ml-data")                                  # git init + dvc init (or an existing repo)
repo.add_remote("storage", "s3://bucket/dvc")                   # or a local / network folder
v = repo.track(df, "data/customers.parquet", message="Q2 refresh", tag="customers-v2", push=True)
repo.versions("data/customers.parquet")                         # rev, author, date, md5, size, tags
old = repo.read("data/customers.parquet", rev="customers-v1")   # any git rev / tag / branch
repo.log_to_run(run, "data/customers.parquet")                  # lineage: run -> exact data version
export_repository(data_repo, "customers", repo)                 # replay DataRepository history into DVC
```

All operations shell out to the real `git` and `dvc` commands (the same ones you'd type), so
the result is an ordinary DVC repository that `dvc pull` / `dvc diff` / CI can use.

### class `DVCError`

```python
class DVCError(RuntimeError):
```

A git or dvc command failed.

### class `DVCVersion`

```python
class DVCVersion:
```

| Field | Type | Default |
|---|---|---|
| `path` | `str` | required |
| `rev` | `str` | required |
| `md5` | `str` | required |
| `size` | `int \| None` | required |
| `message` | `str` | required |
| `tags` | `list` | required |
| `author` | `str \| None` | `None` |
| `date` | `str \| None` | `None` |

#### `DVCVersion.ref` *(property)*

```python
ref: str
```

### class `DVCRepo`

```python
class DVCRepo:
```

A git + DVC repository on disk.

**Constructor**

```python
DVCRepo(root)
```

#### `DVCRepo.git()`

```python
git(*args, check: bool=True) -> str
```

#### `DVCRepo.dvc()`

```python
dvc(*args, check: bool=True) -> str
```

#### `DVCRepo.init()` *(classmethod)*

```python
init(root, *, git: bool=True) -> 'DVCRepo'
```

Create (or reuse) a git + DVC repository at `root`.

#### `DVCRepo.add_remote()`

```python
add_remote(name: str, url: str, *, default: bool=True) -> None
```

Register where data is pushed (`s3://`, `gs://`, `azure://`, `ssh://` or a folder).

#### `DVCRepo.push()`

```python
push(remote: str | None=None, targets=()) -> None
```

#### `DVCRepo.pull()`

```python
pull(remote: str | None=None, targets=()) -> None
```

#### `DVCRepo.track()`

```python
track(data, path, *, message: str | None=None, tag: str | None=None, push: bool=False, remote: str | None=None) -> DVCVersion
```

Write `data` (a DataFrame, or an existing file to copy) to `path` in the repo, put
it under DVC, and commit the pointer to git (optionally tagging the commit and pushing the
data). If the data is identical to the tracked version, nothing new is committed.

#### `DVCRepo.version()`

```python
version(path, rev: str='HEAD') -> DVCVersion
```

The tracked version of `path` at a git revision.

#### `DVCRepo.versions()`

```python
versions(path) -> pl.DataFrame
```

Every committed version of `path`, newest first.

#### `DVCRepo.read()`

```python
read(path, rev: str | None=None, *, remote: str | None=None) -> pl.DataFrame
```

A dataset as it was at git revision `rev` (a commit, tag or branch; default: the
workspace). Data comes from the local DVC cache, or the remote if it isn't cached.

#### `DVCRepo.checkout()`

```python
checkout(path, rev: str) -> Path
```

Restore the workspace copy of `path` to the version at `rev` (fetched from the
remote when it isn't in the local cache). Commit or check out the `.dvc` file again to
go back.

#### `DVCRepo.status()`

```python
status() -> str
```

#### `DVCRepo.log_to_run()`

```python
log_to_run(run, path, rev: str='HEAD') -> DVCVersion
```

Record in a tracking run exactly which DVC data version it used.

### `export_repository()`

```python
def export_repository(data_repo, name: str, dvc_repo: DVCRepo, path=None, *, push: bool=False) -> list
```

Replay the versions of dataset `name` from a `DataRepository`
into DVC as one git commit per version, tagged `<name>-v<N>` (versions already exported
are skipped). Returns the `DVCVersion` list.

---

## `mlpilot.integrations.evidently`

<sub>Source: [`mlpilot/integrations/evidently.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/integrations/evidently.py)</sub>

Evidently integration (needs `evidently>=0.7`; the `[evidently]` extra).

Use Evidently's reports and UI next to the built-in monitoring:

```python
from mlpilot.integrations.evidently import evidently_report
res = evidently_report(reference_df, current_df, model=model, html="drift.html")
res.drifted_columns, res.drift_share        # parsed from Evidently's own drift tests
res.compare(ac.drift_report(reference_df, current_df))   # agreement with our PSI report
res.snapshot                                # the Evidently Snapshot (save_json, dict, upload to a workspace)
```

With `model=` the model's predictions (and class probabilities) are added to both datasets,
so Evidently's classification / regression presets run as well when the target is present.

### class `EvidentlyResult`

```python
class EvidentlyResult:
```

| Field | Type | Default |
|---|---|---|
| `snapshot` | `Any` | required |
| `column_drift` | `dict` | required |
| `metrics` | `dict` | required |

#### `EvidentlyResult.drifted_columns` *(property)*

```python
drifted_columns: list
```

#### `EvidentlyResult.drift_share` *(property)*

```python
drift_share: float
```

#### `EvidentlyResult.compare()`

```python
compare(ours) -> dict
```

Agreement with a `DriftReport` (moderate or major
drift counts as drift on our side).

#### `EvidentlyResult.save_html()`

```python
save_html(path) -> None
```

### `evidently_report()`

```python
def evidently_report(reference, current, *, model=None, target: str | None=None, columns: list | None=None, html=None, json_path=None) -> EvidentlyResult
```

Run Evidently's data-drift preset (plus classification / regression presets when a model
and labels are available) on `current` vs `reference`.

---

## `mlpilot.integrations.feast`

<sub>Source: [`mlpilot/integrations/feast.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/integrations/feast.py)</sub>

Feast integration (needs `feast`; the `[feast]` extra).

Start with the built-in `FeatureStore`, and move to Feast when you
need its online stores (Redis, DynamoDB, Bigtable, Postgres...), its feature server or its UI.
The export writes an ordinary Feast repository, so the `feast` CLI works on it:

```python
from mlpilot.integrations.feast import export_to_feast, feast_features
fs = export_to_feast(store, "feast_repo", project="growth")     # feature_store.yaml + definitions, applied
fs.get_historical_features(entity_df.to_pandas(), feast_features(store, ["customer_stats:*"])).to_df()
fs.get_online_features(["customer_stats:spend_30d"], [{"customer_id": 42}]).to_dict()
```

What gets exported:

* `feature_store.yaml`  local provider, file offline store, SQLite online store (edit it to
  point at your production stores; the definitions don't change).
* `features.py`         Entities, FileSources and FeatureViews generated from the registry
  (types from the stored schema, TTL, description, owner, tags).
* `data/<view>.parquet` each view's full offline history (timestamps in UTC).

An entity with several join keys becomes one Feast entity per key (Feast entities have one key).
Point-in-time semantics match: feature timestamp <= entity timestamp, within the TTL (verified
value-for-value against Feast 0.67). One difference: Feast's *file* offline store can leave out
some entity rows that have no feature value inside the TTL window, while
`FeatureStore.get_historical_features` keeps them with nulls so no labelled row is lost.

### `feast_features()`

```python
def feast_features(store, refs: list) -> list
```

Our feature references (`"view:feature"` / `"view:*"`) as Feast references.

### `export_to_feast()`

```python
def export_to_feast(store, repo_dir, *, project: str='mlpilot', views: list | None=None, apply: bool=True, materialize: bool=True, online_store_yaml: str | None=None)
```

Write a Feast repository for `store` (all views, or `views`), then optionally
`feast apply` it and materialize the online store. Returns a `feast.FeatureStore`.

`online_store_yaml` replaces the default SQLite `online_store:` block, e.g.
`"online_store:\n  type: redis\n  connection_string: redis:6379\n"`.

### `to_feast_entity_df()`

```python
def to_feast_entity_df(entity_df, timestamp: str='event_timestamp')
```

A pandas entity DataFrame Feast accepts (UTC timestamps in `event_timestamp`).

---

## `mlpilot.integrations.kafka`

<sub>Source: [`mlpilot/integrations/kafka.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/integrations/kafka.py)</sub>

Streaming with Kafka (needs `confluent-kafka`): consume JSON records in micro-batches,
score them, and produce the predictions to another topic.

```python
src = KafkaSource("orders", bootstrap_servers="kafka:9092", group_id="churn-scorer")
score_stream(model, src, KafkaSink("orders-scored", bootstrap_servers="kafka:9092"))
```

Offsets are committed only after a batch has been scored and delivered (at-least-once).

### class `KafkaSource`

```python
class KafkaSource:
```

**Constructor**

```python
KafkaSource(topic: str, *, bootstrap_servers: str, group_id: str, consumer=None, **config)
```

#### `KafkaSource.batches()`

```python
batches(*, max_records: int=1000, timeout: float=1.0, max_batches: int | None=None)
```

Yield `(DataFrame, messages)` micro-batches; invalid JSON messages are skipped (counted).

#### `KafkaSource.commit()`

```python
commit(messages)
```

#### `KafkaSource.close()`

```python
close()
```

### class `KafkaSink`

```python
class KafkaSink:
```

**Constructor**

```python
KafkaSink(topic: str, *, bootstrap_servers: str | None=None, producer=None, **config)
```

#### `KafkaSink.write()`

```python
write(records: list, keys: list | None=None)
```

### `score_stream()`

```python
def score_stream(model, source: KafkaSource, sink, *, key: str | None=None, keep=None, max_records: int=1000, max_batches: int | None=None) -> dict
```

Consume -> score -> produce until `max_batches` (None = forever). `sink` is a
`KafkaSink` or any callable taking a list of output dicts.

---

## `mlpilot.integrations.mlflow`

<sub>Source: [`mlpilot/integrations/mlflow.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/integrations/mlflow.py)</sub>

MLflow integration (needs `mlflow`, the `[mlflow]` extra).

Keep using MLflow's UI, model registry and serving if your company runs MLflow:

```python
from mlpilot.integrations import mlflow as acm
acm.export_run(run, tracking_uri="http://mlflow:5000")          # params, metric history, artifacts
acm.export_tracker(tracker)                                     # every run (re-running skips copies)
info = acm.log_automl(model, test_df, experiment="churn", register_as="churn")
acm.export_registry(registry, "churn")                          # versions + stages -> aliases
model = acm.load_automl("models:/churn@production")             # back to an AutoML object
```

Models are logged as MLflow **pyfunc** models with a signature and an input example, so all of
MLflow works on them:
```
mlflow.pyfunc.load_model("models:/churn/1").predict(pandas_df)
mlflow models serve -m models:/churn@production --env-manager local
```

The pyfunc output is a DataFrame: `<target>_pred` plus one `proba_<class>` column per class
for classifiers. The saved AutoML directory is stored inside the MLflow model (its checksum is
verified again when it is loaded).

### `find_exported()`

```python
def find_exported(run_id: str, *, tracking_uri: str | None=None, experiment: str | None=None) -> str | None
```

The MLflow run id a tracked run was already exported to, or `None`.

### `export_run()`

```python
def export_run(run, *, tracking_uri: str | None=None, experiment: str | None=None, skip_existing: bool=True) -> str
```

Copy a tracked `Run` into MLflow: params, every metric
value with its step and time, tags, dataset fingerprints, start/end time, status and all
artifacts. Returns the MLflow run id.

With `skip_existing` (default), a run that was already exported (matched by the
`mlpilot.run_id` tag) is not copied again, so syncing is safe to repeat.

### `export_tracker()`

```python
def export_tracker(tracker, *, experiment: str | None=None, tracking_uri: str | None=None, mlflow_experiment: str | None=None) -> dict
```

Export every run of a `Tracker` (or one experiment).
Returns `{our run id: mlflow run id}`; runs already exported are skipped.

### class `AutoMLPyfunc`

```python
class AutoMLPyfunc(mlflow.pyfunc.PythonModel):
```

Wraps a saved AutoML / AutoCluster / Forecaster directory as an MLflow pyfunc model.

#### `AutoMLPyfunc.load_context()`

```python
load_context(context)
```

#### `AutoMLPyfunc.predict()`

```python
predict(context, model_input, params=None)
```

### `log_automl()`

```python
def log_automl(model, test_df=None, *, tracking_uri: str | None=None, experiment: str | None=None, register_as: str | None=None, run_name: str | None=None, tags: dict | None=None)
```

Log a fitted AutoML model to MLflow as a pyfunc model: settings, the CV score, test
metrics (when `test_df` is given; its first rows also become the input example and
signature), and the model itself. `register_as` also creates a registered model version.

Returns MLflow's `ModelInfo` (`.run_id`, `.model_uri`, `.registered_model_version`).

### `load_automl()`

```python
def load_automl(model_uri: str, *, tracking_uri: str | None=None)
```

Load a model logged by `log_automl` / `export_registry` back as the original
AutoML object (`runs:/<id>/model`, `models:/<name>/<version>`, `models:/<name>@<alias>`).

### `export_registry()`

```python
def export_registry(registry, name: str, *, register_as: str | None=None, tracking_uri: str | None=None, experiment: str | None=None, example=None) -> dict
```

Copy every version of model `name` from a `ModelRegistry`
into the MLflow model registry. Each version keeps its metrics and description, the stage
becomes an MLflow **alias** (`production`, `staging`, `dev`) and a `stage` tag.
Versions already exported are skipped. Returns `{our version: mlflow version}`.

---

## `mlpilot.integrations.prefect`

<sub>Source: [`mlpilot/integrations/prefect.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/integrations/prefect.py)</sub>

Prefect tasks and a ready-made flow (needs `prefect`).

```python
from mlpilot.integrations.prefect import ml_lifecycle_flow
ml_lifecycle_flow(source="data/churn.parquet", target="churned", registry="/models", name="churn",
                  score_source="data/new.parquet", score_output="out/scored.parquet")
```

### `ml_lifecycle_flow()`

```python
def ml_lifecycle_flow(source, target: str, registry: str, name: str, score_source=None, score_output=None, promote_to: str='production', require: dict | None=None)
```

---

## `mlpilot.integrations.tasks`

<sub>Source: [`mlpilot/integrations/tasks.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/integrations/tasks.py)</sub>

The units of work every orchestrator integration runs (plain functions, JSON-able results).

They are what an Airflow operator / Prefect task / Dagster op calls, and can be used directly
from any scheduler (cron, Argo, Step Functions …).

### `train_and_register()`

```python
def train_and_register(source, target: str, *, registry: str, name: str, tracker: str | None=None, experiment: str | None=None, test_fraction: float=0.2, promote_to: str | None=None, require: dict | None=None, read_kwargs: dict | None=None, automl_kwargs: dict | None=None, sign_with=None) -> dict
```

Read data, train AutoML on a train split, evaluate on the held-out split, track the run,
register the model and (optionally) promote it if it passes `require`.

### `batch_score_from_registry()`

```python
def batch_score_from_registry(registry: str, name: str, source, output, *, stage: str='production', chunk_size: int=100000, keep='all') -> dict
```

Score a file with the model currently in `stage`.

### `monitor_and_retrain()`

```python
def monitor_and_retrain(registry: str, name: str, batch, monitor_path, *, train_source=None, holdout_source=None, stage: str='production', promote_to: str='staging', alert_handlers=()) -> dict
```

Check a batch against the saved monitor; if retraining is recommended and training data is
given, train a challenger, register it and promote it only if it beats the champion.

### `save_monitor()`

```python
def save_monitor(registry: str, name: str, reference, path, *, stage: str='production') -> str
```

Fit a monitor for the model in `stage` on reference data (e.g. the holdout) and save it.
