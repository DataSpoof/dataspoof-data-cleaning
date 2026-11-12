# Tracking & registry: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Tracking & registry guide](../guide/registry.md).

**On this page:**

- [`mlpilot.registry`](#mlpilotregistry)
- [`mlpilot.registry.models`](#mlpilotregistrymodels): `PromotionError`, `ModelVersion`, `ModelRegistry`
- [`mlpilot.registry.tracking`](#mlpilotregistrytracking): `Run`, `Tracker`

---

## `mlpilot.registry`

<sub>Source: [`mlpilot/registry/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/registry/__init__.py)</sub>

Experiment tracking & model registry (module 7).

* `Tracker`        runs with params, metrics (with steps), datasets (fingerprints), artifacts,
  git commit and user; `run.log_automl(model, train, test)` logs it all.
* `ModelRegistry`  versioned models promoted dev → staging → production with metric gates,
  an audit history and rollback.

Both are plain directories of JSON files — no server, easy to back up or put on a shared drive.

---

## `mlpilot.registry.models`

<sub>Source: [`mlpilot/registry/models.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/registry/models.py)</sub>

Model registry: versioned models moving through dev → qa → staging → production.

Layout:
```
<root>/<model name>/v<N>/model/        the saved model (checksummed by AutoML.save)
<root>/<model name>/v<N>/version.json  metrics, run, dataset, stage, who/when
<root>/<model name>/history.jsonl      every stage change, with who, when and why
```

### class `PromotionError`

```python
class PromotionError(Exception):
```

A model version failed the checks required to move to a stage.

### class `ModelVersion`

```python
class ModelVersion:
```

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `version` | `int` | required |
| `stage` | `str` | required |
| `path` | `Path` | required |
| `created_at` | `str` | required |
| `created_by` | `str` | required |
| `metrics` | `dict` | required |
| `tags` | `dict` | required |
| `description` | `str \| None` | required |
| `run_id` | `str \| None` | required |
| `dataset` | `dict \| None` | required |
| `task` | `str \| None` | required |
| `target` | `str \| None` | required |
| `model_sha256` | `str \| None` | required |

#### `ModelVersion.model_path` *(property)*

```python
model_path: Path
```

#### `ModelVersion.load()`

```python
load()
```

### class `ModelRegistry`

```python
class ModelRegistry:
```

A model registry in a directory (default `./ac_registry`).

```python
reg = ModelRegistry("ac_registry")
v = reg.register(model, "churn", run=run)                  # -> v1 in "dev"
reg.promote("churn", v.version, "staging")
reg.promote("churn", v.version, "production", require={"test_roc_auc": 0.85})
model = reg.load("churn")                                   # current production model
```

**Constructor**

```python
ModelRegistry(root='ac_registry')
```

#### `ModelRegistry.register()`

```python
register(model, name: str, *, run=None, stage: str='dev', metrics: dict | None=None, tags: dict | None=None, description: str | None=None, dataset=None, sign_with=None, sign_password: str | None=None) -> ModelVersion
```

Add a new version of model `name`.

`model` is a fitted AutoML / AutoCluster object or a saved model directory.
Metrics come from `run` (a tracked Run) and/or `metrics`; an AutoML model also
contributes its cross-validation score. `dataset` (a DataFrame) records the training
data fingerprint (the run's `train` dataset is used otherwise). `sign_with` (an
Ed25519 private key, see `mlpilot.security`) signs the stored model so
deployments can verify it with `AC_MODEL_PUBLIC_KEY`.

#### `ModelRegistry.models()`

```python
models() -> pl.DataFrame
```

Every registered model: latest version and which version is in each stage.

#### `ModelRegistry.versions()`

```python
versions(name: str) -> pl.DataFrame
```

All versions of a model with their stage and metrics (`metric.*` columns).

#### `ModelRegistry.get()`

```python
get(name: str, version: int | None=None, stage: str | None=None) -> ModelVersion
```

A specific version, the newest version in `stage`, or (neither given) the production
version if there is one, else the newest version.

#### `ModelRegistry.load()`

```python
load(name: str, version: int | None=None, stage: str | None=None)
```

Load a model (default: the production version). The checksum is verified on load.

#### `ModelRegistry.history()`

```python
history(name: str) -> pl.DataFrame
```

#### `ModelRegistry.check()`

```python
check(name: str, version: int, require: dict) -> list
```

Failed requirements for a version, e.g. `{"test_roc_auc": 0.85, "test_rmse": ("<=", 12)}`.

#### `ModelRegistry.promote()`

```python
promote(name: str, version: int, stage: str, *, reason: str | None=None, require: dict | None=None, archive_existing: bool=True, must_beat_production: str | None=None) -> ModelVersion
```

Move a version to `stage`. Moving to production archives the current production
version (`archive_existing`). `require` sets metric thresholds, and
`must_beat_production="test_roc_auc"` requires the candidate to be at least as good as
the current production model on that metric (use a `-` prefix, e.g. `"-test_rmse"`,
for lower-is-better). Failing checks raise `PromotionError` and nothing changes.

#### `ModelRegistry.rollback()`

```python
rollback(name: str, reason: str='rollback') -> ModelVersion
```

Put the previous production version back into production.

---

## `mlpilot.registry.tracking`

<sub>Source: [`mlpilot/registry/tracking.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/registry/tracking.py)</sub>

Experiment tracking: every training run's parameters, metrics, datasets and artifacts.

Layout (plain files, easy to back up, diff or sync to object storage):
```
<root>/<experiment>/<run_id>/run.json        params, tags, datasets, status, git, timings
<root>/<experiment>/<run_id>/metrics.jsonl   one line per logged metric value (with step)
<root>/<experiment>/<run_id>/artifacts/      files, saved models, model cards
```

### class `Run`

```python
class Run:
```

One tracked run. Use as a context manager: it ends as `finished` or, if an
exception escapes, as `failed` with the error recorded.

**Constructor**

```python
Run(path: Path, meta: dict)
```

#### `Run.id` *(property)*

```python
id: str
```

#### `Run.experiment` *(property)*

```python
experiment: str
```

#### `Run.name` *(property)*

```python
name: str
```

#### `Run.status` *(property)*

```python
status: str
```

#### `Run.params` *(property)*

```python
params: dict
```

#### `Run.tags` *(property)*

```python
tags: dict
```

#### `Run.datasets` *(property)*

```python
datasets: list
```

#### `Run.artifacts_dir` *(property)*

```python
artifacts_dir: Path
```

#### `Run.metrics` *(property)*

```python
metrics: dict
```

Latest value of every metric.

#### `Run.metric_history()`

```python
metric_history(key: str | None=None) -> pl.DataFrame
```

#### `Run.to_dict()`

```python
to_dict() -> dict
```

#### `Run.log_param()`

```python
log_param(key: str, value) -> 'Run'
```

#### `Run.log_params()`

```python
log_params(params: dict) -> 'Run'
```

#### `Run.log_metric()`

```python
log_metric(key: str, value: float, step: int | None=None) -> 'Run'
```

#### `Run.log_metrics()`

```python
log_metrics(metrics: dict, step: int | None=None) -> 'Run'
```

#### `Run.set_tag()`

```python
set_tag(key: str, value) -> 'Run'
```

#### `Run.log_dataset()`

```python
log_dataset(df, name: str='train', role: str | None=None) -> dict
```

Record which data the run used (shape, schema and content hash, not the data).

#### `Run.log_artifact()`

```python
log_artifact(source, name: str | None=None) -> Path
```

Copy a file or directory into the run's artifacts.

#### `Run.log_text()`

```python
log_text(text: str, name: str) -> Path
```

#### `Run.log_model()`

```python
log_model(model, name: str='model') -> Path
```

Save a fitted AutoML / AutoCluster model into the run's artifacts.

#### `Run.log_automl()`

```python
log_automl(model, train_df=None, test_df=None, *, card: bool=True, **card_kw) -> 'Run'
```

Log everything about a fitted AutoML model in one call: settings, the CV score and
leaderboard, datasets, held-out metrics (with `test_df`), the model and a model card.

#### `Run.end()`

```python
end(status: str='finished', error: str | None=None) -> 'Run'
```

#### `Run.load_model()`

```python
load_model(name: str | None=None)
```

### class `Tracker`

```python
class Tracker:
```

Tracks experiment runs in a directory (default `./ac_runs`).

```python
tracker = Tracker("ac_runs", experiment="churn")
with tracker.start_run("lgbm-v2") as run:
    run.log_automl(model, train, test)
tracker.runs()          # every run as a table: params, latest metrics, status
```

**Constructor**

```python
Tracker(root='ac_runs', experiment: str='default')
```

#### `Tracker.start_run()`

```python
start_run(name: str | None=None, *, experiment: str | None=None, tags: dict | None=None, params: dict | None=None) -> Run
```

#### `Tracker.get_run()`

```python
get_run(run_id: str) -> Run
```

#### `Tracker.experiments()`

```python
experiments() -> list
```

#### `Tracker.runs()`

```python
runs(experiment: str | None=None, status: str | None=None) -> pl.DataFrame
```

All runs (newest first): id, name, status, timings, `param.*` and latest `metric.*`.

#### `Tracker.best_run()`

```python
best_run(metric: str, experiment: str | None=None, mode: str='max') -> Run
```

The finished run with the best latest value of `metric` (`mode` max or min).

#### `Tracker.compare()`

```python
compare(run_ids: list) -> pl.DataFrame
```

Params and metrics of several runs side by side (one column per run).
