# AutoML: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [AutoML guide](../guide/automl.md).

**On this page:**

- [`mlpilot.automl`](#mlpilotautoml): `load_model`, `automl`
- [`mlpilot.automl.automl`](#mlpilotautomlautoml): `AutoML`
- [`mlpilot.automl.cluster`](#mlpilotautomlcluster): `AutoCluster`
- [`mlpilot.automl.frame`](#mlpilotautomlframe): `infer_kinds`, `to_model_frame`, `make_preprocessor`
- [`mlpilot.automl.models`](#mlpilotautomlmodels): `ModelSpec`, `sample_params`, `model_specs`, `available_models`
- [`mlpilot.automl.tasks`](#mlpilotautomltasks): `TaskInfo`, `detect_task`

---

## `mlpilot.automl`

<sub>Source: [`mlpilot/automl/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/automl/__init__.py)</sub>

AutoML: module 5 of the platform.

* `AutoML(target)`   classification / regression (incl. time series): task detection,
  leakage check, cleaning + feature engineering, model search with
  cross-validation and a time budget, ensembling, evaluation, save/load.
* `AutoCluster()`    unlabelled data: choose the clustering method and number of clusters.
* `automl(df, target=None)`  one call that picks the right one.
* `detect_task`      what kind of problem a dataset is.

### `load_model()`

```python
def load_model(path, *, public_key=None)
```

Load any saved AutoML / AutoCluster model directory (the checksum is verified first).

With `public_key` (or the `AC_MODEL_PUBLIC_KEY` environment variable) the model's
Ed25519 signature is verified before anything is unpickled (see `security.sign_model`).

### `automl()`

```python
def automl(df, target: str | None=None, **kwargs)
```

Fit `AutoML(target)` when a target is given, otherwise `AutoCluster()`.

---

## `mlpilot.automl.automl`

<sub>Source: [`mlpilot/automl/automl.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/automl/automl.py)</sub>

AutoML for supervised problems: classification and regression, including time series.

### class `AutoML`

```python
class AutoML:
```

Automatic model selection, tuning and ensembling for one target column.

`AutoML(target).fit(df)`:

1. detects the task (binary / multiclass / regression; time series when rows are ordered
   by a date column) and picks a metric (ROC-AUC, macro-F1, RMSE) unless `metric` is set;
2. checks for target leakage and drops leaky columns (`on_leakage`);
3. cleans and engineers features with a target-aware pipeline (`preprocess="auto"`:
   `default_cleaning_pipeline` + `default_feature_pipeline`), or yours, or none;
4. cross-validates a baseline and every candidate model with sensible defaults, then
   spends the rest of `time_budget` on random hyperparameter search around the best
  model families (stratified K-fold; time-ordered splits for time series);
5. tries a soft-voting ensemble of the best three model families;
6. refits the winner on all training rows.

Then `predict`, `predict_proba`, `evaluate`, `feature_importance`, `save`/`load`.

**Constructor**

```python
AutoML(target: str, *, task: str='auto', time_col='auto', metric: str | None=None, time_budget: float=60.0, max_trials: int | None=None, cv: int=5, preprocess='auto', include=None, exclude=None, ensemble: bool=True, on_leakage: str='drop', search_sample: int=50000, seed: int=42, verbose: bool=True, n_jobs: int=1, search: str='random', imbalance: str | None='auto', conformal: bool=True)
```

#### `AutoML.is_classification` *(property)*

```python
is_classification: bool
```

#### `AutoML.fit()`

```python
fit(df) -> 'AutoML'
```

#### `AutoML.predict()`

```python
predict(df) -> pl.Series
```

Predicted labels (classification) or values (regression).

#### `AutoML.predict_proba()`

```python
predict_proba(df) -> pl.DataFrame
```

Class probabilities, one `proba_<class>` column per class.

#### `AutoML.predict_interval()`

```python
predict_interval(df, coverage: float=0.9) -> pl.DataFrame
```

Regression: `prediction, lower, upper` such that about `coverage` of true values
fall inside (split-conformal on cross-validation residuals).

#### `AutoML.predict_set()`

```python
predict_set(df, coverage: float=0.9) -> pl.Series
```

Classification: the set of labels that contains the true one with probability
`coverage` (larger sets = the model is less sure).

#### `AutoML.predict_with_proba()`

```python
predict_with_proba(df)
```

`(labels, probabilities)` from one preprocessing pass (classification), or
`(values, None)` for regression — what a serving endpoint needs, at half the cost of
calling `predict` and `predict_proba` separately.

#### `AutoML.evaluate()`

```python
evaluate(df) -> dict
```

Metrics on labelled data (e.g. a held-out test set).

#### `AutoML.feature_importance()`

```python
feature_importance(df=None, n_repeats: int=5) -> pl.DataFrame
```

Permutation importance: how much the score drops when each feature is shuffled.

Uses `df` (labelled) if given, else a sample of the training rows.

#### `AutoML.set_inference_threads()`

```python
set_inference_threads(n: int=1) -> 'AutoML'
```

Threads each fitted estimator uses to predict. Defaults to 1 after fit/load: for the
small batches a REST API scores, starting a thread pool per call costs far more than it
saves (~80 ms per single-row request with a random forest). Use -1 for large batch jobs.

#### `AutoML.report()`

```python
report(df, **kw)
```

Detailed held-out metrics: an `EvaluationReport`.

#### `AutoML.explain()`

```python
explain(df, **kw)
```

SHAP values for rows of `df`: an `Explanation`.

#### `AutoML.error_analysis()`

```python
error_analysis(df, **kw)
```

Worst predictions and failing data segments: an `ErrorReport`.

#### `AutoML.fairness()`

```python
fairness(df, sensitive, **kw)
```

Bias checks across groups of the `sensitive` column(s): a `FairnessReport`.

#### `AutoML.model_card()`

```python
model_card(test_df=None, **kw)
```

An auto-generated `ModelCard` (`.save("card.html")` / `.md` / `.json`).

#### `AutoML.leaderboard_` *(property)*

```python
leaderboard_: pl.DataFrame
```

Every trial, best first (`score` is higher-is-better; error metrics are negative).

#### `AutoML.summary()`

```python
summary() -> str
```

#### `AutoML.save()`

```python
save(path)
```

Save to a directory (manifest.json + model.joblib + preprocess.json).

#### `AutoML.load()` *(classmethod)*

```python
load(path) -> 'AutoML'
```

Load a model saved with `save` (verifies the model checksum first).

---

## `mlpilot.automl.cluster`

<sub>Source: [`mlpilot/automl/cluster.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/automl/cluster.py)</sub>

AutoML for unlabelled data: choose a clustering method and number of clusters.

### class `AutoCluster`

```python
class AutoCluster:
```

Pick the best clustering method and number of clusters by silhouette score.

Tries each method in `methods` (k-means, Gaussian mixture) for every `k` in
`k_range` on a sample, scores with the silhouette coefficient (also reporting
Calinski–Harabasz and Davies–Bouldin), then refits the winner on all rows. Inputs are
cleaned, dates expanded and categories one-hot encoded, and everything is standardised
so no single column dominates the distances.

**Constructor**

```python
AutoCluster(*, k_range=(2, 8), methods=METHODS, preprocess='auto', sample: int=5000, seed: int=42, verbose: bool=True)
```

#### `AutoCluster.fit()`

```python
fit(df) -> 'AutoCluster'
```

#### `AutoCluster.predict()`

```python
predict(df) -> pl.Series
```

Cluster label (0..k-1) for each row.

#### `AutoCluster.cluster_profiles()`

```python
cluster_profiles(df) -> pl.DataFrame
```

Size, share and the mean of every numeric column per cluster.

#### `AutoCluster.leaderboard_` *(property)*

```python
leaderboard_: pl.DataFrame
```

#### `AutoCluster.summary()`

```python
summary() -> str
```

#### `AutoCluster.save()`

```python
save(path)
```

#### `AutoCluster.load()` *(classmethod)*

```python
load(path) -> 'AutoCluster'
```

---

## `mlpilot.automl.frame`

<sub>Source: [`mlpilot/automl/frame.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/automl/frame.py)</sub>

Turning a Polars frame into a model-ready matrix.

### `infer_kinds()`

```python
def infer_kinds(df: pl.DataFrame) -> dict
```

`{column: "num" | "cat"}` for every column a model can use (others are ignored).

### `to_model_frame()`

```python
def to_model_frame(df: pl.DataFrame, kinds: dict)
```

pandas frame with exactly `kinds`' columns: numbers as floats, text with a MISSING token.

Columns missing from `df` are added empty, so prediction-time data always matches the
columns the model was trained on.

### `make_preprocessor()`

```python
def make_preprocessor(kinds: dict, scale: bool)
```

Impute (+ scale) numbers, one-hot encode text; unknown categories are ignored.

---

## `mlpilot.automl.models`

<sub>Source: [`mlpilot/automl/models.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/automl/models.py)</sub>

Candidate models and their hyperparameter search spaces.

### class `ModelSpec`

```python
class ModelSpec:
```

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `build` | `Callable` | required |
| `space` | `dict` | required |
| `defaults` | `dict` | required |
| `scale` | `bool` | `False` |
| `max_rows` | `int \| None` | `None` |
| `baseline` | `bool` | `False` |
| `opt_in` | `bool` | `False` |

### `sample_params()`

```python
def sample_params(space: dict, rng: np.random.Generator) -> dict
```

### `model_specs()`

```python
def model_specs(task: str) -> list
```

Candidate models for `task` (binary | multiclass | regression), cheapest first.

### `available_models()`

```python
def available_models(task: str='binary') -> list
```

Names of the models AutoML can try for `task` in this environment (`mlp` only runs
when named in `include=`).

---

## `mlpilot.automl.tasks`

<sub>Source: [`mlpilot/automl/tasks.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/automl/tasks.py)</sub>

Problem-type detection.

### class `TaskInfo`

```python
class TaskInfo:
```

| Field | Type | Default |
|---|---|---|
| `task` | `str` | required |
| `time_series` | `bool` | required |
| `time_col` | `str \| None` | required |
| `n_classes` | `int \| None` | required |
| `reason` | `str` | required |

#### `TaskInfo.is_classification` *(property)*

```python
is_classification: bool
```

### `detect_task()`

```python
def detect_task(df, target: str | None=None, *, time_col='auto') -> TaskInfo
```

Work out what kind of problem `df` + `target` is.

* no target -> `clustering`;
* two distinct target values -> `binary`; text labels or ≤ 10 whole-number values ->
  `multiclass`; otherwise `regression`;
* `time_series` is True when `time_col` is given, or (`"auto"`) when a date column is
  sorted and mostly unique, meaning rows form a time sequence. Time series get
  time-ordered cross-validation so the model is never tested on the past.
