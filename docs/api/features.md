# Feature engineering: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Feature engineering guide](../guide/features.md).

**On this page:**

- [`mlpilot.features`](#mlpilotfeatures)
- [`mlpilot.features.dates`](#mlpilotfeaturesdates): `DateTimeFeatures`
- [`mlpilot.features.defaults`](#mlpilotfeaturesdefaults): `default_feature_pipeline`
- [`mlpilot.features.encoding`](#mlpilotfeaturesencoding): `TargetEncode`
- [`mlpilot.features.generators`](#mlpilotfeaturesgenerators): `Interactions`, `GroupAggregations`, `LagFeatures`
- [`mlpilot.features.leakage`](#mlpilotfeaturesleakage): `LeakageError`, `LeakageFinding`, `LeakageReport`, `detect_leakage`, `DropLeakage`
- [`mlpilot.features.selection`](#mlpilotfeaturesselection): `feature_matrix`, `target_vector`, `SelectFeatures`
- [`mlpilot.features.text`](#mlpilotfeaturestext): `TextEmbeddings`

---

## `mlpilot.features`

<sub>Source: [`mlpilot/features/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/features/__init__.py)</sub>

Feature engineering: module 4 of the platform, built on the shared pipeline core.

Every class is a pipeline `Step` (fit on training data, replay on new data, saved as JSON):

* `DateTimeFeatures`   calendar parts, cyclical encodings, elapsed days
* `TargetEncode`       leakage-safe (out-of-fold) target encoding
* `Interactions`       products / ratios of the most informative numeric columns
* `GroupAggregations`  per-group statistics (mean, std, count, ...)
* `LagFeatures`        time-series lags and rolling means
* `TextEmbeddings`     TF-IDF + SVD (built in) or sentence-transformers vectors
* `SelectFeatures`     importance / mutual-information / variance selection
* `DropLeakage`        detect and drop columns that leak the target
  (`detect_leakage(df, target)` runs the check on its own)

`default_feature_pipeline(target)` wires them together. One-hot / ordinal / frequency
encoding and rare-category grouping come from the cleaning module (`EncodeCategorical`,
`GroupRareCategories`).

---

## `mlpilot.features.dates`

<sub>Source: [`mlpilot/features/dates.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/features/dates.py)</sub>

Date/time feature extraction.

### class `DateTimeFeatures`

```python
class DateTimeFeatures(Step):
```

Expand date/datetime columns into calendar parts, cyclical encodings and elapsed days.

Outputs `<col>_year`, `_month`, `_day`, `_dayofweek` (1 = Monday), `_dayofyear`,
`_quarter`, `_is_weekend`, `_is_month_start`, `_is_month_end`, `_hour` (only for
columns that carry a time of day), sine/cosine encodings of month, weekday and hour (so
December sits next to January), and `_elapsed_days` since the earliest training date.

**Constructor**

```python
DateTimeFeatures(*, columns=None, parts=None, drop_original=True, fit_only=None)
```

---

## `mlpilot.features.defaults`

<sub>Source: [`mlpilot/features/defaults.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/features/defaults.py)</sub>

The default, target-aware feature-engineering pipeline.

### `default_feature_pipeline()`

```python
def default_feature_pipeline(target: str, *, time_col: str | None=None, name: str='feature-engineering', on_schema_mismatch: str='warn', verbose: bool=False, audit_log=None, **overrides) -> Pipeline
```

The recommended feature pipeline for a supervised problem.

Steps: (time-series lags) → date/time features → drop leaky columns → embed free text → target-encode
high-cardinality categories (≥ 16 values, out-of-fold) → one-hot encode the rest →
select features by importance. Off by default: `interactions`, `group_aggregations`.

With `time_col` (time series), target lags/rolling means are added and target encoding is
turned off, because random folds would let future rows inform past ones.

Override any step by name: `False` removes it, `True` enables an optional one, a dict
enables it with those parameters, e.g. `interactions={"max_features": 4}`.

---

## `mlpilot.features.encoding`

<sub>Source: [`mlpilot/features/encoding.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/features/encoding.py)</sub>

Leakage-safe target encoding.

### class `TargetEncode`

```python
class TargetEncode(Step):
```

Replace categories with the (smoothed) average target value of that category.

Leakage-safe: while fitting, each training row is encoded with statistics from the
*other* folds (out-of-fold), so a row never sees its own label. New data uses statistics
from all training rows. Unseen or missing categories get the overall average.

* numeric / binary targets: one column with the mean (binary: share of the positive class,
  the later of the two labels in sorted order);
* multiclass targets: one `<col>_te_<class>` column per class (up to `max_classes`).

`smoothing` blends rare categories towards the overall average:
`(n·mean + smoothing·prior) / (n + smoothing)`.

**Constructor**

```python
TargetEncode(*, target=None, columns=None, min_unique=0, max_unique_ratio=0.9, smoothing=10.0, n_folds=5, max_classes=10, seed=42, fit_only=None)
```

#### `TargetEncode.fit_transform()`

```python
fit_transform(df)
```

Fit, then encode the training rows out-of-fold so no row sees its own label.

---

## `mlpilot.features.generators`

<sub>Source: [`mlpilot/features/generators.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/features/generators.py)</sub>

Feature generators: interactions, group aggregations, and time-series lags/rolling windows.

### class `Interactions`

```python
class Interactions(Step):
```

Pairwise products and ratios of the most informative numeric columns.

Picks the `max_features` numeric columns most correlated with a numeric target (or with
the highest variance when there's no numeric target) and adds `a_x_b` (product) and
`a_div_b` (ratio; null when dividing by zero) for every pair.

**Constructor**

```python
Interactions(*, columns=None, max_features=6, operations=('multiply', 'ratio'), target=None, fit_only=None)
```

### class `GroupAggregations`

```python
class GroupAggregations(Step):
```

Add per-group statistics of numeric columns, e.g. the average fare of a passenger's class.

For each grouping column (auto: up to `max_keys` text columns with 2–50 distinct values)
the learned table adds `<key>_count` and `<col>_<agg>_by_<key>` columns. The target is
never aggregated (that would leak it). Unseen groups get nulls.

**Constructor**

```python
GroupAggregations(*, group_by=None, columns=None, aggregations=('mean', 'std'), max_groups=1000, max_keys=3, max_columns=10, target=None, fit_only=None)
```

### class `LagFeatures`

```python
class LagFeatures(Step):
```

Time-series lags and rolling means, computed in time order (optionally per entity).

Adds `<col>_lag<k>` for each lag and `<col>_roll<w>` (mean of the previous `w` values,
excluding the current one, so it never peeks at the present). Lags of the target are
included by default; they're computed from earlier rows, so they're legitimate. Row order
is preserved. To predict new periods, pass recent history (with known targets) together
with the new rows; if the target column is absent, its lag columns are null.

**Constructor**

```python
LagFeatures(*, time_col, columns=None, lags=(1, 2, 3), windows=(3, 7), group_by=None, include_target=True, target=None, fit_only=None)
```

---

## `mlpilot.features.leakage`

<sub>Source: [`mlpilot/features/leakage.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/features/leakage.py)</sub>

Target-leakage detection.

Leakage is information in the features that wouldn't exist at prediction time, typically a
column recorded *after* the outcome (`refund_issued` when predicting churn) or a copy of the
target. It makes models look excellent in testing and fail in production.

`detect_leakage(df, target)` scores every column **on its own** with cross-validation (a
small decision tree for numbers/dates, an out-of-fold category average for text). A single
column that predicts the target almost perfectly (default ≥ 0.98 ROC-AUC / balanced accuracy /
R²) is flagged `high`. Columns whose name contains the target's name are flagged `medium`.

### class `LeakageError`

```python
class LeakageError(Exception):
```

Raised by `DropLeakage(on_detect="raise")`; `.report` holds the LeakageReport.

**Constructor**

```python
LeakageError(report: 'LeakageReport')
```

### class `LeakageFinding`

```python
class LeakageFinding:
```

| Field | Type | Default |
|---|---|---|
| `column` | `str` | required |
| `severity` | `str` | required |
| `reason` | `str` | required |
| `score` | `float \| None` | required |
| `detail` | `str` | required |

### class `LeakageReport`

```python
class LeakageReport:
```

| Field | Type | Default |
|---|---|---|
| `target` | `str` | required |
| `task` | `str` | required |
| `metric` | `str` | required |
| `scores` | `dict` | required |
| `findings` | `list` | required |

#### `LeakageReport.leaky_columns` *(property)*

```python
leaky_columns: list
```

Columns flagged `high`: almost certainly leaking.

#### `LeakageReport.to_dict()`

```python
to_dict() -> dict
```

#### `LeakageReport.to_frame()`

```python
to_frame() -> pl.DataFrame
```

#### `LeakageReport.summary()`

```python
summary() -> str
```

### `detect_leakage()`

```python
def detect_leakage(df, target: str, *, threshold: float=0.98, cv: int=3, sample: int=5000, seed: int=42) -> LeakageReport
```

Score each column alone against `target` and report suspiciously strong ones.

### class `DropLeakage`

```python
class DropLeakage(Step):
```

Detect target leakage while fitting and drop (or warn about / refuse) leaky columns.

`on_detect`: `"drop"` (default) removes `high` findings (plus `medium` ones if
`drop_suspicious_names`); `"warn"` keeps them but warns; `"raise"` stops with a
`LeakageError`. The report is kept in `report_` and the findings in the saved state.

**Constructor**

```python
DropLeakage(*, target=None, threshold=0.98, drop_suspicious_names=False, on_detect='drop', cv=3, sample=5000, seed=42, fit_only=None)
```

---

## `mlpilot.features.selection`

<sub>Source: [`mlpilot/features/selection.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/features/selection.py)</sub>

Automatic feature selection.

### `feature_matrix()`

```python
def feature_matrix(df: pl.DataFrame, cols: list) -> np.ndarray
```

Float matrix of `cols` with nulls filled by the column median (0 if all null).

### `target_vector()`

```python
def target_vector(y: pl.Series)
```

(kind, y as numpy) with labels encoded as integers for classification.

### class `SelectFeatures`

```python
class SelectFeatures(Step):
```

Keep the numeric features that matter; drop the rest.

* `importance` (default): random-forest importance; keep features until their cumulative
  importance reaches `cumulative_importance` (0.95), capped at `max_features`.
* `mutual_info`: mutual information with the target; keep scores above `min_score`
  (top `max_features` if set).
* `variance`: only drop constant columns (no target needed).

Constant columns are always dropped. Columns in `keep`, text columns and the target are
never touched.

**Constructor**

```python
SelectFeatures(*, target=None, method='importance', max_features=None, cumulative_importance=0.95, min_score=0.0, keep=None, sample=20000, seed=42, fit_only=None)
```

---

## `mlpilot.features.text`

<sub>Source: [`mlpilot/features/text.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/features/text.py)</sub>

Text embeddings: dense numeric vectors for free-text columns.

### class `TextEmbeddings`

```python
class TextEmbeddings(Step):
```

Turn free-text columns into `<col>_emb0..N` numeric columns.

`method="tfidf_svd"` (default, no extra dependencies): TF-IDF over the training
vocabulary, compressed with truncated SVD (latent semantic analysis). Everything learned is
stored as plain JSON, so saved pipelines reproduce the same vectors.

`method="sentence_transformers"`: neural sentence embeddings from a pretrained model
(`pip install sentence-transformers`); only the model name is stored.

By default every column whose semantic type is `text` (long, varied strings) is embedded.

`dtype` sets how the vectors are stored: `"float64"` (default), `"float32"` (2x
smaller), `"float16"` (4x) or `"int8"` (8x: each dimension scaled to -127..127 with a
scale learned on the training data, see `mlpilot.compression.quantize_vectors`).

**Constructor**

```python
TextEmbeddings(*, columns=None, method='tfidf_svd', n_components=16, max_features=1000, min_df=2, model='all-MiniLM-L6-v2', drop_original=True, seed=0, dtype='float64', fit_only=None)
```
