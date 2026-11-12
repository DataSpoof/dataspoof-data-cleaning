# Evaluation & explainability: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Evaluation & explainability guide](../guide/evaluation.md).

**On this page:**

- [`mlpilot.evaluation`](#mlpilotevaluation)
- [`mlpilot.evaluation.adapter`](#mlpilotevaluationadapter): `as_polars`, `ModelAdapter`
- [`mlpilot.evaluation.card`](#mlpilotevaluationcard): `ModelCard`, `model_card`
- [`mlpilot.evaluation.errors`](#mlpilotevaluationerrors): `ErrorReport`, `error_analysis`
- [`mlpilot.evaluation.explain`](#mlpilotevaluationexplain): `Explanation`, `shapley_values`, `explain`, `partial_dependence`
- [`mlpilot.evaluation.fairness`](#mlpilotevaluationfairness): `FairnessResult`, `FairnessReport`, `cramers_v`, `find_proxies`, `fairness_report`
- [`mlpilot.evaluation.metrics`](#mlpilotevaluationmetrics): `EvaluationReport`, `reliability`, `evaluate_model`

---

## `mlpilot.evaluation`

<sub>Source: [`mlpilot/evaluation/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/evaluation/__init__.py)</sub>

Evaluation & explainability: metrics, SHAP explanations, error analysis, fairness, model cards.

Every function takes a fitted `AutoML` model (or a scikit-learn estimator plus
`target=...`) and labelled data:
```
report = ac.evaluate_model(model, test)            # metrics, confusion matrix, calibration
shap = ac.explain(model, test)                     # per-row + global SHAP values
errors = ac.error_analysis(model, test)            # worst predictions & failing segments
fair = ac.fairness_report(model, test, ["Sex"])    # bias across groups
ac.model_card(model, test, sensitive=["Sex"]).save("card.html")
```

---

## `mlpilot.evaluation.adapter`

<sub>Source: [`mlpilot/evaluation/adapter.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/evaluation/adapter.py)</sub>

A uniform view of a fitted model, so every evaluation tool works the same way.

### `as_polars()`

```python
def as_polars(df) -> pl.DataFrame
```

### class `ModelAdapter`

```python
class ModelAdapter:
```

Wraps a fitted `AutoML` model or a scikit-learn estimator.

Internally classes are integer codes `0..k-1` aligned with `classes` (the original
labels); `X(df)` returns the model-input frame (pandas) for a Polars/pandas frame.

**Constructor**

```python
ModelAdapter(*, name, target, task, classes, to_X, predict_raw, proba, automl=None)
```

#### `ModelAdapter.of()` *(classmethod)*

```python
of(model, target: str | None=None) -> 'ModelAdapter'
```

#### `ModelAdapter.is_classification` *(property)*

```python
is_classification: bool
```

#### `ModelAdapter.X()`

```python
X(df)
```

#### `ModelAdapter.predict()`

```python
predict(X) -> np.ndarray
```

Class codes (classification) or values (regression) for a model-input frame.

#### `ModelAdapter.proba()`

```python
proba(X) -> np.ndarray
```

#### `ModelAdapter.y()`

```python
y(df) -> np.ndarray
```

True target as class codes (classification) or floats (regression).

#### `ModelAdapter.positive_index()`

```python
positive_index(pos_label=None) -> int
```

Index of the positive class for a binary model (default: the second sorted class).

#### `ModelAdapter.class_index()`

```python
class_index(label) -> int
```

#### `ModelAdapter.output_name()`

```python
output_name(pos: int | None=None) -> str
```

#### `ModelAdapter.labelled()`

```python
labelled(df) -> pl.DataFrame
```

Rows with a known target (unlabelled rows can't be evaluated).

---

## `mlpilot.evaluation.card`

<sub>Source: [`mlpilot/evaluation/card.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/evaluation/card.py)</sub>

Auto-generated model cards (Mitchell et al., 2019) in Markdown, HTML and JSON.

### class `ModelCard`

```python
class ModelCard:
```

A model card: `to_markdown()`, `to_html()`, `to_dict()` / `to_json()`, `save(path)`.

| Field | Type | Default |
|---|---|---|
| `data` | `dict` | required |

#### `ModelCard.limitations` *(property)*

```python
limitations: list
```

#### `ModelCard.ethical_considerations` *(property)*

```python
ethical_considerations: list
```

#### `ModelCard.to_dict()`

```python
to_dict() -> dict
```

#### `ModelCard.to_json()`

```python
to_json(indent: int=2) -> str
```

#### `ModelCard.to_markdown()`

```python
to_markdown() -> str
```

#### `ModelCard.to_html()`

```python
to_html() -> str
```

#### `ModelCard.save()`

```python
save(path) -> Path
```

Write to `.md`, `.html` or `.json` (chosen by the file extension).

### `model_card()`

```python
def model_card(model, test_df=None, *, target: str | None=None, sensitive=None, name: str | None=None, version: str='1.0', description: str | None=None, intended_use: str | None=None, out_of_scope: str | None=None, owners=None, limitations=None, ethical_considerations=None, pos_label=None, explain_rows: int=100, quality: bool=True, seed: int=42) -> ModelCard
```

Generate a model card for a fitted `AutoML` model (or scikit-learn estimator + `target`).

Model details, training data, cross-validation leaderboard and settings come from the
model itself. With labelled `test_df` the card adds held-out metrics with confidence
intervals, calibration, SHAP feature importance, error analysis, a data-quality score,
and — given `sensitive` columns — a fairness review. Limitations and ethical
considerations are generated from what was found; add your own with `limitations` /
`ethical_considerations` and describe the purpose with `intended_use`.

---

## `mlpilot.evaluation.errors`

<sub>Source: [`mlpilot/evaluation/errors.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/evaluation/errors.py)</sub>

Error analysis: which predictions are worst, and which segments of the data the model fails on.

### class `ErrorReport`

```python
class ErrorReport:
```

Where a model goes wrong.

`slices` ranks data segments by excess error versus the whole dataset; a slice is
`significant` when it is worse than the rest beyond chance (one-sided test, Benjamini–
Hochberg corrected across all slices tested). `worst` lists the individual worst
predictions (most confident mistakes / largest errors) with their input values.

| Field | Type | Default |
|---|---|---|
| `task` | `str` | required |
| `model` | `str` | required |
| `target` | `str` | required |
| `n_rows` | `int` | required |
| `metric` | `str` | required |
| `overall` | `float` | required |
| `min_size` | `int` | required |
| `slices` | `pl.DataFrame` | required |
| `worst` | `pl.DataFrame` | required |
| `confusions` | `pl.DataFrame \| None` | `None` |

#### `ErrorReport.problem_slices` *(property)*

```python
problem_slices: pl.DataFrame
```

#### `ErrorReport.summary()`

```python
summary(top: int=8) -> str
```

#### `ErrorReport.to_dict()`

```python
to_dict() -> dict
```

#### `ErrorReport.to_json()`

```python
to_json(indent: int=2) -> str
```

### `error_analysis()`

```python
def error_analysis(model, df, *, target: str | None=None, columns=None, top: int=20, min_size: int | None=None, bins: int=5, max_categories: int=20, intersections: bool=True, alpha: float=0.05) -> ErrorReport
```

Find the worst predictions and the data segments where the model underperforms.

Segments come from the *original* columns of `df` (`columns` to restrict them): one
per category, quantile bins for numbers, and missing values; with `intersections` the
worst segments are also crossed in pairs ("Sex = male & Pclass = 3"). Segments smaller
than `min_size` rows (default: max(10, 1% of rows)) are ignored.

---

## `mlpilot.evaluation.explain`

<sub>Source: [`mlpilot/evaluation/explain.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/evaluation/explain.py)</sub>

SHAP explanations: per-row and global feature contributions, plus partial dependence.

Values are interventional Shapley values against a background sample: for every explained
row, `base_value + sum(values) == model output` exactly. With up to `exact_max_features` (8)
features every coalition is evaluated (exact Shapley); otherwise antithetic permutation
sampling is used, which is unbiased and still exactly additive. Works with any model,
including ensembles and the full AutoML preprocessing, because it only calls the model.

### class `Explanation`

```python
class Explanation:
```

Shapley values for a set of rows.

`values` is `(rows, features)` for regression and binary models (contributions to
the predicted value / positive-class probability) and `(rows, features, classes)`
for multiclass models (contributions to each class probability).

| Field | Type | Default |
|---|---|---|
| `values` | `np.ndarray` | required |
| `base_value` | `np.ndarray \| float` | required |
| `features` | `list` | required |
| `data` | `pd.DataFrame` | required |
| `output` | `str` | required |
| `method` | `str` | required |
| `classes` | `list \| None` | `None` |
| `positive_class` | `object` | `None` |

#### `Explanation.is_multiclass` *(property)*

```python
is_multiclass: bool
```

#### `Explanation.predictions` *(property)*

```python
predictions: np.ndarray
```

Model output reconstructed from the explanation (base + contributions).

#### `Explanation.global_importance()`

```python
global_importance(class_=None) -> pl.DataFrame
```

Mean |SHAP| per feature (the share of total attribution), with the direction of effect.

`direction` is the rank correlation between a numeric feature's value and its
contribution: positive means higher values push the output up.

#### `Explanation.for_row()`

```python
for_row(i: int, class_=None, top: int | None=None) -> pl.DataFrame
```

Contributions for one explained row, largest first.

For multiclass models the predicted class is explained unless `class_` is given.

#### `Explanation.explain_row()`

```python
explain_row(i: int, class_=None, top: int=5) -> str
```

A one-paragraph, human-readable explanation of one prediction.

#### `Explanation.to_frame()`

```python
to_frame(class_=None) -> pl.DataFrame
```

One row per explained row, one column per feature (`shap_<feature>`).

#### `Explanation.to_shap()`

```python
to_shap()
```

The same values as a `shap.Explanation` (requires `pip install shap`), for shap's plots.

#### `Explanation.summary()`

```python
summary(top: int=10) -> str
```

### `shapley_values()`

```python
def shapley_values(f, X: pd.DataFrame, background: pd.DataFrame, *, exact_max_features: int=8, n_permutations: int=10, seed: int=42)
```

Shapley values of `f` (frame -> `(n, k)` outputs) for every row of `X`.

Returns `(values (n, p, k), base (k,), method)`.

### `explain()`

```python
def explain(model, df, *, target: str | None=None, rows: int | None=100, background=None, n_background: int=20, n_permutations: int=10, exact_max_features: int=8, pos_label=None, seed: int=42) -> Explanation
```

SHAP values for (up to `rows` randomly chosen) rows of `df`.

Features are the model's inputs after AutoML preprocessing (e.g. one-hot columns made by
the feature pipeline). `background` (default: a sample of `df`) is the reference the
contributions are measured against; its mean output is the `base_value`.
Pass `rows=None` to explain every row.

### `partial_dependence()`

```python
def partial_dependence(model, df, feature: str, *, target: str | None=None, grid: int=20, sample: int=500, pos_label=None, seed: int=42) -> pl.DataFrame
```

Average model output as one feature is swept across its range (others left as they are).

Numeric features use `grid` quantile points; categorical features use their values.

---

## `mlpilot.evaluation.fairness`

<sub>Source: [`mlpilot/evaluation/fairness.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/evaluation/fairness.py)</sub>

Fairness and bias checks across groups of a sensitive attribute (sex, age band, region …).

### class `FairnessResult`

```python
class FairnessResult:
```

Fairness of a model with respect to one sensitive attribute (or an intersection).

| Field | Type | Default |
|---|---|---|
| `attribute` | `str` | required |
| `groups` | `pl.DataFrame` | required |
| `disparities` | `dict` | required |
| `flags` | `list` | required |
| `reference_group` | `str \| None` | required |
| `excluded_groups` | `list` | required |
| `used_by_model` | `bool` | required |
| `notes` | `list` | required |

#### `FairnessResult.passed` *(property)*

```python
passed: bool
```

### class `FairnessReport`

```python
class FairnessReport:
```

| Field | Type | Default |
|---|---|---|
| `task` | `str` | required |
| `model` | `str` | required |
| `target` | `str` | required |
| `n_rows` | `int` | required |
| `positive_class` | `object` | required |
| `results` | `list` | required |
| `proxies` | `pl.DataFrame` | required |
| `limits` | `dict` | required |

#### `FairnessReport.passed` *(property)*

```python
passed: bool
```

#### `FairnessReport.flags` *(property)*

```python
flags: list
```

#### `FairnessReport.summary()`

```python
summary() -> str
```

#### `FairnessReport.to_dict()`

```python
to_dict() -> dict
```

#### `FairnessReport.to_json()`

```python
to_json(indent: int=2) -> str
```

### `cramers_v()`

```python
def cramers_v(a: np.ndarray, b: np.ndarray) -> float
```

Bias-corrected Cramér's V between two label arrays (0 = unrelated, 1 = identical).

### `find_proxies()`

```python
def find_proxies(df: pl.DataFrame, sensitive: list, *, exclude=(), threshold: float=0.5, bins: int=5) -> pl.DataFrame
```

Features strongly associated with a sensitive attribute (Cramér's V on binned values).

A proxy lets a model discriminate even when the attribute itself is removed.

### `fairness_report()`

```python
def fairness_report(model, df, sensitive, *, target: str | None=None, pos_label=None, threshold: float | None=None, min_group_size: int=20, bins: int=4, intersectional: bool=True, reference=None, limits: dict | None=None, proxy_threshold: float=0.5) -> FairnessReport
```

Check a model for bias across the groups of each `sensitive` column.

Classification (binary, or multiclass with `pos_label`): per-group selection rate,
TPR, FPR, precision, accuracy and AUC; flags demographic parity, the four-fifths
(disparate impact) rule, equal opportunity, equalized odds, predictive parity and
accuracy gaps beyond `limits`. Regression: per-group error and bias.
Numeric attributes (e.g. age) are cut into `bins` quantile bands. Groups smaller than
`min_group_size` are reported but not compared. With two or more attributes and
`intersectional`, their combinations are checked too (e.g. sex × age band). Also
lists features that act as **proxies** for a sensitive attribute.

---

## `mlpilot.evaluation.metrics`

<sub>Source: [`mlpilot/evaluation/metrics.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/evaluation/metrics.py)</sub>

Detailed evaluation of a fitted model on labelled data.

### class `EvaluationReport`

```python
class EvaluationReport:
```

Everything about how well a model does on one labelled dataset.

| Field | Type | Default |
|---|---|---|
| `task` | `str` | required |
| `model` | `str` | required |
| `target` | `str` | required |
| `n_rows` | `int` | required |
| `metrics` | `dict` | required |
| `confidence_intervals` | `dict` | `dict()` |
| `classes` | `list \| None` | `None` |
| `positive_class` | `object` | `None` |
| `threshold` | `float \| None` | `None` |
| `confusion_matrix` | `pl.DataFrame \| None` | `None` |
| `per_class` | `pl.DataFrame \| None` | `None` |
| `calibration` | `pl.DataFrame \| None` | `None` |
| `roc_curve` | `pl.DataFrame \| None` | `None` |
| `pr_curve` | `pl.DataFrame \| None` | `None` |
| `thresholds` | `pl.DataFrame \| None` | `None` |
| `residuals` | `dict \| None` | `None` |
| `error_by_range` | `pl.DataFrame \| None` | `None` |

#### `EvaluationReport.is_classification` *(property)*

```python
is_classification: bool
```

#### `EvaluationReport.best_threshold()`

```python
best_threshold(by: str='f1') -> float
```

Decision threshold that maximises `f1`, `youden` (TPR − FPR) or `accuracy`.

#### `EvaluationReport.summary()`

```python
summary() -> str
```

#### `EvaluationReport.to_dict()`

```python
to_dict(curves: bool=True) -> dict
```

#### `EvaluationReport.to_json()`

```python
to_json(path=None, indent: int=2) -> str
```

### `reliability()`

```python
def reliability(confidence: np.ndarray, correct: np.ndarray, bins: int=10)
```

Calibration table (equal-width bins) and the expected calibration error.

### `evaluate_model()`

```python
def evaluate_model(model, df, *, target: str | None=None, threshold: float | None=None, pos_label=None, bootstrap: int=200, seed: int=42) -> EvaluationReport
```

Detailed metrics for a fitted `AutoML` model (or scikit-learn estimator + `target`).

Classification: accuracy, balanced accuracy, precision/recall/F1 (macro, weighted and
per class), MCC, Cohen's kappa, ROC-AUC, PR-AUC, log loss, Brier score, a confusion matrix,
a calibration table with the expected calibration error (ECE), and — for binary models —
ROC/PR curves and a threshold table. `threshold` changes the binary decision cut-off.

Regression: RMSE, MAE, median AE, max error, R², explained variance, bias, MAPE/sMAPE,
residual distribution and error by range of the actual value.

Headline metrics get 95% bootstrap confidence intervals (`bootstrap` resamples; 0 = off).
