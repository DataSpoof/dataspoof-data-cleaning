# Model security: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Model security guide](../guide/model-security.md).

**On this page:**

- [`mlpilot.poisoning`](#mlpilotpoisoning): `PoisoningReport`, `poisoning_report`
- [`mlpilot.robustness`](#mlpilotrobustness): `RobustnessReport`, `robustness_report`

---

## `mlpilot.poisoning`

<sub>Source: [`mlpilot/poisoning.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/poisoning.py)</sub>

Model security, part 2: is the training data poisoned or mislabelled?

```python
rep = poisoning_report(train_df, target="fraud")
print(rep.summary())
rep.suspicious                    # row index, score, reasons - review these first
clean = rep.clean(train_df)       # drop the flagged rows (or rep.clean(train_df, min_score=...))
```

Checks (no model needed; works for classification and regression targets):

* **Label issues** (classification, *confident learning*): out-of-fold probabilities from a
  quick model; a row is suspect when the model is confidently sure it belongs to another class
  (P(given label) below that class's average self-confidence and another class above its own).
  Catches label-flipping attacks and ordinary annotation mistakes.
* **Conflicting duplicates**: identical feature values with different labels.
* **Backdoor triggers**: a rare value (or a numeric value repeated far more than its
  neighbours) whose rows almost all carry one label - the signature of a planted trigger.
  Tested with a one-sided binomial test, corrected for the number of candidates (Holm).
* **Outliers**: Isolation Forest on numeric features - injected points often sit far from the
  real data.
* **Regression**: rows whose target is extreme given out-of-fold predictions (robust z > 4).

### class `PoisoningReport`

```python
class PoisoningReport:
```

| Field | Type | Default |
|---|---|---|
| `rows` | `int` | required |
| `task` | `str` | required |
| `suspicious` | `pl.DataFrame` | required |
| `label_issues` | `int` | required |
| `conflicting_duplicates` | `int` | required |
| `triggers` | `pl.DataFrame` | required |
| `outliers` | `int` | required |

#### `PoisoningReport.clean()`

```python
clean(df, *, min_score: float=0.0) -> pl.DataFrame
```

`df` without the flagged rows (those scoring above `min_score`).

#### `PoisoningReport.summary()`

```python
summary() -> str
```

### `poisoning_report()`

```python
def poisoning_report(df, target: str, *, cv: int=5, seed: int=0, contamination: float=0.01, trigger_max_support: float=0.05, trigger_min_rows: int=8, alpha: float=0.01, classification: bool | None=None) -> PoisoningReport
```

Run all checks; see the module docstring. `trigger_max_support` is the largest share of
rows a trigger value may cover; `contamination` the expected share of outliers.

---

## `mlpilot.robustness`

<sub>Source: [`mlpilot/robustness.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/robustness.py)</sub>

Model security, part 1: how easily can a model's predictions be pushed around?

```python
rep = robustness_report(model, test_df)          # AutoML model, or a sklearn model with target=
print(rep.summary())
rep.features                                     # per-feature sensitivity table
rep.passed(max_noise_flip=0.05, min_median_distance=0.25)
```

Black-box tests (only `predict`/`predict_proba` are used, so any model works, and the full
preprocessing is part of what is attacked):

* **Noise stability**: numeric inputs get Gaussian noise of 1%, 5% and 10% of each column's
  standard deviation; how often does the prediction change?
* **Minimal adversarial perturbation** (evasion attack): for each sampled row, a greedy
  coordinate search moves the most effective numeric feature step by step until the predicted
  class flips (classification) or the prediction moves by `regression_shift` standard
  deviations. The distance needed is reported in standard-deviation units (L-infinity); a model
  that flips at 0.05 sd can be fooled by changes nobody would notice.
* **Feature sensitivity**: flip rate when one feature is replaced by another observed value
  (categoricals), shifted by +-1 sd (numerics) or set to missing.
* **Extreme inputs**: values far outside the training range must not crash the model or produce
  NaN / infinite outputs.

### class `RobustnessReport`

```python
class RobustnessReport:
```

| Field | Type | Default |
|---|---|---|
| `task` | `str` | required |
| `rows` | `int` | required |
| `noise` | `dict` | required |
| `attack` | `dict` | required |
| `features` | `pl.DataFrame` | required |
| `extreme` | `dict` | required |
| `notes` | `list` | `list()` |

#### `RobustnessReport.passed()`

```python
passed(*, max_noise_flip: float=0.05, min_median_distance: float=0.25) -> bool
```

True when 5%-noise flips at most `max_noise_flip` of predictions, the median
adversarial distance is at least `min_median_distance` sd, and extreme inputs are handled.

#### `RobustnessReport.summary()`

```python
summary() -> str
```

### `robustness_report()`

```python
def robustness_report(model, df, *, target: str | None=None, n: int=200, noise=(0.01, 0.05, 0.1), budget: float=2.0, step: float=0.05, regression_shift: float=0.5, seed: int=0, features: list | None=None) -> RobustnessReport
```

Run the black-box robustness tests on up to `n` rows of `df` (see the module docstring).
`budget` caps the adversarial search (in sd units); `step` is its resolution.
