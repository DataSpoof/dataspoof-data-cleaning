# Profiling & data quality: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Profiling & data quality guide](../guide/quality.md).

**On this page:**

- [`mlpilot.quality`](#mlpilotquality)
- [`mlpilot.quality.checks`](#mlpilotqualitychecks): `jsonable`, `grade`, `Issue`, `QualityReport`, `check_quality`
- [`mlpilot.quality.contracts`](#mlpilotqualitycontracts): `ContractError`, `ContractViolation`, `RuleResult`, `ContractResult`, `Contract`
- [`mlpilot.quality.pipeline_steps`](#mlpilotqualitypipeline_steps): `QualityWarning`, `QualityGateError`, `ValidateContract`, `QualityGate`
- [`mlpilot.quality.profile`](#mlpilotqualityprofile): `semantic_type`, `ColumnProfile`, `Profile`, `profile`
- [`mlpilot.quality.schema`](#mlpilotqualityschema): `infer_schema`, `SchemaDiff`, `compare_schemas`

---

## `mlpilot.quality`

<sub>Source: [`mlpilot/quality/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/quality/__init__.py)</sub>

Profiling & data quality: module 2 of the platform, built on the shared pipeline core.

* `profile(df)`         automated EDA: per-column stats, semantic types, correlations,
  plain-English insights and the quality report.
* `check_quality(df)`   every data-quality check plus a 0–100 score and grade, split into
  completeness / uniqueness / validity / consistency and per column.
* `Contract`            declarative data contracts (YAML/JSON); `validate` any batch;
  `Contract.infer` writes one from data you trust.
* `compare_schemas`     added / removed / retyped columns between two datasets.
* `ValidateContract` / `QualityGate`   pipeline steps that enforce all of the above.

---

## `mlpilot.quality.checks`

<sub>Source: [`mlpilot/quality/checks.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/quality/checks.py)</sub>

Data-quality checks and a 0–100 quality score.

`check_quality(df)` runs every check and returns a `QualityReport`: the list of issues
(each with a severity, the affected column and how many cells it touches) plus scores.

Scoring
-------
Four standard data-quality dimensions, each 0–100:

* **completeness** – share of cells that hold a real value (nulls, text placeholders such as
  "NA"/"?" and numeric codes such as -999 all count as missing).
* **uniqueness**   – share of rows that are not exact repeats of an earlier row.
* **validity**     – share of rule-checked cells that pass: values inside name-implied ranges
  (age, percent, price ...), valid emails / phone numbers / calendar dates, and numbers stored
  as text without non-numeric blockers. 100 if no rule applied.
* **consistency**  – share of text cells free of formatting drift: stray whitespace, the same
  value spelled with different letter case, and minority date formats in mixed-format
  columns. 100 if there is no text.

The overall score is the weighted mean (default weights: completeness 0.35, validity 0.30,
consistency 0.20, uniqueness 0.15) and maps to a grade: A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60,
else F. Each column also gets a score with the same weights over the dimensions that apply
to it. Because the overall score averages over every cell, one broken column in a wide
table barely moves it, so check `weakest_columns()` as well. Statistical observations (outliers, skew, negatives, rare categories, constant and
ID-like columns) are reported as issues but don't lower the score, because they aren't
necessarily wrong.

### `jsonable()`

```python
def jsonable(obj)
```

Convert to plain JSON types (NaN/inf -> None, dates -> ISO strings).

### `grade()`

```python
def grade(score: float) -> str
```

### class `Issue`

```python
class Issue:
```

One detected data-quality problem.

| Field | Type | Default |
|---|---|---|
| `check` | `str` | required |
| `severity` | `str` | required |
| `title` | `str` | required |
| `metric` | `str` | required |
| `detail` | `str` | required |
| `column` | `str \| None` | `None` |
| `affected` | `int` | `0` |
| `dimension` | `str \| None` | `None` |

#### `Issue.to_dict()`

```python
to_dict() -> dict
```

### class `QualityReport`

```python
class QualityReport:
```

Result of `check_quality`. Print it for a readable summary.

| Field | Type | Default |
|---|---|---|
| `rows` | `int` | required |
| `columns` | `int` | required |
| `score` | `float` | required |
| `grade` | `str` | required |
| `dimensions` | `dict` | required |
| `column_scores` | `dict` | required |
| `issues` | `list` | required |
| `weights` | `dict` | required |

#### `QualityReport.counts` *(property)*

```python
counts: dict
```

#### `QualityReport.weakest_columns()`

```python
weakest_columns(k: int=3, below: float=90.0) -> list
```

Up to `k` `(column, score)` pairs scoring under `below`, worst first.

The overall score averages over every cell, so one broken column in a wide table
barely moves it; check these too.

#### `QualityReport.filter()`

```python
filter(severity=None, check=None, column=None) -> list
```

#### `QualityReport.to_dict()`

```python
to_dict() -> dict
```

#### `QualityReport.to_json()`

```python
to_json(path=None, indent: int=2) -> str
```

#### `QualityReport.to_frame()`

```python
to_frame() -> pl.DataFrame
```

Issues as a DataFrame (one row per issue).

#### `QualityReport.summary()`

```python
summary(top: int=10) -> str
```

### `check_quality()`

```python
def check_quality(df, *, weights: dict | None=None) -> QualityReport
```

Run every data-quality check on `df` and score it. See the module docstring.

---

## `mlpilot.quality.contracts`

<sub>Source: [`mlpilot/quality/contracts.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/quality/contracts.py)</sub>

Data contracts: declare what valid data looks like, then validate every batch against it.

A contract is plain YAML/JSON, so it can live in version control and be reviewed like code:
```
name: customers
extra_columns: warn          # allow | warn | error
no_duplicate_rows: true
unique: [customer_id]        # key columns (a list, or a list of lists)
row_count: {min: 100}
columns:
  customer_id: {type: integer, required: true, unique: true}
  email:       {type: string, format: email, max_null_fraction: 0.01}
  age:         {type: number, min: 0, max: 120, warn_only: [min, max]}
  plan:        {type: string, allowed: [basic, pro, enterprise]}
```

Column rules: `type` (integer | number | string | boolean | date | datetime | any),
`required` (no nulls), `max_null_fraction`, `unique`, `min`, `max`, `allowed`,
`pattern` (regex), `format` (email | url | uuid | ipv4 | phone), `min_length`,
`max_length`. Every rule is an `error` by default; `severity: warn` softens a whole
column and `warn_only: [rule, ...]` softens individual rules.

`Contract.infer(df)` writes a starting contract from data you trust; structural rules
(columns, types, required, unique) are errors and statistical ones (ranges, allowed values,
lengths, null fractions) are warnings, because new data legitimately drifts. Review and
tighten it before enforcing.

### class `ContractError`

```python
class ContractError(ValueError):
```

The contract definition itself is invalid.

### class `ContractViolation`

```python
class ContractViolation(Exception):
```

Data failed a contract. `.result` holds the full `ContractResult`.

**Constructor**

```python
ContractViolation(result: 'ContractResult')
```

### class `RuleResult`

```python
class RuleResult:
```

| Field | Type | Default |
|---|---|---|
| `rule` | `str` | required |
| `column` | `str \| None` | required |
| `passed` | `bool` | required |
| `severity` | `str` | required |
| `failed` | `int` | required |
| `checked` | `int` | required |
| `message` | `str` | required |
| `examples` | `list` | required |

### class `ContractResult`

```python
class ContractResult:
```

Outcome of validating one DataFrame. `passed` is False only if an *error* rule failed.

| Field | Type | Default |
|---|---|---|
| `contract` | `str` | required |
| `rows` | `int` | required |
| `results` | `list` | required |

#### `ContractResult.failures` *(property)*

```python
failures: list
```

#### `ContractResult.errors` *(property)*

```python
errors: list
```

#### `ContractResult.warnings` *(property)*

```python
warnings: list
```

#### `ContractResult.passed` *(property)*

```python
passed: bool
```

#### `ContractResult.raise_if_failed()`

```python
raise_if_failed() -> 'ContractResult'
```

#### `ContractResult.to_dict()`

```python
to_dict() -> dict
```

#### `ContractResult.to_json()`

```python
to_json(path=None, indent: int=2) -> str
```

#### `ContractResult.to_frame()`

```python
to_frame() -> pl.DataFrame
```

#### `ContractResult.summary()`

```python
summary() -> str
```

### class `Contract`

```python
class Contract:
```

A data contract. Build it from a dict/YAML/JSON, or infer one with `Contract.infer`.

**Constructor**

```python
Contract(columns: dict | None=None, *, name: str='contract', version=1, description: str='', row_count: dict | None=None, unique=None, no_duplicate_rows: bool=False, extra_columns: str='warn', severity: str='error')
```

#### `Contract.to_dict()`

```python
to_dict() -> dict
```

#### `Contract.from_dict()` *(classmethod)*

```python
from_dict(data: dict) -> 'Contract'
```

#### `Contract.from_yaml()` *(classmethod)*

```python
from_yaml(text: str) -> 'Contract'
```

#### `Contract.from_file()` *(classmethod)*

```python
from_file(path) -> 'Contract'
```

#### `Contract.to_yaml()`

```python
to_yaml(path=None) -> str
```

#### `Contract.to_json()`

```python
to_json(path=None, indent: int=2) -> str
```

#### `Contract.infer()` *(classmethod)*

```python
infer(df, *, name: str='inferred-contract', null_slack: float=0.05, max_allowed_values: int=20, range_margin: float=0.0, extra_columns: str='warn') -> 'Contract'
```

Write a starting contract from data you trust (review it before enforcing).

`null_slack` is added to the observed null fraction; `range_margin` widens
numeric ranges by that share of the observed range.

#### `Contract.validate()`

```python
validate(df) -> ContractResult
```

Check `df` against every rule and return a `ContractResult`.

---

## `mlpilot.quality.pipeline_steps`

<sub>Source: [`mlpilot/quality/pipeline_steps.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/quality/pipeline_steps.py)</sub>

Pipeline steps that enforce data quality inside a `Pipeline`.

* `ValidateContract`: check every batch against a data contract. With no contract given,
  one is inferred from the training data during `fit` and saved inside the pipeline.
* `QualityGate`: stop (or warn) when the quality score or high-severity issues cross a limit.

Both pass data through unchanged; put them first to guard raw input, last to guard output.

### class `QualityWarning`

```python
class QualityWarning(MLPilotWarning):
```

Warning category for contract or quality-gate failures with `on_fail="warn"`.

### class `QualityGateError`

```python
class QualityGateError(Exception):
```

Data didn't meet a `QualityGate`. `.report` holds the QualityReport.

**Constructor**

```python
QualityGateError(message: str, report)
```

### class `ValidateContract`

```python
class ValidateContract(Step):
```

Validate each batch against a data contract (inferred from training data if not given).

**Constructor**

```python
ValidateContract(*, contract=None, on_fail: str='raise', infer_options: dict | None=None, fit_only=None)
```

#### `ValidateContract.contract` *(property)*

```python
contract: Contract
```

### class `QualityGate`

```python
class QualityGate(Step):
```

Stop (or warn) when the quality score, a dimension or high-severity issues cross a limit.

**Constructor**

```python
QualityGate(*, min_score: float | None=80.0, max_high_issues: int | None=None, min_dimensions: dict | None=None, on_fail: str='raise', fit_only=None)
```

---

## `mlpilot.quality.profile`

<sub>Source: [`mlpilot/quality/profile.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/quality/profile.py)</sub>

Automated EDA: column profiles, semantic types, correlations and plain-English insights.

### `semantic_type()`

```python
def semantic_type(series: pl.Series, name: str | None=None) -> str
```

What a column *means*, beyond its storage type (see `SEMANTIC_TYPES`).

### class `ColumnProfile`

```python
class ColumnProfile:
```

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `dtype` | `str` | required |
| `semantic_type` | `str` | required |
| `count` | `int` | required |
| `nulls` | `int` | required |
| `null_pct` | `float` | required |
| `unique` | `int` | required |
| `unique_pct` | `float` | required |
| `top_values` | `list` | required |
| `numeric` | `dict \| None` | `None` |
| `text` | `dict \| None` | `None` |
| `temporal` | `dict \| None` | `None` |

#### `ColumnProfile.to_dict()`

```python
to_dict() -> dict
```

### class `Profile`

```python
class Profile:
```

Result of `profile`. Print it for an overview; `profile["col"]` for one column.

| Field | Type | Default |
|---|---|---|
| `rows` | `int` | required |
| `columns` | `int` | required |
| `memory_mb` | `float` | required |
| `missing_cells` | `int` | required |
| `missing_pct` | `float` | required |
| `duplicate_rows` | `int` | required |
| `created_at` | `str` | required |
| `column_profiles` | `dict` | required |
| `correlations` | `list` | required |
| `insights` | `list` | required |
| `quality` | `QualityReport \| None` | `None` |

#### `Profile.to_dict()`

```python
to_dict() -> dict
```

#### `Profile.to_json()`

```python
to_json(path=None, indent: int=2) -> str
```

#### `Profile.to_frame()`

```python
to_frame() -> pl.DataFrame
```

One row per column with the key statistics.

#### `Profile.summary()`

```python
summary() -> str
```

### `profile()`

```python
def profile(df, *, top_n: int=10, correlation_threshold: float=0.5, quality: bool=True) -> Profile
```

Profile a DataFrame: per-column statistics, semantic types, correlations, insights and
(by default) the full data-quality report from `check_quality`.

---

## `mlpilot.quality.schema`

<sub>Source: [`mlpilot/quality/schema.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/quality/schema.py)</sub>

Schema inference and comparison: catch added, removed and retyped columns.

### `infer_schema()`

```python
def infer_schema(df) -> dict
```

`{column: dtype-string}` for a DataFrame (pandas is converted).

### class `SchemaDiff`

```python
class SchemaDiff:
```

Differences between an expected and an actual schema.

| Field | Type | Default |
|---|---|---|
| `added` | `list` | required |
| `removed` | `list` | required |
| `changed` | `dict` | required |

#### `SchemaDiff.identical` *(property)*

```python
identical: bool
```

#### `SchemaDiff.compatible` *(property)*

```python
compatible: bool
```

True when nothing expected is missing or retyped (new columns are tolerated).

#### `SchemaDiff.to_dict()`

```python
to_dict() -> dict
```

#### `SchemaDiff.summary()`

```python
summary() -> str
```

### `compare_schemas()`

```python
def compare_schemas(expected, actual) -> SchemaDiff
```

Compare two schemas; each may be a DataFrame or a `{column: dtype}` dict.
