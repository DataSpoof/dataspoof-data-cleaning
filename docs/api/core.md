# Pipeline core: API reference

Generated from the source docstrings.

**On this page:**

- [`mlpilot.core`](#mlpilotcore)
- [`mlpilot.core.pipeline`](#mlpilotcorepipeline): `SchemaMismatchError`, `MLPilotWarning`, `ServiceWarning`, `PipelineWarning`, `PipelineStepError`, `make_step`, `StepReport`, `RunReport`, `Pipeline`
- [`mlpilot.core.registry`](#mlpilotcoreregistry): `register_step`, `register_function`, `register_function_namespace`, `get_step_class`, `resolve_function`, `available_steps`
- [`mlpilot.core.step`](#mlpilotcorestep): `NotFittedError`, `Step`, `FunctionStep`

---

## `mlpilot.core`

<sub>Source: [`mlpilot/core/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/core/__init__.py)</sub>

Shared pipeline core for the DataSpoof data-science modules.

This subpackage deliberately imports nothing from the rest of `mlpilot`: cleaning
is just the first module built on it, and the core can be lifted into its own package when
feature engineering, AutoML, etc. are added.

Concepts
--------
`Step`      learns rules in `fit` (JSON-serialisable `state_`) and replays them in
              `transform`.
`Pipeline`  ordered steps; `fit_transform` on training data, `transform` on anything
              later; `save`/`load` as JSON; a `RunReport` per run.
Registry      config files refer to steps by name (`register_step`, `register_function`).

---

## `mlpilot.core.pipeline`

<sub>Source: [`mlpilot/core/pipeline.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/core/pipeline.py)</sub>

`Pipeline`: an ordered, savable, auditable sequence of steps.

* `fit` / `fit_transform` learn every step's rules on training data.
* `transform` replays exactly those rules on new data (test sets, production batches).
* `save` / `load` persist a fitted pipeline as plain JSON (never pickle), so it is safe to
  load and easy to review or diff.
* Every run produces a `RunReport` (rows, columns and nulls per step, timings, data
  fingerprints, config hash, versions), optionally appended to an audit log.

### class `SchemaMismatchError`

```python
class SchemaMismatchError(ValueError):
```

Input to `transform` doesn't match the training schema (policy `"raise"`).

### class `MLPilotWarning`

```python
class MLPilotWarning(UserWarning):
```

Base category for every warning this library issues (filter them all with this class).

### class `ServiceWarning`

```python
class ServiceWarning(MLPilotWarning, RuntimeWarning):
```

An optional external service (LLM, alert webhook) failed; a fallback was used.

### class `PipelineWarning`

```python
class PipelineWarning(MLPilotWarning):
```

Warning category for pipeline problems such as schema drift.

### class `PipelineStepError`

```python
class PipelineStepError(RuntimeError):
```

A step failed; the message names the step and the phase.

### `make_step()`

```python
def make_step(spec) -> Step
```

Build a step from a `Step`, a name, `(name, params)` or `{"step", "params", ...}`.

Registered stateful steps win over functions of the same name; prefix a name with
`fn:` to force the plain-function version.

### class `StepReport`

```python
class StepReport:
```

| Field | Type | Default |
|---|---|---|
| `index` | `int` | required |
| `step` | `str` | required |
| `skipped` | `bool` | required |
| `rows_in` | `int` | required |
| `rows_out` | `int` | required |
| `cols_in` | `int` | required |
| `cols_out` | `int` | required |
| `nulls_in` | `int` | required |
| `nulls_out` | `int` | required |
| `added` | `list` | required |
| `removed` | `list` | required |
| `seconds` | `float` | required |
| `log` | `str` | `''` |

### class `RunReport`

```python
class RunReport:
```

What a single `fit`/`transform` run did. Print it for a readable summary.

| Field | Type | Default |
|---|---|---|
| `run_id` | `str` | required |
| `pipeline` | `str` | required |
| `mode` | `str` | required |
| `started_at` | `str` | required |
| `finished_at` | `str` | required |
| `seconds` | `float` | required |
| `config_hash` | `str` | required |
| `versions` | `dict` | required |
| `input` | `dict` | required |
| `output` | `dict` | required |
| `steps` | `list` | required |
| `warnings` | `list` | required |

#### `RunReport.to_dict()`

```python
to_dict() -> dict
```

#### `RunReport.to_json()`

```python
to_json(path=None, indent: int=2) -> str
```

#### `RunReport.to_frame()`

```python
to_frame() -> pl.DataFrame
```

#### `RunReport.summary()`

```python
summary() -> str
```

### class `Pipeline`

```python
class Pipeline:
```

An ordered sequence of steps that is fitted once and replayed on any new data.

Parameters
----------
steps : list of Step instances, step names, `(name, params)` tuples or
    `{"step": name, "params": {...}}` dicts.
name : label used in reports and saved files.
on_schema_mismatch : `"warn"` (default), `"raise"` or `"ignore"` when `transform`
    receives columns/dtypes that differ from training.
verbose : show the steps' own printed output (otherwise it is captured in the report).
audit_log : optional path; every run appends one JSON line with its `RunReport`.
target : optional label column. It is hidden from ordinary steps (so they can't impute,
    clip, encode or drop it), passed to target-aware steps, and kept aligned when rows are
  removed. It may be absent when transforming new data for prediction.

**Constructor**

```python
Pipeline(steps=(), name: str='pipeline', *, on_schema_mismatch: str='warn', verbose: bool=False, audit_log=None, target: str | None=None)
```

#### `Pipeline.add()`

```python
add(step, **params) -> 'Pipeline'
```

Append a step (`pipe.add("impute", numeric="mean")`). Resets the fitted state.

#### `Pipeline.is_fitted` *(property)*

```python
is_fitted: bool
```

#### `Pipeline.fit()`

```python
fit(df) -> 'Pipeline'
```

#### `Pipeline.fit_transform()`

```python
fit_transform(df) -> pl.DataFrame
```

#### `Pipeline.transform()`

```python
transform(df) -> pl.DataFrame
```

#### `Pipeline.to_config()`

```python
to_config() -> dict
```

The pipeline's recipe (steps + params), without learned state.

#### `Pipeline.config_hash()`

```python
config_hash() -> str
```

#### `Pipeline.from_config()` *(classmethod)*

```python
from_config(config, **settings) -> 'Pipeline'
```

Build an unfitted pipeline from a dict or a `.json` / `.yaml` file.

#### `Pipeline.save_config()`

```python
save_config(path) -> Path
```

Write the recipe (no learned state) as JSON, or YAML for `.yaml`/`.yml` paths.

#### `Pipeline.save()`

```python
save(path) -> Path
```

Save the pipeline, including everything it learned, as JSON.

#### `Pipeline.load()` *(classmethod)*

```python
load(path, **settings) -> 'Pipeline'
```

Load a pipeline saved with `save`, ready to `transform` new data.

---

## `mlpilot.core.registry`

<sub>Source: [`mlpilot/core/registry.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/core/registry.py)</sub>

Step and function registry for the pipeline core.

Stateful step classes register themselves with `register_step`. Plain `df -> df`
functions become usable in pipelines either one at a time (`register_function`) or by
exposing a whole module (`register_function_namespace`).

### `register_step()`

```python
def register_step(cls)
```

Class decorator: make a `Step` subclass usable in configs under `cls.name`.

### `register_function()`

```python
def register_function(fn=None, *, name=None)
```

Register a `df -> df` function so pipelines and saved pipelines can refer to it by name.

Use directly (`register_function(my_fn)`) or as a decorator
(`@register_function(name="my_step")`).

### `register_function_namespace()`

```python
def register_function_namespace(module: ModuleType, exclude=())
```

Expose every public function defined in `module` as a pipeline step, minus `exclude`.

### `get_step_class()`

```python
def get_step_class(name: str)
```

### `resolve_function()`

```python
def resolve_function(name: str)
```

Return the function registered under `name`, or `None`.

### `available_steps()`

```python
def available_steps() -> dict
```

Return `{name: {"kind", "description"}}` for every step name a config can use.

`stateful` steps learn their rules in `fit` and replay them in `transform`.
`function` steps re-run their logic on every batch. When a function shares its name
with a stateful step, use it as `fn:<name>`.

---

## `mlpilot.core.step`

<sub>Source: [`mlpilot/core/step.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/core/step.py)</sub>

The `Step` contract shared by every module (cleaning, features, AutoML, ...).

### class `NotFittedError`

```python
class NotFittedError(RuntimeError):
```

Raised when `transform` is called before `fit`.

### class `Step`

```python
class Step:
```

Base class for every pipeline step.

Subclass contract
-----------------
* `name`: unique registry key used in configs (`{"step": name}`).
* `__init__` takes keyword-only, JSON-serialisable parameters and forwards them to
  `super().__init__(fit_only=fit_only, **params)`; they are kept in `self.params`.
* `_fit(df)` learns everything the step needs from *training* data and returns it as a
  JSON-serialisable dict, stored in `self.state_`. Stateless steps don't override it.
* `_transform(df)` applies `self.state_` and must never learn from `df`. That is
  what makes cleaning new data reproduce training-time behaviour exactly.
* `fit_only = True` marks steps that only run while fitting (e.g. dropping duplicate
  or outlier rows); `Pipeline.transform` skips them.

Targets
-------
In a `Pipeline(target=...)` the target column is hidden from ordinary steps, so they
can't impute, clip, encode or drop it. Steps that need it (target encoding, feature
selection, leakage checks) set `uses_target = True` and receive the full frame; during
`transform` on new data the target may be absent and they must cope. Steps that only
remove rows implement `row_mask` so the pipeline can drop the same rows from the target.

**Constructor**

```python
Step(fit_only: bool | None=None, **params: Any)
```

#### `Step.label` *(property)*

```python
label: str
```

Name shown in reports.

#### `Step.row_mask()`

```python
row_mask(df: pl.DataFrame)
```

For steps that only remove rows: a boolean Series of rows to keep. Others: `None`.

#### `Step.is_fitted` *(property)*

```python
is_fitted: bool
```

#### `Step.fit()`

```python
fit(df: pl.DataFrame) -> 'Step'
```

#### `Step.transform()`

```python
transform(df: pl.DataFrame) -> pl.DataFrame
```

#### `Step.fit_transform()`

```python
fit_transform(df: pl.DataFrame) -> pl.DataFrame
```

#### `Step.to_dict()`

```python
to_dict(include_state: bool=True) -> dict
```

### class `FunctionStep`

```python
class FunctionStep(Step):
```

Wrap a stateless `df -> df` function (e.g. `ac.normalize_whitespace`) as a step.

Function steps learn nothing in `fit`; they re-run their logic on every batch.

**Constructor**

```python
FunctionStep(function, fit_only: bool | None=None, **kwargs: Any)
```

#### `FunctionStep.label` *(property)*

```python
label: str
```

#### `FunctionStep.to_dict()`

```python
to_dict(include_state: bool=True) -> dict
```
