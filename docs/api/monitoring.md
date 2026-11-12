# Monitoring: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Monitoring guide](../guide/monitoring.md).

**On this page:**

- [`mlpilot.monitoring`](#mlpilotmonitoring)
- [`mlpilot.monitoring.drift`](#mlpilotmonitoringdrift): `psi`, `js_distance`, `severity_of`, `ColumnBaseline`, `fit_column`, `Baseline`, `compare_column`, `DriftReport`, `compare`, `drift_report`
- [`mlpilot.monitoring.estimate`](#mlpilotmonitoringestimate): `PerformanceEstimator`, `domain_classifier_drift`
- [`mlpilot.monitoring.monitor`](#mlpilotmonitoringmonitor): `RetrainPolicy`, `Alert`, `MonitorReport`, `webhook_alert`, `Monitor`, `read_prediction_log`, `RetrainResult`, `retrain`, `retrain_if_needed`
- [`mlpilot.monitoring.ops`](#mlpilotmonitoringops): `pagerduty_alert`, `opsgenie_alert`, `join_labels`, `join_feedback`, `sql_sink`

---

## `mlpilot.monitoring`

<sub>Source: [`mlpilot/monitoring/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/monitoring/__init__.py)</sub>

Monitoring (module 9): data & prediction drift, accuracy decay, alerts, retraining triggers.

* `drift_report(reference, current)`      PSI / JS distance / chi-square per column.
* `Monitor(model, reference)`             `check(batch)` -> alerts + retrain recommendation;
  saves baselines only (no raw data).
* `retrain(model, train, holdout, registry=..., name=...)`  champion vs challenger, with
  registration and gated promotion.
* `webhook_alert(url)`                    send alerts to Slack/Teams/any webhook.
* `PerformanceEstimator(model)`           accuracy / ROC AUC / MAE estimated *without labels*
  (Monitor.check uses it when a batch has no labels).
* `domain_classifier_drift(ref, cur)`     multivariate drift: can a model tell the two apart?

---

## `mlpilot.monitoring.drift`

<sub>Source: [`mlpilot/monitoring/drift.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/monitoring/drift.py)</sub>

Data drift: how far a new batch's distributions have moved from a reference.

A `Baseline` keeps only binned distributions and summary statistics of the reference data
(never raw rows), so it can be saved as JSON and shipped with a model. Drift per column is
measured with the Population Stability Index (PSI: < 0.1 stable, 0.1–0.25 moderate,
> 0.25 major), the Jensen–Shannon distance, a chi-square test, missing-rate change and the
share of categories never seen in the reference.

### `psi()`

```python
def psi(p: np.ndarray, q: np.ndarray) -> float
```

Population Stability Index between reference proportions `p` and current `q`.

### `js_distance()`

```python
def js_distance(p, q) -> float
```

### `severity_of()`

```python
def severity_of(value: float, moderate: float=0.1, major: float=0.25) -> str
```

### class `ColumnBaseline`

```python
class ColumnBaseline:
```

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `kind` | `str` | required |
| `proportions` | `list` | required |
| `labels` | `list` | required |
| `edges` | `list \| None` | `None` |
| `n` | `int` | `0` |
| `mean` | `float \| None` | `None` |
| `std` | `float \| None` | `None` |
| `missing_rate` | `float` | `0.0` |

#### `ColumnBaseline.counts()`

```python
counts(s: pl.Series) -> np.ndarray
```

Counts of a new series in this baseline's bins/categories (+ missing).

#### `ColumnBaseline.unseen_share()`

```python
unseen_share(s: pl.Series) -> float
```

### `fit_column()`

```python
def fit_column(s: pl.Series, bins: int=10) -> ColumnBaseline | None
```

Baseline for one column, or None for columns drift isn't meaningful for (dates, IDs).

### class `Baseline`

```python
class Baseline:
```

| Field | Type | Default |
|---|---|---|
| `columns` | `dict` | required |
| `schema` | `dict` | required |
| `n_rows` | `int` | required |

#### `Baseline.fit()` *(classmethod)*

```python
fit(df: pl.DataFrame, columns=None, bins: int=10) -> 'Baseline'
```

#### `Baseline.to_dict()`

```python
to_dict() -> dict
```

#### `Baseline.from_dict()` *(classmethod)*

```python
from_dict(d: dict) -> 'Baseline'
```

### `compare_column()`

```python
def compare_column(base: ColumnBaseline, s: pl.Series) -> dict
```

### class `DriftReport`

```python
class DriftReport:
```

| Field | Type | Default |
|---|---|---|
| `columns` | `pl.DataFrame` | required |
| `missing_columns` | `list` | required |
| `new_columns` | `list` | required |
| `dtype_changes` | `dict` | required |
| `n_reference` | `int` | required |
| `n_current` | `int` | required |

#### `DriftReport.drifted` *(property)*

```python
drifted: list
```

#### `DriftReport.major` *(property)*

```python
major: list
```

#### `DriftReport.drift_share` *(property)*

```python
drift_share: float
```

#### `DriftReport.summary()`

```python
summary(top: int=10) -> str
```

#### `DriftReport.to_dict()`

```python
to_dict() -> dict
```

#### `DriftReport.to_json()`

```python
to_json(indent=2) -> str
```

### `compare()`

```python
def compare(baseline: Baseline, current: pl.DataFrame) -> DriftReport
```

### `drift_report()`

```python
def drift_report(reference, current, columns=None, bins: int=10) -> DriftReport
```

Compare two datasets column by column (`reference` may also be a fitted `Baseline`).

---

## `mlpilot.monitoring.estimate`

<sub>Source: [`mlpilot/monitoring/estimate.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/monitoring/estimate.py)</sub>

Concept drift *before* the outcomes arrive: estimate live performance without labels.

Labels often arrive weeks after a prediction (did the customer churn? was the loan repaid?).
These estimators warn about a model getting worse in the meantime:

```python
est = PerformanceEstimator(model).fit(reference_with_labels)     # held-out data with labels
est.estimate(todays_batch)                                       # no labels needed
```
{'metric': 'accuracy', 'estimated': 0.81, 'reference': 0.90, 'low': 0.79, 'high': 0.83, ...}
```python
domain_classifier_drift(reference, todays_batch)                 # multivariate drift
```

* **Classification - confidence-based performance estimation (CBPE):** the model's confidence is
  calibrated on the reference data (isotonic regression of "was it right?" on confidence, and of
  the true label on P(positive) for binary models). On new data, the expected accuracy is the
  mean calibrated confidence; for binary models the expected ROC AUC is computed by simulating
  labels from the calibrated probabilities. This tracks performance under *covariate shift*
  (inputs moving into regions where the model is less sure). It cannot see a change in P(y|x)
  that leaves the model confident - only real labels can (`Monitor.check_labels`).
* **Regression - direct loss estimation:** expected absolute error per prediction-range bin,
  learnt on the reference data, re-weighted by today's predictions.
* **Multivariate drift - domain classifier:** a model trained to tell reference rows from current
  rows. ROC AUC near 0.5 means no detectable change; the features it relies on are the drift.

Everything is stored as plain numbers (JSON-safe), never raw data.

### class `PerformanceEstimator`

```python
class PerformanceEstimator:
```

Estimates a model's accuracy (and ROC AUC for binary models) or MAE without labels.

**Constructor**

```python
PerformanceEstimator(model, *, bins: int=10, simulations: int=200, seed: int=0)
```

#### `PerformanceEstimator.fit()`

```python
fit(reference) -> 'PerformanceEstimator'
```

#### `PerformanceEstimator.estimate()`

```python
estimate(batch) -> dict
```

Estimated metric for an unlabelled batch, with a 90% interval and the reference value.

#### `PerformanceEstimator.to_dict()`

```python
to_dict() -> dict
```

#### `PerformanceEstimator.from_dict()` *(classmethod)*

```python
from_dict(model, d: dict) -> 'PerformanceEstimator'
```

### `domain_classifier_drift()`

```python
def domain_classifier_drift(reference, current, *, columns: list | None=None, cv: int=3, seed: int=0, max_rows: int=20000) -> dict
```

Train a classifier to separate reference rows from current rows (cross-validated).
Returns `{"auc", "drift": bool, "top_features": [(column, importance)...]}`; AUC above
~0.55-0.6 means the two samples are distinguishable, i.e. the joint distribution moved.

---

## `mlpilot.monitoring.monitor`

<sub>Source: [`mlpilot/monitoring/monitor.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/monitoring/monitor.py)</sub>

Production monitoring: data drift, prediction drift, accuracy decay, alerts and retraining.

### class `RetrainPolicy`

```python
class RetrainPolicy:
```

When a batch should raise alerts and trigger retraining.

* `max_drift_share`: share of monitored columns with *major* drift (PSI ≥ 0.25) → critical
* `prediction_psi`: PSI of the prediction distribution → warning (critical at 2×)
* `max_metric_drop`: absolute drop of a higher-is-better metric (accuracy, AUC, R²) → critical
* `max_error_increase`: relative increase of RMSE / MAE → critical
* `min_rows`: smaller batches get a warning that the statistics are unreliable

| Field | Type | Default |
|---|---|---|
| `max_drift_share` | `float` | `0.3` |
| `prediction_psi` | `float` | `0.2` |
| `max_metric_drop` | `float` | `0.05` |
| `max_error_increase` | `float` | `0.1` |
| `min_rows` | `int` | `100` |

### class `Alert`

```python
class Alert:
```

| Field | Type | Default |
|---|---|---|
| `severity` | `str` | required |
| `kind` | `str` | required |
| `message` | `str` | required |
| `value` | `float \| None` | `None` |
| `threshold` | `float \| None` | `None` |

### class `MonitorReport`

```python
class MonitorReport:
```

| Field | Type | Default |
|---|---|---|
| `time` | `str` | required |
| `rows` | `int` | required |
| `data_drift` | `DriftReport` | required |
| `prediction_drift` | `dict \| None` | required |
| `performance` | `dict \| None` | required |
| `alerts` | `list` | required |
| `retrain_recommended` | `bool` | required |
| `name` | `str \| None` | `None` |

#### `MonitorReport.critical` *(property)*

```python
critical: list
```

#### `MonitorReport.summary()`

```python
summary() -> str
```

#### `MonitorReport.to_dict()`

```python
to_dict() -> dict
```

### `webhook_alert()`

```python
def webhook_alert(url: str, *, timeout: float=5.0, headers: dict | None=None)
```

An alert handler that POSTs each report (JSON) to `url` (Slack/Teams incoming webhooks,
or any HTTP endpoint). For incident tools see `ops.pagerduty_alert` / `ops.opsgenie_alert`.

### class `Monitor`

```python
class Monitor:
```

Watches a deployed model.

```python
mon = Monitor(model, reference=holdout, log_path="monitor.jsonl")
report = mon.check(todays_batch)            # labels optional: include the target column
report.retrain_recommended, report.alerts
mon.save("monitor.json")                    # baselines only - no raw data
```

Data drift is checked on the raw input columns the model uses; prediction drift on its
outputs; performance (when the batch has labels) against the reference metrics.

**Constructor**

```python
Monitor(model, reference=None, *, policy: RetrainPolicy | None=None, log_path=None, columns=None, reference_metrics: dict | None=None, name: str | None=None, estimate_performance: bool=True)
```

#### `Monitor.fit()`

```python
fit(reference) -> 'Monitor'
```

Learn the reference distributions (use held-out data, not the training set, so the
reference metrics are realistic).

#### `Monitor.on_check()`

```python
on_check(handler) -> 'Monitor'
```

Register `handler(report)`; called after every check (e.g. `ops.sql_sink(db)`).

#### `Monitor.check_labels()`

```python
check_labels(joined: pl.DataFrame, *, prediction: str='prediction', label: str='label') -> dict
```

Live performance from predictions that were actually served and their ground truth
(see `monitoring.ops.join_labels`), compared with the reference metrics. Raises the
same `performance` alerts as `check` and calls the alert handlers.

#### `Monitor.on_alert()`

```python
on_alert(handler) -> 'Monitor'
```

Register `handler(report, alerts)`; called whenever a check raises alerts.

#### `Monitor.check()`

```python
check(batch) -> MonitorReport
```

#### `Monitor.history()`

```python
history() -> pl.DataFrame
```

Past checks from `log_path`: time, rows, drift share, metric, alert count.

#### `Monitor.save()`

```python
save(path) -> Path
```

#### `Monitor.load()` *(classmethod)*

```python
load(path, model, *, log_path=None) -> 'Monitor'
```

### `read_prediction_log()`

```python
def read_prediction_log(path, since: float | None=None) -> pl.DataFrame
```

Inputs logged by the REST API (`prediction_log=`) as a frame, ready for `Monitor.check`.

### class `RetrainResult`

```python
class RetrainResult:
```

| Field | Type | Default |
|---|---|---|
| `challenger` | `object` | required |
| `metric` | `str` | required |
| `champion_score` | `float \| None` | required |
| `challenger_score` | `float` | required |
| `improved` | `bool` | required |
| `version` | `Any` | `None` |
| `promoted_to` | `str \| None` | `None` |

#### `RetrainResult.summary()`

```python
summary() -> str
```

### `retrain()`

```python
def retrain(model, train_df, holdout_df, *, registry=None, name: str | None=None, promote_to: str | None='staging', min_improvement: float=0.0, **automl_kwargs) -> RetrainResult
```

Train a challenger on fresh data and compare it with the current model on `holdout_df`.

Settings (task, metric, leakage handling) are copied from the current model. With
`registry` and `name` the challenger is registered, and promoted to `promote_to`
only if it beats the current model by at least `min_improvement` on the holdout.

### `retrain_if_needed()`

```python
def retrain_if_needed(monitor: Monitor, batch, train_df, holdout_df, **kwargs)
```

Check `batch`; retrain only if the policy recommends it. Returns `(report, result | None)`.

---

## `mlpilot.monitoring.ops`

<sub>Source: [`mlpilot/monitoring/ops.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/monitoring/ops.py)</sub>

Operations around a Monitor: incident alerting, delayed ground-truth labels, result sinks.

* `pagerduty_alert(routing_key)` / `opsgenie_alert(api_key)`  incident tools (Slack/Teams:
  `webhook_alert`). Critical alerts page; warnings are sent as low priority.
* `join_labels(prediction_log, labels, on="order_id")`  match labels that arrive days later
  to the predictions that were actually served, then `Monitor.check_labels(joined)`.
* `sql_sink(db, "model_monitoring")`  store every check's summary in a database table.

### `pagerduty_alert()`

```python
def pagerduty_alert(routing_key: str, *, source: str='mlpilot', component: str | None=None, url: str='https://events.pagerduty.com/v2/enqueue', timeout: float=5.0)
```

Alert handler for PagerDuty Events API v2. One incident per model + alert kind
(`dedup_key`), so a recurring drift alert updates the open incident instead of paging again.
`routing_key` may be a secret reference (`vault://...`).

### `opsgenie_alert()`

```python
def opsgenie_alert(api_key: str, *, region: str='us', team: str | None=None, timeout: float=5.0)
```

Alert handler for Opsgenie (`region="eu"` for EU accounts). Critical -> P1, warning -> P3.
`api_key` may be a secret reference (`vault://...`).

### `join_labels()`

```python
def join_labels(prediction_log, labels: pl.DataFrame, *, on: str, label: str | None=None, since: float | None=None) -> pl.DataFrame
```

Predictions that were served (from the REST API's `prediction_log`) joined with ground
truth that arrived later. `on` is an input field identifying the entity (e.g. an order id);
`label` the column in `labels` holding the outcome (default: the one non-key column).

Returns the logged inputs plus `prediction`, `label`, `variant` and `time`; rows
whose label hasn't arrived yet are dropped.

### `join_feedback()`

```python
def join_feedback(prediction_log, feedback_log, *, since: float | None=None) -> pl.DataFrame
```

Predictions joined with the outcomes users sent to `POST /feedback` (matched on the
prediction id the API returned). Returns the logged inputs plus `prediction`, `label`,
`variant`, `version`, `rating`, `comment` and `time` - ready for
`Monitor.check_labels` (only rows with a label) or for reviewing ratings and comments.
The latest feedback per prediction wins.

### `sql_sink()`

```python
def sql_sink(db, table: str='model_monitoring')
```

`Monitor.on_check` handler writing one summary row per check to `table` (created if
missing) via an `ingestion.SQLSource` — for dashboards in the warehouse/BI tool.
