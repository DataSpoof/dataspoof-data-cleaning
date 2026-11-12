# Forecasting: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Forecasting guide](../guide/forecasting.md).

**On this page:**

- [`mlpilot.forecast`](#mlpilotforecast): `infer_frequency`, `BacktestReport`, `Forecaster`, `pd_categorical`, `reconcile_bottom_up`

---

## `mlpilot.forecast`

<sub>Source: [`mlpilot/forecast.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/forecast.py)</sub>

Time-series forecasting: multi-step horizons, many series at once, backtests, intervals.

`Forecaster(target, time_col, horizon=14, group="store")` learns from history and predicts the
next `horizon` periods for every series:

* **models** — a seasonal-naive baseline and a gradient-boosting model on lags, rolling means and
  calendar features (recursive multi-step); the better one on the backtest is kept.
* **backtest** — rolling-origin evaluation (`folds` cut-offs, each forecasting `horizon` steps
  on data the model never saw): MAE, RMSE, sMAPE and MASE (error relative to seasonal-naive).
* **intervals** — conformal: the empirical spread of backtest errors at each step ahead, so the
  stated coverage (`coverage=0.9`) holds on past forecasts.
* `reconcile_bottom_up` makes forecasts of a hierarchy add up (stores -> region -> total).

### `infer_frequency()`

```python
def infer_frequency(times: pl.Series) -> str
```

'h', 'd', 'w', 'mo', 'q' or 'y' from the median spacing of the timestamps.

### class `BacktestReport`

```python
class BacktestReport:
```

| Field | Type | Default |
|---|---|---|
| `model` | `str` | required |
| `folds` | `int` | required |
| `horizon` | `int` | required |
| `metrics` | `dict` | required |
| `by_step` | `pl.DataFrame` | required |
| `coverage` | `float \| None` | `None` |

#### `BacktestReport.summary()`

```python
summary() -> str
```

### class `Forecaster`

```python
class Forecaster:
```

```python
fc = Forecaster("sales", "date", horizon=14, group="store").fit(history)
fc.backtest_                  # how accurate it was on held-out history
fc.predict()                  # store, date, forecast, lower, upper for the next 14 days
```

**Constructor**

```python
Forecaster(target: str, time_col: str, *, horizon: int=14, group: str | None=None, freq: str | None=None, season: int | None=None, lags: list | None=None, folds: int=3, coverage: float=0.9, models=('seasonal_naive', 'gradient_boosting'), known_future: list | None=None, seed: int=42)
```

#### `Forecaster.fit()`

```python
fit(df) -> 'Forecaster'
```

#### `Forecaster.predict()`

```python
predict(horizon: int | None=None, *, future=None, history=None) -> pl.DataFrame
```

Forecasts (with `lower`/`upper` bounds) for the `horizon` periods after the end of
the training data, or after `history` (newer data, same columns) if given.

#### `Forecaster.save()`

```python
save(path)
```

Save to a directory (checksummed model + manifest + the history tail it needs).

#### `Forecaster.load()` *(classmethod)*

```python
load(path) -> 'Forecaster'
```

#### `Forecaster.summary()`

```python
summary() -> str
```

### `pd_categorical()`

```python
def pd_categorical(levels)
```

### `reconcile_bottom_up()`

```python
def reconcile_bottom_up(forecasts: pl.DataFrame, hierarchy: dict, *, group: str, time_col: str, value_cols=('forecast', 'lower', 'upper')) -> pl.DataFrame
```

Add parent levels to bottom-level forecasts so the hierarchy adds up.

`hierarchy` maps each bottom series to its ancestors, e.g.
`{"store_1": ["north", "total"], "store_2": ["north", "total"], "store_3": ["south", "total"]}`.
Point forecasts are summed exactly; interval bounds are summed too (conservative).
