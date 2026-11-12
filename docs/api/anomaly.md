# Anomaly detection: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Anomaly detection guide](../guide/anomaly.md).

**On this page:**

- [`mlpilot.anomaly`](#mlpilotanomaly): `AutoAnomaly`

---

## `mlpilot.anomaly`

<sub>Source: [`mlpilot/anomaly.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/anomaly.py)</sub>

Unsupervised anomaly detection with explanations.

`AutoAnomaly().fit(df)` scores how unusual each row is by combining three detectors that catch
different kinds of anomaly — Isolation Forest (rare combinations), Local Outlier Factor (points
far from their neighbours) and a robust z-score (extreme single values) — as an average of their
robustly standardised scores. `contamination` (default `"auto"`: from the score distribution, capped at
5%) sets how many rows are flagged. `explain(df)` says which features make a row unusual.

### class `AutoAnomaly`

```python
class AutoAnomaly:
```

```python
det = AutoAnomaly().fit(transactions)
det.score(new)               # anomaly_score in [0, 1] + is_anomaly
det.explain(new, top=3)      # the features that make each flagged row unusual
```

**Constructor**

```python
AutoAnomaly(*, contamination='auto', methods=('isolation_forest', 'lof', 'robust_z'), exclude=None, seed: int=42)
```

#### `AutoAnomaly.fit()`

```python
fit(df) -> 'AutoAnomaly'
```

#### `AutoAnomaly.score()`

```python
score(df) -> pl.DataFrame
```

`anomaly_score` (0-1: the share of training rows that were less unusual) and
`is_anomaly` for every row of `df`.

#### `AutoAnomaly.explain()`

```python
explain(df, *, top: int=3, only_anomalies: bool=True) -> pl.DataFrame
```

For each (anomalous) row: the `top` numeric features furthest from typical, as
robust z-scores, and categorical values that were rare in the training data.

#### `AutoAnomaly.summary()`

```python
summary() -> str
```
