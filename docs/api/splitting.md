# Data splitting: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Data splitting guide](../guide/splitting.md).

**On this page:**

- [`mlpilot.splitting`](#mlpilotsplitting): `SplitWarning`, `SplitResult`, `split_data`, `check_split`, `cv_folds`

---

## `mlpilot.splitting`

<sub>Source: [`mlpilot/splitting.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/splitting.py)</sub>

Data splitting: train / validation / test splits and cross-validation folds that don't leak.

```python
s = split_data(df, target="churned", test_size=0.2, val_size=0.1)        # stratified (classification)
s = split_data(df, target="churned", group="customer_id")                # a customer is never on both sides
s = split_data(df, target="sales", time="date", gap="7d")                # train on the past, test on the future
s.train, s.val, s.test
print(s.report())                       # sizes, class balance, time ranges, and the leakage checks
check_split(my_train, my_test, group="patient_id")   # audit a split you made yourself
for train, test in cv_folds(df, target="churned", n_splits=5, group="customer_id"): ...
```

Strategy is chosen from the arguments:

* `time=`   chronological: the oldest rows train, the newest test (`gap` drops the rows just
  before each boundary, so features computed over a window can't leak forward).
* `group=`  every row of a group lands in the same split (patients, customers, sessions...).
  With a classification target, groups are also balanced by class.
* otherwise   random; stratified by the target when it is a classification target.

Every split is checked afterwards: groups appearing in two splits, time ranges overlapping, and
identical rows appearing in both train and test (duplicates are the most common silent leak).

### class `SplitWarning`

```python
class SplitWarning(MLPilotWarning):
```

A split could not be made exactly as requested (e.g. a class too rare to stratify).

### class `SplitResult`

```python
class SplitResult:
```

| Field | Type | Default |
|---|---|---|
| `train` | `pl.DataFrame` | required |
| `test` | `pl.DataFrame` | required |
| `val` | `pl.DataFrame \| None` | `None` |
| `strategy` | `str` | `'random'` |
| `target` | `str \| None` | `None` |
| `group` | `str \| None` | `None` |
| `time` | `str \| None` | `None` |
| `dropped_for_gap` | `int` | `0` |
| `issues` | `list` | `list()` |

#### `SplitResult.parts` *(property)*

```python
parts: dict
```

#### `SplitResult.sizes()`

```python
sizes() -> dict
```

#### `SplitResult.report()`

```python
report() -> str
```

#### `SplitResult.save()`

```python
save(directory) -> Path
```

Write each part as Parquet plus `split.json` (strategy, sizes, row counts, checks).

### `split_data()`

```python
def split_data(df, target: str | None=None, *, test_size: float=0.2, val_size: float=0.0, stratify: bool | None=None, group: str | None=None, time: str | None=None, gap=None, seed: int=0, shuffle: bool=True) -> SplitResult
```

Split `df` into train / (val) / test. Sizes are fractions (< 1) or row counts. See the
module docstring for how the strategy is chosen.

### `check_split()`

```python
def check_split(train, test, val=None, *, group: str | None=None, time: str | None=None, target: str | None=None, ignore: list | None=None, _report: bool=False) -> list
```

Leakage problems between splits (an empty list means none found):

* the same `group` value in two splits;
* `time`: a later split containing rows older than the newest training row;
* identical rows (ignoring `target` and `ignore` columns) in train and test/val;
* a classification class present in test but never seen in train.

### `cv_folds()`

```python
def cv_folds(df, target: str | None=None, *, n_splits: int=5, stratify: bool | None=None, group: str | None=None, time: str | None=None, gap=0, seed: int=0, indices: bool=False)
```

Cross-validation folds as `(train, test)` DataFrames (or index arrays with
`indices=True`), using the same strategy rules as `split_data`. Time folds expand
forward (train on everything before each test window); `gap` rows are skipped between.
