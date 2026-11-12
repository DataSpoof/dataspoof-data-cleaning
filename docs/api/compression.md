# Quantization & compression: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Quantization & compression guide](../guide/quantization.md).

**On this page:**

- [`mlpilot.compression`](#mlpilotcompression): `DtypeReport`, `plan_dtypes`, `optimize_dtypes`, `OptimizeDtypes`, `quantize_vectors`, `dequantize_vectors`

---

## `mlpilot.compression`

<sub>Source: [`mlpilot/compression.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/compression.py)</sub>

Data compression: smaller column types without changing values, and quantized vectors.

```python
small, rep = optimize_dtypes(df)          # Int64 ids -> Int32, 0/1 flags -> Int8, prices -> Float32, ...
print(rep.summary())                      # 41.2 MB -> 12.9 MB (3.2x); one row per changed column
pipe = ac.Pipeline([..., OptimizeDtypes()])        # learn the types on training data, reuse them
```

```python
q, params = quantize_vectors(embeddings, "int8")    # 8x smaller than float64
approx = dequantize_vectors(q, params)
```

`optimize_dtypes` never changes a value unless you ask it to:

* integers move to the smallest signed type that holds the column's min and max;
* floats (`floats="safe"`, the default) become Float32 only when every value survives the
  round trip at the precision the data is written in: prices like 19.99 do, 123456789.12 does
  not (Float32 keeps about 7 significant digits) and stays Float64. Columns of continuous
  measurements stay Float64 too; pass `floats="float32"` to convert them anyway (the report
  shows the largest change), or `floats=None` to leave floats alone;
* text with few distinct values becomes Categorical (`categorical` = max distinct/rows).

### class `DtypeReport`

```python
class DtypeReport:
```

| Field | Type | Default |
|---|---|---|
| `frame` | `pl.DataFrame` | required |
| `bytes_before` | `int` | required |
| `bytes_after` | `int` | required |

#### `DtypeReport.ratio` *(property)*

```python
ratio: float
```

#### `DtypeReport.changed` *(property)*

```python
changed: dict
```

`{column: new dtype}` for every column that changed type.

#### `DtypeReport.summary()`

```python
summary() -> str
```

### `plan_dtypes()`

```python
def plan_dtypes(df: pl.DataFrame, *, floats: str | None='safe', categorical: float | None=0.5, unsigned: bool=False, exclude=()) -> dict
```

`{column: (new dtype, max value change)}` for the columns `optimize_dtypes` would change.

### `optimize_dtypes()`

```python
def optimize_dtypes(df: pl.DataFrame, *, floats: str | None='safe', categorical: float | None=0.5, unsigned: bool=False, exclude=()) -> tuple
```

Shrink `df`'s column types; returns `(smaller_df, DtypeReport)` (see the module docstring).
`unsigned=True` also uses UInt types for non-negative integers (beware: subtracting
unsigned columns can wrap around).

### class `OptimizeDtypes`

```python
class OptimizeDtypes(Step):
```

Pipeline step: learn compact column types on training data, apply them to new data.

On new data a column is only cast when its values still fit the learned type (an id above
the training range, say); otherwise it keeps its type and a warning says so, so scoring
never fails or silently wraps around.

**Constructor**

```python
OptimizeDtypes(*, floats: str | None='safe', categorical: float | None=0.5, unsigned: bool=False, exclude=(), fit_only=None)
```

### `quantize_vectors()`

```python
def quantize_vectors(x, mode: str='int8', *, scale=None) -> tuple
```

Quantize a 2-D array of vectors (embeddings). Returns `(quantized, params)`.

* `float32` / `float16`: 2x / 4x smaller than float64; float16 keeps ~3 significant digits.
* `int8`: 8x smaller. Each dimension is scaled symmetrically to -127..127 using its largest
  absolute value (`scale`, learned from `x` unless given, e.g. from training data);
  values beyond the training range are clipped. Use `dequantize_vectors` to get floats back.

### `dequantize_vectors()`

```python
def dequantize_vectors(q, params: dict) -> np.ndarray
```

Float vectors back from `quantize_vectors` output.
