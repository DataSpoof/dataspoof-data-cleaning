# Quantization & compression

Quantization stores numbers with fewer bits: a model's weights as 8-bit integers or 16-bit
floats instead of 32-bit floats, or embeddings as int8 instead of float64. You get smaller
files, less memory and sometimes faster inference. MLPilot quantizes in three places, and
always measures the cost before you rely on it:

| What | How | Typical saving |
|---|---|---|
| Neural-network models | int8 / fp16 ONNX export (`quantize_model`, `export_model(..., quantize=)`) | 2–4× smaller files |
| Embeddings | `TextEmbeddings(dtype="int8")`, `quantize_vectors` | 2–8× less memory |
| Tables | `optimize_dtypes`, the `OptimizeDtypes` step | 1.5–4× less memory, values unchanged |

## Models: int8 and fp16 ONNX

```python
import mlpilot as ac

model = ac.AutoML("churned", include=["mlp"]).fit(train)      # a neural network (opt-in model family)
rep = ac.quantize_model(model, test, out="build/quantized")   # held-out rows, including the target
print(rep.summary())
```

```
fp32       42.1 KB   1.00x  roc_auc 0.9549             agreement 100.00%     0.75 ms/1k rows
int8       15.1 KB   2.79x  roc_auc 0.9549 (+0.0000)   agreement  99.20%     0.52 ms/1k rows   <- recommended
fp16       22.5 KB   1.87x  roc_auc 0.9550 (+0.0001)   agreement 100.00%     0.44 ms/1k rows
```

The report shows each variant's file size, its speed (ONNX Runtime on CPU), the metric on
your held-out data, and **agreement**: the share of predictions identical to the original
model's. For regression, a prediction counts as unchanged when it moves by less than 5% of the
predictions' standard deviation.

The recommended variant is the smallest one that loses at most `max_drop` of the metric
(default 0.01) and agrees with the original on at least `min_agreement` of rows (default 0.99).
`rep.recommended.path` is the file to deploy. The full report is also saved as
`quantization.json`.

```python
ac.export_model(model, "build/onnx", "onnx", sample=test, quantize="int8")   # -> build/onnx/model.int8.onnx
ac.quantize_onnx("my_cnn.onnx", "my_cnn.int8.onnx")                          # any ONNX model you have
```

```bash
mlpilot quantize --model build/model --data build/test.parquet --out build/quantized --modes int8,fp16
mlpilot export --model build/model --out build/onnx --format onnx --quantize fp16
```

| Mode | What changes | Size | Accuracy |
|---|---|---|---|
| `int8` / `uint8` | Weights of matrix-multiply and convolution layers become 8-bit integers. Activations are quantized on the fly (ONNX Runtime dynamic quantization). | ~3–4× smaller | small loss; check the report |
| `fp16` | Float32 weights are *stored* as float16 and cast back when the model loads. The maths stays float32, so it runs at full speed on every CPU. | ~2× smaller | almost lossless |

**What quantization can and can't shrink.** It works on neural networks: AutoML's opt-in `mlp`
model, and your own ONNX models (image CNNs, transformers, sentence embedders). Tree ensembles
(random forest, extra trees), linear models and k-NN are different. They export to ONNX-ML
operators that keep their own float32 parameters, and ONNX Runtime can't quantize those (its
tree kernels accept only float32 and float64). For those models the report says
*nothing to quantize* instead of writing a file that is no smaller.

On the CPU with small models, speed barely changes. The win is file size and memory. Larger
networks and batch sizes gain more speed from int8.

`mlp` is opt-in: AutoML only tries it when you name it in `include=[...]`, so default runs are
unchanged. On tabular data a tree ensemble usually wins. Train both and compare the
leaderboard, `rep.summary()` and your deployment budget.

### As a release gate

```yaml
# gates.yml
quantization:
  mode: int8
  max_drop: 0.01        # largest allowed metric loss
  min_agreement: 0.99   # share of predictions that must not change
  required: false       # true: fail models that have nothing to quantize
```

`mlpilot gate` then fails the build when the quantized model loses more than allowed. The
size, compression and metric change go into the gate report.

## Embeddings: float32, float16, int8

```python
ac.TextEmbeddings(columns=["review"], dtype="int8")      # also "float32", "float16" (default "float64")

from mlpilot.compression import quantize_vectors, dequantize_vectors
q, params = quantize_vectors(vectors, "int8")            # per-dimension scale, learned from these vectors
q_new, _ = quantize_vectors(new_vectors, "int8", scale=params["scale"])   # reuse the training scale
approx = dequantize_vectors(q, params)
```

`int8` scales each dimension to −127…127 by its largest absolute value on the training data.
Pipelines save that scale, so new data is quantized the same way, and values outside the
training range are clipped.

In testing, a review classifier used 32 TF-IDF/SVD dimensions on 2,000 training rows:

| dtype | Memory | Test ROC AUC |
|---|---|---|
| float64 | 512 KB | 0.9231 |
| float32 | 256 KB | 0.9231 |
| float16 | 128 KB | 0.9231 |
| int8 | 64 KB | 0.9219 |

## Tables: smaller column types, same values

```python
small, rep = ac.optimize_dtypes(df)
print(rep.summary())
# 7.8 MB -> 4.8 MB (1.6x)
#   id: Int64 -> Int32
#   flag: Int64 -> Int8
#   price: Float64 -> Float32, values change by up to 1.46e-05
#   city: String -> Categorical
```

`optimize_dtypes` doesn't change values unless you ask it to:

- **Integers** move to the smallest signed type that holds the column's minimum and maximum.
  Ids above 2.1 billion stay `Int64`. Pass `unsigned=True` to allow `UInt` types.
- **Floats** (`floats="safe"`, the default) become `Float32` only when every value survives
  the round trip at the precision the data is written in:
  - prices like `19.99` convert;
  - `123456789.12` stays `Float64`, because Float32 keeps only about 7 significant digits;
  - continuous measurements stay `Float64` too.

  `floats="float32"` converts every float column and reports the largest change it made;
  `floats=None` leaves floats alone.
- **Text** with few distinct values (at most `categorical` = 50% of rows) becomes `Categorical`.

In a pipeline, the `OptimizeDtypes` step learns the types once on training data. Every batch
then comes out with the same schema. When new data no longer fits a learned type, for example
an id beyond the training range, that column keeps its type and you get a warning. It never
overflows.

```python
pipe = ac.Pipeline([ac.Impute(), ac.ClipOutliers(), ac.OptimizeDtypes()]).fit(train)
```

`save_cleaned_data(..., quantize=True)` uses the same safe rules. It used to cast `Int64` to
`Int32` blindly, which failed on large ids.
