"""Model quantization: smaller ONNX models with 8-bit or 16-bit weights, measured before you ship.

>>> rep = quantize_model(model, test_df, out="build/quantized")     # AutoML model + held-out rows
>>> print(rep.summary())
fp32   155.3 KB  1.00x  roc_auc 0.9412           agreement 100.00%
int8    43.6 KB  3.56x  roc_auc 0.9405 (-0.0007)  agreement  99.70%   <- recommended
fp16    78.9 KB  1.97x  roc_auc 0.9412 (-0.0000)  agreement 100.00%
>>> rep.recommended.path                                           # build/quantized/model.int8.onnx
>>> quantize_onnx("my_model.onnx", "my_model.int8.onnx")           # any ONNX model (vision, NLP, ...)

Modes:

* ``int8`` / ``uint8``: dynamic quantization (ONNX Runtime). Weights of matrix-multiply and
  convolution layers are stored as 8-bit integers (about 4x smaller); activations are
  quantized on the fly. The biggest saving, with a small accuracy cost.
* ``fp16``: float32 weight tensors are *stored* as float16 (2x smaller) and cast back to
  float32 when the model loads, so it runs everywhere at full float32 speed. Almost lossless.

What it helps: neural networks: AutoML's opt-in ``mlp`` model (``AutoML(..., include=["mlp"])``),
and your own ONNX models (CNNs, transformers, sentence embedders). Tree ensembles, linear models
and k-NN export to ONNX operators that keep their own float32 parameters, which ONNX Runtime
cannot quantize; for those the report says there is nothing to quantize instead of producing
a file that is no smaller.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

MODES = ("int8", "uint8", "fp16")
# Regression "agreement": a prediction counts as unchanged when it moves by less than this share
# of the standard deviation of the original predictions.
REGRESSION_TOLERANCE = 0.05
_WEIGHT_OPS = {"MatMul", "Gemm", "Conv", "ConvTranspose", "Attention", "LSTM", "GRU", "RNN", "EmbedLayerNormalization",
               "Gather"}
_HIGHER_IS_BETTER = {"roc_auc": True, "accuracy": True, "f1": True, "r2": True, "rmse": False, "mae": False,
                     "log_loss": False}


def _ort():
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError('Quantization needs ONNX: pip install "dataspoof-mlpilot[onnx]"') from exc
    return onnx, ort


@contextlib.contextmanager
def _quiet():
    """ONNX Runtime's quantizer logs advice on the root logger; keep it out of the user's output."""
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)
    try:
        yield
    finally:
        root.setLevel(level)


def quantizable_weights(model) -> dict:
    """How many float32 weights an ONNX model (path or ``ModelProto``) has, and how many sit in
    layers that quantization can shrink. ``{"weights", "int8_weights", "fp16_weights"}``."""
    onnx, _ = _ort()
    from onnx import numpy_helper
    m = onnx.load(str(model)) if isinstance(model, (str, Path)) else model
    consumers: dict = {}
    for node in m.graph.node:
        for i, name in enumerate(node.input):
            consumers.setdefault(name, []).append((node.op_type, i))
    total = int8 = fp16 = 0
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype != np.float32:
            continue
        total += arr.size
        if arr.size >= 16:
            fp16 += arr.size
        if any(op in _WEIGHT_OPS and idx > 0 or op == "Gather" and idx == 0 for op, idx in consumers.get(init.name, [])):
            int8 += arr.size
    return {"weights": total, "int8_weights": int8, "fp16_weights": fp16}


def _fp16_weights(m, min_size: int = 16):
    """Store float32 initializers as float16 and cast them back at load time (weight-only fp16)."""
    from onnx import TensorProto, helper, numpy_helper
    g = m.graph
    keep, casts = [], []
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= min_size and np.all(np.abs(arr[np.isfinite(arr)]) < 65000):
            half = numpy_helper.from_array(arr.astype(np.float16), init.name + "__fp16")
            keep.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT,
                                          name=init.name + "__to_fp32"))
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    return m, len(casts)


def quantize_onnx(src, dst, mode: str = "int8", *, per_channel: bool = False) -> Path:
    """Quantize any ONNX model file. ``mode``: ``"int8"``, ``"uint8"`` or ``"fp16"`` (see the
    module docstring). Raises ``ValueError`` when the model has no weights the mode can shrink."""
    onnx, _ = _ort()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    counts = quantizable_weights(src)
    if counts["int8_weights" if mode in ("int8", "uint8") else "fp16_weights"] == 0:
        raise ValueError(f"{src.name} has no weights that {mode} quantization can shrink. Tree ensembles, linear "
                         "models and k-NN keep their parameters inside ONNX-ML operators, which ONNX Runtime "
                         "cannot quantize; quantization helps neural networks (AutoML(..., include=['mlp'])).")
    if mode == "fp16":
        m, _ = _fp16_weights(onnx.load(str(src)))
        onnx.save(m, str(dst))
        return dst
    from onnxruntime.quantization import QuantType, quantize_dynamic
    with _quiet():
        quantize_dynamic(str(src), str(dst), per_channel=per_channel,
                         weight_type=QuantType.QInt8 if mode == "int8" else QuantType.QUInt8)
    return dst


# ── measuring ───────────────────────────────────────────────────────────────
@dataclasses.dataclass
class QuantizedVariant:
    mode: str                         # "fp32" (the unquantized ONNX export), "int8", "uint8", "fp16"
    path: Path | None
    size_bytes: int | None
    compression: float | None         # fp32 size / this size
    latency_ms: float | None          # median time to score 1,000 rows, ONNX Runtime on CPU
    score: float | None               # the metric on the held-out rows
    score_change: float | None        # score - fp32 score (negative = worse, for higher-is-better metrics)
    agreement: float | None           # share of predictions the same as the original model's (regression:
                                      # within REGRESSION_TOLERANCE standard deviations)
    note: str = ""

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["path"] = str(self.path) if self.path else None
        return d


@dataclasses.dataclass
class QuantizationReport:
    metric: str
    higher_is_better: bool
    weights: dict                     # quantizable_weights() of the fp32 export
    variants: list                    # [QuantizedVariant], fp32 first
    max_drop: float
    min_agreement: float
    recommended: QuantizedVariant | None

    def variant(self, mode: str) -> QuantizedVariant | None:
        return next((v for v in self.variants if v.mode == mode), None)

    def drop(self, v: QuantizedVariant) -> float | None:
        """How much worse ``v`` scores than fp32 (positive = worse), whatever the metric direction."""
        if v.score_change is None:
            return None
        return -v.score_change if self.higher_is_better else v.score_change

    def ok(self, v: QuantizedVariant) -> bool:
        d = self.drop(v)
        return v.path is not None and d is not None and d <= self.max_drop and (v.agreement or 0) >= self.min_agreement

    def passed(self, mode: str | None = None) -> bool:
        """``mode`` (default: the recommended one) is within ``max_drop`` and ``min_agreement``."""
        v = self.variant(mode) if mode else self.recommended
        return bool(v is not None and self.ok(v))

    def to_frame(self) -> pl.DataFrame:
        rows = [{**v.as_dict(), "within_limits": self.ok(v) if v.mode != "fp32" else None} for v in self.variants]
        return pl.DataFrame(rows, infer_schema_length=None)

    def summary(self) -> str:
        lines = []
        for v in self.variants:
            if v.path is None:
                lines.append(f"{v.mode:<6} skipped: {v.note}")
                continue
            change = "" if v.mode == "fp32" or v.score_change is None else f" ({v.score_change:+.4f})"
            score = "n/a" if v.score is None else f"{v.score:.4f}"
            mark = "   <- recommended" if self.recommended is v else ""
            lines.append(f"{v.mode:<6} {v.size_bytes / 1024:8.1f} KB  {v.compression:5.2f}x  {self.metric} {score}{change:<11}"
                         f"  agreement {100 * (v.agreement or 0):6.2f}%  {v.latency_ms:7.2f} ms/1k rows{mark}")
        if self.recommended is None:
            lines.append("No quantized variant is recommended: " + (
                "this model has nothing to quantize." if not self.weights["int8_weights"] and not self.weights["fp16_weights"]
                else f"none stays within a {self.max_drop} {self.metric} drop and {self.min_agreement:.0%} agreement."))
        return "\n".join(lines)

    def to_json(self, path) -> Path:
        path = Path(path)
        data = {"metric": self.metric, "higher_is_better": self.higher_is_better, "weights": self.weights,
                "max_drop": self.max_drop, "min_agreement": self.min_agreement,
                "recommended": self.recommended.mode if self.recommended else None,
                "variants": [v.as_dict() for v in self.variants]}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def __repr__(self) -> str:
        rec = self.recommended.mode if self.recommended else None
        return f"QuantizationReport(metric={self.metric!r}, recommended={rec!r}, variants={[v.mode for v in self.variants]})"


def _default_metric(model) -> str:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return "rmse"
    return "roc_auc" if len(classes) == 2 else "accuracy"


def _score(metric: str, y, pred, proba) -> float | None:
    from sklearn import metrics as skm
    try:
        if metric == "roc_auc":
            return float(skm.roc_auc_score(y, proba[:, 1]) if proba.shape[1] == 2
                         else skm.roc_auc_score(y, proba, multi_class="ovr"))
        if metric == "accuracy":
            return float(skm.accuracy_score(y, pred))
        if metric == "f1":
            return float(skm.f1_score(y, pred, average="binary" if proba is not None and proba.shape[1] == 2 else "macro"))
        if metric == "log_loss":
            return float(skm.log_loss(y, proba, labels=list(range(proba.shape[1]))))
        if metric == "rmse":
            return float(np.sqrt(skm.mean_squared_error(y, pred)))
        if metric == "mae":
            return float(skm.mean_absolute_error(y, pred))
        if metric == "r2":
            return float(skm.r2_score(y, pred))
    except ValueError:
        return None
    raise ValueError(f"metric must be one of {sorted(_HIGHER_IS_BETTER)}")


def _run(sess, feeds: dict, classify: bool):
    out = sess.run(None, feeds)
    if classify:
        proba = np.asarray(out[1], dtype=float) if len(out) > 1 else None
        return np.asarray(out[0]).ravel().astype(int), proba
    return np.asarray(out[0], dtype=float).ravel(), None


def _latency(sess, feeds: dict, rows: int, repeats: int = 7) -> float:
    sess.run(None, feeds)                                       # warm-up
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        sess.run(None, feeds)
        times.append(time.perf_counter() - t)
    return float(np.median(times) * 1000 * 1000 / max(rows, 1))


def quantize_model(model, data, *, modes=("int8", "fp16"), out=None, metric: str | None = None,
                   max_drop: float = 0.01, min_agreement: float = 0.99) -> QuantizationReport:
    """Export ``model`` (an AutoML model, a saved model directory or a registry model) to ONNX,
    quantize it in each of ``modes``, and measure every variant on held-out ``data`` (raw rows
    including the target): file size, latency, the metric and agreement with the original model.

    The ``recommended`` variant is the smallest one whose metric is at most ``max_drop`` worse
    than the float32 export and whose predictions agree with the original on at least
    ``min_agreement`` of rows. Files go to ``out`` (default: a temporary directory), together
    with ``quantization.json``."""
    _, ort = _ort()
    from .export import _to_onnx, onnx_inputs
    from .scoring import resolve_model
    model = resolve_model(model)
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise ValueError(f"Unknown modes {bad}; choose from {MODES}")
    out_dir = Path(out) if out is not None else Path(tempfile.mkdtemp(prefix="mlpilot-quantize-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    classify = getattr(model, "classes_", None) is not None
    metric = metric or _default_metric(model)
    if metric not in _HIGHER_IS_BETTER:
        raise ValueError(f"metric must be one of {sorted(_HIGHER_IS_BETTER)}")
    df = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)
    target = getattr(model, "target", None)
    if target is None or target not in df.columns:
        raise ValueError(f"data must contain the target column {target!r} to measure the quantized models")

    fp32 = _to_onnx(model, out_dir / "model.onnx", None)
    feeds = onnx_inputs(model, df)
    native = np.asarray(model.model_.predict(model._features(df)))
    if classify:
        index = {c: i for i, c in enumerate(model.classes_)}
        index_s = {str(c): i for i, c in enumerate(model.classes_)}
        y = np.array([index.get(v, index_s.get(str(v), -1)) for v in df[target].to_list()])
        keep = y >= 0
        native = native.astype(int)
    else:
        y = df[target].cast(pl.Float64).to_numpy()
        keep = np.isfinite(y)

    def measure(mode: str, path: Path, base_size: int | None, base_score: float | None) -> QuantizedVariant:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        pred, proba = _run(sess, feeds, classify)
        if classify:
            agreement = float(np.mean(pred == native))
        else:
            scale = float(np.std(native)) or 1.0
            agreement = float(np.mean(np.abs(pred - native) <= REGRESSION_TOLERANCE * scale))
        score = _score(metric, y[keep], pred[keep], proba[keep] if proba is not None else None)
        size = path.stat().st_size
        return QuantizedVariant(mode, path, size, (base_size or size) / size, _latency(sess, feeds, df.height), score,
                                None if base_score is None or score is None else score - base_score, agreement)

    base = measure("fp32", fp32, None, None)
    weights = quantizable_weights(fp32)
    variants = [base]
    for mode in modes:
        key = "int8_weights" if mode in ("int8", "uint8") else "fp16_weights"
        if not weights[key]:
            variants.append(QuantizedVariant(mode, None, None, None, None, None, None, None,
                                             note="nothing to quantize: this model's parameters live inside ONNX-ML "
                                                  "operators (tree ensemble / linear / k-NN), which ONNX Runtime "
                                                  "cannot quantize"))
            continue
        path = quantize_onnx(fp32, out_dir / f"model.{mode}.onnx", mode)
        variants.append(measure(mode, path, base.size_bytes, base.score))
    rep = QuantizationReport(metric, _HIGHER_IS_BETTER[metric], weights, variants, max_drop, min_agreement, None)
    candidates = [v for v in variants[1:] if rep.ok(v) and (v.compression or 1) > 1.05]
    rep.recommended = min(candidates, key=lambda v: v.size_bytes or 0) if candidates else None
    rep.to_json(out_dir / "quantization.json")
    return rep
