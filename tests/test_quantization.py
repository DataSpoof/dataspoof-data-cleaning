"""Quantization: int8 / fp16 ONNX models, value-preserving column types, quantized embeddings,
and the CI/CD quantization gate."""

import contextlib
import io
import json
import warnings

import numpy as np
import polars as pl
import pytest

import mlpilot as ac
from mlpilot.compression import OptimizeDtypes, dequantize_vectors, optimize_dtypes, quantize_vectors

try:
    import onnxruntime  # noqa: F401
    import skl2onnx  # noqa: F401
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
needs_onnx = pytest.mark.skipif(not HAS_ONNX, reason="needs skl2onnx + onnxruntime ([onnx] extra)")


def _data(n=2400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    logit = X[:, 0] + X[:, 1] * X[:, 2] - X[:, 3] + np.sin(2 * X[:, 4])
    return pl.DataFrame({f"x{i}": X[:, i] for i in range(10)}).with_columns(
        pl.Series("y", (logit + rng.normal(scale=0.5, size=n) > 0).astype(int)),
        pl.Series("reg", 3 * logit + rng.normal(scale=0.3, size=n)))


def _fit(df, target, include, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return ac.AutoML(target, time_budget=20, max_trials=2, include=include, ensemble=False, **kw).fit(df)


@pytest.fixture(scope="module")
def data():
    df = _data()
    return df.head(1800), df.tail(600)


@pytest.fixture(scope="module")
def mlp_model(data):
    train, _ = data
    return _fit(train.drop("reg"), "y", ["mlp"])


@pytest.fixture(scope="module")
def linear_model(data):
    train, _ = data
    return _fit(train.drop("reg"), "y", ["logistic_regression"])


# ── AutoML's opt-in neural network ──────────────────────────────────────────
def test_mlp_is_opt_in(data, mlp_model):
    assert "mlp" in ac.available_models("binary") and "mlp" in ac.available_models("regression")
    assert mlp_model.leaderboard_["model"][0] == "mlp"
    default = _fit(data[0].drop("reg").head(600), "y", None)
    assert "mlp" not in {t["model"] for t in default.trials_}


# ── model quantization ──────────────────────────────────────────────────────
@needs_onnx
def test_quantize_model_classification(data, mlp_model, tmp_path):
    _, test = data
    rep = ac.quantize_model(mlp_model, test.drop("reg"), out=tmp_path)
    fp32, int8, fp16 = (rep.variant(m) for m in ("fp32", "int8", "fp16"))
    assert rep.metric == "roc_auc" and fp32.agreement == 1.0
    assert int8.size_bytes < fp32.size_bytes / 2 and fp16.size_bytes < fp32.size_bytes * 0.7
    assert abs(int8.score_change) < 0.02 and int8.agreement > 0.97
    assert abs(fp16.score_change) < 0.002 and fp16.agreement > 0.995
    assert rep.recommended is not None and rep.passed()
    assert rep.recommended.path.exists() and "recommended" in rep.summary()
    saved = json.loads((tmp_path / "quantization.json").read_text())
    assert saved["recommended"] == rep.recommended.mode and len(saved["variants"]) == 3
    assert rep.to_frame().height == 3


@needs_onnx
def test_quantize_model_regression(data, tmp_path):
    train, test = data
    model = _fit(train.drop("y"), "reg", ["mlp"])
    rep = ac.quantize_model(model, test.drop("y"), out=tmp_path, modes=["fp16"], max_drop=0.05)
    v = rep.variant("fp16")
    assert rep.metric == "rmse" and not rep.higher_is_better
    assert v.agreement > 0.99 and abs(v.score_change) < 0.01 and rep.passed("fp16")


@needs_onnx
def test_models_with_nothing_to_quantize_say_so(data, linear_model, tmp_path):
    _, test = data
    rep = ac.quantize_model(linear_model, test.drop("reg"), out=tmp_path)
    assert rep.recommended is None and not rep.passed()
    assert all(v.path is None and "nothing to quantize" in v.note for v in rep.variants[1:])
    assert rep.weights["int8_weights"] == 0
    assert "nothing to quantize" in rep.summary()
    with pytest.raises(ValueError, match="no weights"):
        ac.quantize_onnx(rep.variants[0].path, tmp_path / "x.onnx", "int8")


@needs_onnx
def test_export_model_quantized(data, mlp_model, tmp_path):
    _, test = data
    path = ac.export_model(mlp_model, tmp_path / "onnx", "onnx", sample=test.drop("reg").head(300), quantize="int8")
    assert path.name == "model.int8.onnx" and (tmp_path / "onnx" / "model.onnx").exists()
    io_info = json.loads((tmp_path / "onnx" / "io.json").read_text())
    assert io_info["quantized"]["mode"] == "int8" and io_info["quantized"]["agreement_on_sample"] > 0.95
    with pytest.raises(ValueError, match="format='onnx'"):
        ac.export_model(mlp_model, tmp_path / "b", "bundle", quantize="int8")
    with pytest.raises(ValueError, match="mode"):
        ac.quantize_onnx(path, tmp_path / "y.onnx", "int4")


@needs_onnx
def test_quantization_gate(data, mlp_model, linear_model, tmp_path):
    from mlpilot import cicd
    _, test = data
    gates = {"quantization": {"mode": "fp16", "max_drop": 0.01}}
    res = cicd.gate(mlp_model, test.drop("reg"), gates)
    assert res["passed"], res["failures"]
    assert res["checks"]["quantization"]["passed"] and res["checks"]["quantization"]["compression"] > 1.3
    res = cicd.gate(linear_model, test.drop("reg"), gates)            # nothing to quantize: passes, with a note
    assert res["passed"] and res["checks"]["quantization"]["skipped"]
    res = cicd.gate(linear_model, test.drop("reg"), {"quantization": {"mode": "int8", "required": True}})
    assert not res["passed"] and "nothing to quantize" in res["failures"][0]
    strict = {"quantization": {"mode": "int8", "max_drop": -1.0}}     # impossible limit
    assert not cicd.gate(mlp_model, test.drop("reg"), strict)["passed"]


@needs_onnx
def test_quantize_cli(data, mlp_model, tmp_path, capsys):
    from mlpilot.deploy.cli import main
    _, test = data
    mlp_model.save(tmp_path / "model")
    test.drop("reg").write_parquet(tmp_path / "test.parquet")
    main(["quantize", "--model", str(tmp_path / "model"), "--data", str(tmp_path / "test.parquet"),
          "--out", str(tmp_path / "q"), "--modes", "fp16"])
    assert "fp16" in capsys.readouterr().out and (tmp_path / "q" / "model.fp16.onnx").exists()


# ── value-preserving column types ───────────────────────────────────────────
def _frame(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "id": np.arange(n, dtype=np.int64), "big_id": np.arange(n, dtype=np.int64) + 5_000_000_000,
        "flag": rng.integers(0, 2, n), "neg": -rng.integers(0, 100, n),
        "price": np.round(rng.uniform(1, 500, n), 2), "balance": np.round(rng.uniform(1e8, 2e8, n), 2),
        "ratio": rng.normal(size=n), "city": rng.choice(["Pune", "Delhi", "Mumbai"], n),
        "name": [f"user{i}" for i in range(n)], "maybe": [None if i % 3 else i for i in range(n)]})


def test_optimize_dtypes_keeps_values():
    df = _frame()
    small, rep = optimize_dtypes(df)
    assert small.schema["id"] == pl.Int16 and small.schema["flag"] == pl.Int8 and small.schema["neg"] == pl.Int8
    assert small.schema["big_id"] == pl.Int64                      # does not fit Int32
    assert small.schema["price"] == pl.Float32                     # 2 decimals survive Float32
    assert small.schema["balance"] == pl.Float64                   # 9 digits + cents would not
    assert small.schema["ratio"] == pl.Float64                     # continuous values: left exact
    assert small.schema["city"] == pl.Categorical and small.schema["name"] == pl.String
    assert small["price"].cast(pl.Float64).round(2).equals(df["price"])
    for c in ("id", "flag", "neg", "maybe"):
        assert small[c].cast(pl.Int64).equals(df[c].cast(pl.Int64))
    assert rep.ratio > 1.3 and rep.bytes_after < rep.bytes_before
    assert set(rep.changed) >= {"id", "flag", "price", "city"} and "price" in rep.summary()


def test_optimize_dtypes_options():
    df = _frame()
    lossy, rep = optimize_dtypes(df, floats="float32", categorical=None)
    assert lossy.schema["ratio"] == pl.Float32 and lossy.schema["balance"] == pl.Float32
    assert lossy.schema["city"] == pl.String
    assert rep.frame.filter(pl.col("column") == "balance")["max_change"][0] > 0.5   # reported, not hidden
    untouched, _ = optimize_dtypes(df, floats=None, exclude=["id"])
    assert untouched.schema["price"] == pl.Float64 and untouched.schema["id"] == pl.Int64
    assert optimize_dtypes(df, unsigned=True)[0].schema["flag"] == pl.UInt8
    with pytest.raises(ValueError):
        optimize_dtypes(df, floats="half")


def test_optimize_dtypes_step_learns_types_and_never_overflows(tmp_path):
    df = _frame()
    pipe = ac.Pipeline([OptimizeDtypes()]).fit(df)
    reloaded = ac.Pipeline.load(pipe.save(tmp_path / "p.json"))
    assert reloaded.transform(df.head(10)).schema == pipe.transform(df).schema
    new = df.with_columns(pl.col("flag") + 1000)                   # outside the learned Int8 range
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = reloaded.transform(new)
    assert out.schema["flag"] == pl.Int64 and out["flag"].equals(new["flag"])
    assert any("flag" in str(x.message) for x in w)


def test_save_cleaned_data_handles_large_ids(tmp_path):
    df = pl.DataFrame({"id": [1, 5_000_000_000], "amount": [19.99, 5.5]})
    out = ac.save_cleaned_data(df, str(tmp_path / "c.csv"))
    assert out["id"].to_list() == [1, 5_000_000_000]
    assert pl.read_csv(tmp_path / "c.csv")["amount"].to_list() == [19.99, 5.5]


def test_clip_outliers_on_small_integer_types():
    df = pl.DataFrame({"v": np.r_[np.zeros(95), np.full(5, 100)].astype(np.int8)}).with_columns(pl.col("v").cast(pl.Int8))
    out = ac.ClipOutliers(k=50).fit_transform(df)                   # fences far outside Int8's range
    assert out.schema["v"] == pl.Int8 and out["v"].max() == 100


# ── quantized vectors / embeddings ──────────────────────────────────────────
def test_quantize_vectors():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 32))
    for mode, dtype, nbytes in (("float32", np.float32, 4), ("float16", np.float16, 2), ("int8", np.int8, 1)):
        q, p = quantize_vectors(x, mode)
        assert q.dtype == dtype and q.nbytes == x.size * nbytes
        back = dequantize_vectors(q, p)
        cos = np.sum(back * x, 1) / np.linalg.norm(back, axis=1) / np.linalg.norm(x, axis=1)
        assert cos.min() > 0.999
    _, p = quantize_vectors(x, "int8")
    q, _ = quantize_vectors(x * 10, "int8", scale=p["scale"])      # beyond the training range: clipped
    assert q.max() == 127 and q.min() == -127
    with pytest.raises(ValueError):
        quantize_vectors(x, "int4")


def test_text_embeddings_dtypes(tmp_path):
    rng = np.random.default_rng(0)
    words = "fast slow great awful delivery product refund support friendly broken price quality".split()
    df = pl.DataFrame({"review": [" ".join(rng.choice(words, 8)) for _ in range(300)]})
    for dtype, pl_type in (("float32", pl.Float32), ("float16", pl.Float16), ("int8", pl.Int8)):
        pipe = ac.Pipeline([ac.TextEmbeddings(columns=["review"], n_components=8, dtype=dtype)]).fit(df)
        out = pipe.transform(df)
        assert out.schema["review_emb0"] == pl_type
        reloaded = ac.Pipeline.load(pipe.save(tmp_path / f"{dtype}.json"))
        assert reloaded.transform(df).equals(out)
    full = ac.TextEmbeddings(columns=["review"], n_components=8).fit_transform(df)
    i8 = ac.TextEmbeddings(columns=["review"], n_components=8, dtype="int8").fit_transform(df)
    a = full.select(pl.all()).to_numpy()
    b = i8.select(pl.all()).to_numpy().astype(float)
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    assert np.median(cos) > 0.99                                    # same directions, 8x smaller
    with pytest.raises(ValueError):
        ac.TextEmbeddings(dtype="int4")
