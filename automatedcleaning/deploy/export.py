"""Exporting models: a portable bundle, a single pickle file, ONNX, or a Docker image context."""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import numpy as np

from ..automl._persist import file_sha256
from .scoring import expected_columns, resolve_model


def export_model(model, path, format: str = "bundle", *, sample=None) -> Path:
    """Export a fitted model.

    * ``"bundle"`` (default): a directory with ``manifest.json``, ``model.joblib`` and
      ``preprocess.json`` — the format ``load_model``/``AutoML.load`` read; checksum-verified.
    * ``"pickle"``: one ``.joblib`` file holding the whole model object, plus a ``.sha256``
      file next to it. Convenient, but only load pickles you trust.
    * ``"onnx"``: the trained estimator as an ONNX graph (needs ``pip install skl2onnx``),
      for ONNX Runtime / other languages. The graph takes the *model-ready* columns (after
      AutoML's cleaning & feature steps); ``preprocess.json`` is written next to it, and
      ``io.json`` lists the input names and types. Pass ``sample`` (raw rows) to check the
      ONNX predictions match (needs ``onnxruntime``).
    """
    model = resolve_model(model)
    path = Path(path)
    if format == "bundle":
        return Path(model.save(path))
    if format == "pickle":
        import joblib
        if path.suffix == "":
            path = path.with_suffix(".joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        path.with_name(path.name + ".sha256").write_text(f"{file_sha256(path)}  {path.name}\n")
        return path
    if format == "onnx":
        return _to_onnx(model, path, sample)
    raise ValueError("format must be 'bundle', 'pickle' or 'onnx'")


def load_pickle(path):
    """Load a ``format="pickle"`` export after checking its ``.sha256`` file."""
    import joblib
    path = Path(path)
    expected = path.with_name(path.name + ".sha256").read_text().split()[0]
    if file_sha256(path) != expected:
        raise ValueError(f"{path}: checksum mismatch - the file was modified or corrupted")
    return joblib.load(path)


def _to_onnx(model, path: Path, sample):
    try:
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    except ImportError as exc:
        raise ImportError("ONNX export needs skl2onnx: pip install skl2onnx onnxruntime") from exc
    kinds = model.feature_kinds_
    inputs = [(c, FloatTensorType([None, 1]) if k == "num" else StringTensorType([None, 1])) for c, k in kinds.items()]
    import copy
    estimator = copy.copy(model.model_)
    if hasattr(estimator, "flatten_transform"):
        estimator.flatten_transform = False   # only affects .transform(); skl2onnx needs it off
    options = {"zipmap": False} if getattr(model, "classes_", None) is not None else None
    try:
        onx = None
        errors = []
        for opset in ({"": 17, "ai.onnx.ml": 3}, None):   # ai.onnx.ml 3 keeps tree ensembles convertible
            try:
                onx = to_onnx(estimator, initial_types=inputs, target_opset=opset, options=options)
                break
            except Exception as exc:          # noqa: PERF203 - try the next opset
                errors.append(exc)
        if onx is None:
            raise errors[0]
    except Exception as exc:
        raise RuntimeError(f"This model can't be converted to ONNX ({type(exc).__name__}: {exc}). "
                           "Use format='bundle' or 'pickle', or exclude unsupported models "
                           "(e.g. AutoML(..., include=[...]))") from exc
    if path.suffix != ".onnx":
        path.mkdir(parents=True, exist_ok=True)
        path = path / "model.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString())
    if getattr(model, "preprocess_", None) is not None:
        model.preprocess_.save(path.parent / "preprocess.json")
    io = {"inputs": [{"name": c, "type": "float32" if k == "num" else "string", "shape": [None, 1]}
                     for c, k in kinds.items()],
          "outputs": [o.name for o in onx.graph.output], "classes": getattr(model, "classes_", None),
          "target": getattr(model, "target", None), "raw_input_columns": expected_columns(model),
          "note": "Inputs are model-ready columns: run preprocess.json (automatedcleaning Pipeline) "
                  "on raw data first, then cast numbers to float32 and fill missing text with '__missing__'."}
    (path.parent / "io.json").write_text(json.dumps(io, indent=2, default=str), encoding="utf-8")
    if sample is not None:
        _verify_onnx(model, path, sample)
    return path


def onnx_inputs(model, df) -> dict:
    """Feed dict for ONNX Runtime from raw rows (applies the model's preprocessing)."""
    X = model._features(df)
    feeds = {}
    for c, k in model.feature_kinds_.items():
        col = X[c].to_numpy()
        feeds[c] = (col.astype(np.float32) if k == "num" else col.astype(str)).reshape(-1, 1)
    return feeds


def _verify_onnx(model, path: Path, sample):
    try:
        import onnxruntime as ort
    except ImportError:
        return
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, onnx_inputs(model, sample))
    expected = model.model_.predict(model._features(sample))
    got = np.asarray(out[0]).ravel()
    if model.classes_ is not None:
        ok = np.mean(got.astype(int) == np.asarray(expected).astype(int)) >= 0.99
    else:
        ok = np.allclose(got.astype(float), np.asarray(expected, dtype=float), rtol=1e-3, atol=1e-3)
    if not ok:
        raise RuntimeError("ONNX predictions don't match the original model on the sample; "
                           "don't use this export.")


# ── Docker ──────────────────────────────────────────────────────────────────

_DOCKERFILE = """\
# Generated by automatedcleaning.deploy.build_docker
# Build:  docker build -t {image} .
# Run:    docker run -p {port}:{port} -e AC_API_KEY=change-me {image}

# --- build stage: compiles the library's Rust extension when no prebuilt wheel exists ---
FROM {base} AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates \\
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${{PATH}}"
COPY requirements.txt /tmp/requirements.txt
COPY dist /tmp/dist
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r /tmp/requirements.txt

# --- runtime stage: small image, no compilers, non-root user ---
FROM {base}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \\
    AC_MODEL_PATH=/app/model AC_MODEL_NAME={name} AC_MODEL_VERSION={version}
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && rm -rf /wheels \\
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY --chown=app:app model /app/model
USER app
EXPOSE {port}
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \\
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:{port}/health').status == 200 else 1)"
CMD ["gunicorn", "--bind", "0.0.0.0:{port}", "--workers", "{workers}", "--timeout", "120", \\
     "--access-logfile", "-", "automatedcleaning.deploy.wsgi:app"]
"""


def build_docker(model, out_dir, *, name: str = "model", version: str = "1", port: int = 8000,
                 workers: int = 2, base_image: str = "python:3.11-slim", library_spec: str | None = None,
                 extra_requirements: list | None = None) -> Path:
    """Write a Docker build context that serves the model over REST (gunicorn + /predict).

    ``out_dir`` gets ``Dockerfile``, ``requirements.txt``, ``.dockerignore``, ``README.md`` and
    the saved ``model/``. Build with ``docker build -t <name> <out_dir>``. The image runs as a
    non-root user, has a health check, and reads ``AC_API_KEY`` to require an API key.
    ``library_spec`` pins how the library is installed: default this version from PyPI; or a
    requirement string; or the path of a local wheel/sdist, which is copied into the context."""
    from ._names import safe_image_name
    model = resolve_model(model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model").exists():
        shutil.rmtree(out / "model")
    model.save(out / "model")
    dist = out / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    dist.mkdir()
    (dist / ".keep").write_text("")
    if library_spec is not None and Path(library_spec).is_file():
        # a local wheel / sdist (e.g. from an internal build): ship it inside the context
        shutil.copy2(library_spec, dist / Path(library_spec).name)
        library_spec = f"/tmp/dist/{Path(library_spec).name}"
    if library_spec is None:
        try:
            from importlib.metadata import version as _v
            library_spec = f"automatedcleaning=={_v('automatedcleaning')}"
        except Exception:
            library_spec = "automatedcleaning"
    import sklearn
    reqs = [library_spec, f"scikit-learn=={sklearn.__version__}", "gunicorn>=22.0", *(extra_requirements or [])]
    manifest = json.loads((out / "model" / "manifest.json").read_text(encoding="utf-8"))
    for pkg in ("lightgbm", "xgboost"):
        if pkg in json.dumps(manifest.get("best", {})) and pkg in manifest.get("versions", {}):
            reqs.append(f"{pkg}=={manifest['versions'][pkg]}")
    (out / "requirements.txt").write_text("\n".join(reqs) + "\n", encoding="utf-8")
    image = safe_image_name(name)
    (out / "Dockerfile").write_text(_DOCKERFILE.format(base=base_image, port=int(port), workers=int(workers),
                                                       name=json.dumps(name), version=json.dumps(str(version)),
                                                       image=image), encoding="utf-8")
    (out / ".dockerignore").write_text("**/__pycache__\n*.pyc\n.git\n", encoding="utf-8")
    (out / "README.md").write_text(textwrap.dedent(f"""\
        # {name} v{version} - model API

        ```bash
        docker build -t {image} .
        docker run -p {port}:{port} -e AC_API_KEY=change-me {image}
        curl -H "X-API-Key: change-me" -H "Content-Type: application/json" \\
             -d '{{"records": [{{...one row of input columns...}}]}}' http://localhost:{port}/predict
        ```

        Input columns: {", ".join(expected_columns(model)) or "(see GET /metadata)"}

        Endpoints: `GET /health`, `GET /metadata`, `POST /predict`. Without `AC_API_KEY` the API is open:
        always set it (or put the service behind an authenticating gateway) outside a private network.
        Optional: `AC_PREDICTION_LOG=/data/predictions.jsonl` (mount a volume) to log inputs and predictions
        for drift monitoring; `AC_MAX_ROWS` caps records per request.
        """), encoding="utf-8")
    return out
