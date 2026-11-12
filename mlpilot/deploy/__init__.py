"""Deployment (module 8): scoring, a REST API, batch jobs, exports and Docker images.

* ``score(model, df)`` / ``batch_score(model, "in.csv", "out.parquet")``  predictions,
  streaming files of any size in chunks.
* ``make_app(model)`` / ``serve(model)``   a REST API (WSGI; gunicorn/waitress-ready), with
  ``POST /feedback``, hot reload (``watch=``) and live drift gauges (``monitor=``).
* ``deploy.aws``                          ECR push + blue/green SageMaker endpoints (needs boto3).
* ``export_model(model, path, format="bundle" | "pickle" | "onnx", quantize=None | "int8" | "fp16")``.
* ``quantize_model(model, test_df)``   int8 / fp16 ONNX variants, measured (size, speed, metric).
* ``build_docker(model, "build/")``   a ready-to-build Docker context (non-root, health check).
* CLI: ``python -m mlpilot.deploy serve|score|docker|export|monitor``.
"""

from .export import build_docker, export_model, load_pickle, load_skops, onnx_inputs
from .quantize import QuantizationReport, quantizable_weights, quantize_model, quantize_onnx
from .scoring import batch_score, expected_columns, resolve_model, score
from .server import app_from_registry, make_app, make_asgi_app, serve

__all__ = ["score", "batch_score", "expected_columns", "resolve_model", "make_app", "serve", "export_model",
           "load_pickle", "load_skops", "onnx_inputs", "build_docker", "make_asgi_app", "app_from_registry",
           "quantize_model", "quantize_onnx", "quantizable_weights", "QuantizationReport"]
