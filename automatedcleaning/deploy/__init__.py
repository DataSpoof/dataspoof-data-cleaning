"""Deployment (module 8): scoring, a REST API, batch jobs, exports and Docker images.

* ``score(model, df)`` / ``batch_score(model, "in.csv", "out.parquet")``  predictions,
  streaming files of any size in chunks.
* ``make_app(model)`` / ``serve(model)``   a REST API (WSGI; gunicorn/waitress-ready).
* ``export_model(model, path, format="bundle" | "pickle" | "onnx")``.
* ``build_docker(model, "build/")``   a ready-to-build Docker context (non-root, health check).
* CLI: ``python -m automatedcleaning.deploy serve|score|docker|export|monitor``.
"""

from .export import build_docker, export_model, load_pickle, onnx_inputs
from .scoring import batch_score, expected_columns, resolve_model, score
from .server import make_app, serve

__all__ = ["score", "batch_score", "expected_columns", "resolve_model", "make_app", "serve", "export_model",
           "load_pickle", "onnx_inputs", "build_docker"]
