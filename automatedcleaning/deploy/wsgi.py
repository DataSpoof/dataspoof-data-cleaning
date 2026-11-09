"""WSGI entry point for production servers: ``gunicorn automatedcleaning.deploy.wsgi:app``.

Configured by environment variables:

* ``AC_MODEL_PATH``        saved model directory (required)
* ``AC_API_KEY``           if set, requests need a matching ``X-API-Key`` header
* ``AC_MODEL_NAME`` / ``AC_MODEL_VERSION``   shown in /metadata, responses and metrics
* ``AC_MAX_ROWS``          records per request (default 10000)
* ``AC_PREDICTION_LOG``    JSONL file to log inputs and predictions to (for monitoring)
* ``AC_CANARY_MODEL_PATH`` + ``AC_CANARY_WEIGHT``   canary rollout (e.g. 0.1 = 10% of requests)
* ``AC_SHADOW_MODEL_PATH`` shadow model: scored in the background, only logged
* ``AC_TRACING=1``         OpenTelemetry spans (needs opentelemetry-api + an SDK/exporter)
* ``AC_PUBLIC_METRICS=0``  require the API key for /metrics
* ``AC_FEEDBACK_LOG``      JSONL file for POST /feedback (default: next to the prediction log)
* ``AC_MONITOR_PATH``      a saved Monitor: live drift + estimated performance as Prometheus gauges
* ``AC_WATCH_SECONDS``     hot reload: re-check the model (or the registry) every N seconds
* ``AC_REGISTRY`` + ``AC_MODEL_NAME``   serve the registry's production model instead of
                           ``AC_MODEL_PATH`` (with AC_WATCH_SECONDS, promotions go live by themselves)
"""

import logging
import os

from .server import make_app


def build_app():
    watch = float(os.environ["AC_WATCH_SECONDS"]) if os.environ.get("AC_WATCH_SECONDS") else None
    common = dict(max_rows=int(os.environ.get("AC_MAX_ROWS", "10000")),
                  prediction_log=os.environ.get("AC_PREDICTION_LOG") or None,
                  feedback_log=os.environ.get("AC_FEEDBACK_LOG") or None,
                  monitor=os.environ.get("AC_MONITOR_PATH") or None,
                  tracing=os.environ.get("AC_TRACING", "0") == "1",
                  public_metrics=os.environ.get("AC_PUBLIC_METRICS", "1") != "0")
    if os.environ.get("AC_REGISTRY"):
        from ..registry import ModelRegistry
        from .server import app_from_registry
        if not os.environ.get("AC_MODEL_NAME"):
            raise RuntimeError("AC_REGISTRY needs AC_MODEL_NAME (the registered model to serve).")
        return app_from_registry(ModelRegistry(os.environ["AC_REGISTRY"]), os.environ["AC_MODEL_NAME"],
                                 canary_weight=float(os.environ.get("AC_CANARY_WEIGHT", "0")), watch=watch, **common)
    path = os.environ.get("AC_MODEL_PATH")
    if not path:
        raise RuntimeError("Set AC_MODEL_PATH to a saved model directory (or AC_REGISTRY + AC_MODEL_NAME).")
    canary = None
    if os.environ.get("AC_CANARY_MODEL_PATH"):
        canary = (os.environ["AC_CANARY_MODEL_PATH"], float(os.environ.get("AC_CANARY_WEIGHT", "0.1")))
    return make_app(path, name=os.environ.get("AC_MODEL_NAME"), version=os.environ.get("AC_MODEL_VERSION"),
                    canary=canary, shadow=os.environ.get("AC_SHADOW_MODEL_PATH") or None, watch=watch, **common)


logging.basicConfig(level=os.environ.get("AC_LOG_LEVEL", "INFO"))
app = build_app()
