"""WSGI entry point for production servers: ``gunicorn automatedcleaning.deploy.wsgi:app``.

Configured by environment variables:

* ``AC_MODEL_PATH``      saved model directory (required)
* ``AC_API_KEY``         if set, requests need a matching ``X-API-Key`` header
* ``AC_MODEL_NAME`` / ``AC_MODEL_VERSION``   shown in /metadata and responses
* ``AC_MAX_ROWS``        records per request (default 10000)
* ``AC_PREDICTION_LOG``  JSONL file to log inputs and predictions to (for monitoring)
"""

import logging
import os

from .server import make_app

logging.basicConfig(level=os.environ.get("AC_LOG_LEVEL", "INFO"))

_path = os.environ.get("AC_MODEL_PATH")
if not _path:
    raise RuntimeError("Set AC_MODEL_PATH to a saved model directory.")

app = make_app(_path, name=os.environ.get("AC_MODEL_NAME"), version=os.environ.get("AC_MODEL_VERSION"),
               max_rows=int(os.environ.get("AC_MAX_ROWS", "10000")),
               prediction_log=os.environ.get("AC_PREDICTION_LOG") or None)
