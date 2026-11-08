"""A REST API for a model, as a standard WSGI app (no web-framework dependency).

Run it with ``serve(model)`` for development, or any WSGI server in production — e.g.
``gunicorn automatedcleaning.deploy.wsgi:app`` (Linux) / ``waitress-serve`` (Windows), which
is what the generated Docker image does.

Endpoints::

    GET  /health     liveness: {"status": "ok"}
    GET  /metadata   model name, task, target, classes, expected input columns, version
    POST /predict    {"records": [{...}, ...]}  (or a bare list of records)
                     -> {"predictions": [...], "probabilities": [{class: p}], ...}

Security: set ``api_key`` (or the ``AC_API_KEY`` env var) to require an ``X-API-Key`` header
(compared in constant time). Bodies over ``max_body_bytes`` and requests over ``max_rows``
are rejected. Internal errors return a generic message; details go to the server log.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
import uuid

import polars as pl

from ..quality.checks import jsonable
from .scoring import expected_columns, model_inputs, resolve_model, to_records

log = logging.getLogger("automatedcleaning.deploy")

_STATUS = {200: "200 OK", 400: "400 Bad Request", 401: "401 Unauthorized", 404: "404 Not Found",
           405: "405 Method Not Allowed", 413: "413 Payload Too Large", 415: "415 Unsupported Media Type",
           422: "422 Unprocessable Entity", 500: "500 Internal Server Error"}


class _HTTPError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _metadata(model, name: str | None, version) -> dict:
    from ..automl import AutoCluster
    meta = {"name": name, "version": version, "input_columns": expected_columns(model)}
    if isinstance(model, AutoCluster):
        meta.update(task="clustering", n_clusters=model.best_.get("k"))
    else:
        meta.update(task=model.task_.task, target=model.target, classes=model.classes_,
                    algorithm=model.best_["model"], metric=model.metric_)
    try:
        from importlib.metadata import version as _v
        meta["library_version"] = _v("automatedcleaning")
    except Exception:
        pass
    return meta


def make_app(model, *, api_key: str | None = None, name: str | None = None, version=None,
             max_rows: int = 10_000, max_body_bytes: int = 10 * 1024 * 1024,
             prediction_log=None):
    """Build the WSGI application for ``model`` (an object or a saved-model path).

    ``prediction_log``: path of a JSONL file to append every request's inputs and outputs to
    (feed it to ``automatedcleaning.monitoring`` to watch for drift)."""
    model = resolve_model(model)
    api_key = api_key if api_key is not None else os.environ.get("AC_API_KEY") or None
    meta = jsonable(_metadata(model, name, version))
    lock = threading.Lock()           # preprocessing redirects stdout; keep predictions serialized
    log_lock = threading.Lock()
    from ..automl import AutoCluster
    clustering = isinstance(model, AutoCluster)

    def predict(records: list) -> dict:
        frame = model_inputs(model, pl.DataFrame(records, infer_schema_length=None))
        with lock:
            if clustering:
                return {"clusters": model.predict(frame).to_list()}
            preds = model.predict(frame).to_list()
            out = {"predictions": preds}
            if model.classes_ is not None:
                proba = model.predict_proba(frame)
                names = [str(c) for c in model.classes_]
                out["probabilities"] = [dict(zip(names, map(float, row))) for row in proba.iter_rows()]
        return out

    def read_body(environ) -> list:
        ctype = environ.get("CONTENT_TYPE", "")
        if ctype and "json" not in ctype:
            raise _HTTPError(415, "Send JSON (Content-Type: application/json).")
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError:
            raise _HTTPError(400, "Invalid Content-Length.") from None
        if length > max_body_bytes:
            raise _HTTPError(413, f"Request body exceeds {max_body_bytes} bytes.")
        raw = environ["wsgi.input"].read(length) if length else b""
        try:
            body = json.loads(raw or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _HTTPError(400, "Body is not valid JSON.") from None
        records = body.get("records", body) if isinstance(body, dict) else body
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
            raise _HTTPError(422, 'Expected {"records": [{column: value, ...}, ...]} with at least one record.')
        if len(records) > max_rows:
            raise _HTTPError(413, f"At most {max_rows} records per request.")
        return records

    def app(environ, start_response):
        request_id = environ.get("HTTP_X_REQUEST_ID") or uuid.uuid4().hex[:16]
        method, path = environ.get("REQUEST_METHOD", "GET"), environ.get("PATH_INFO", "/").rstrip("/") or "/"
        t0 = time.perf_counter()
        status, payload = 200, None
        try:
            if path == "/health":
                if method != "GET":
                    raise _HTTPError(405, "Use GET.")
                payload = {"status": "ok"}
            else:
                if api_key is not None:
                    given = environ.get("HTTP_X_API_KEY", "")
                    if not hmac.compare_digest(given.encode(), api_key.encode()):
                        raise _HTTPError(401, "Missing or invalid X-API-Key header.")
                if path == "/metadata":
                    if method != "GET":
                        raise _HTTPError(405, "Use GET.")
                    payload = meta
                elif path == "/predict":
                    if method != "POST":
                        raise _HTTPError(405, "Use POST.")
                    records = read_body(environ)
                    try:
                        payload = predict(records)
                    except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError) as exc:
                        raise _HTTPError(422, f"Could not score the records: {exc}"[:500]) from None
                    payload.update(model=meta.get("name"), version=meta.get("version"), n=len(records))
                    if prediction_log:
                        with log_lock, open(prediction_log, "a", encoding="utf-8") as fh:
                            outputs = payload.get("predictions", payload.get("clusters"))
                            for rec, pred in zip(records, outputs):
                                fh.write(json.dumps(jsonable({"time": time.time(), "request_id": request_id,
                                                              "input": rec, "prediction": pred}),
                                                    default=str) + "\n")
                else:
                    raise _HTTPError(404, "Unknown path. Try GET /health, GET /metadata or POST /predict.")
        except _HTTPError as exc:
            status, payload = exc.status, {"error": exc.message}
        except Exception:
            log.exception("request %s failed", request_id)
            status, payload = 500, {"error": "Internal error.", "request_id": request_id}
        if status == 200 and isinstance(payload, dict) and path == "/predict":
            payload["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        body = json.dumps(jsonable(payload), default=str).encode()
        start_response(_STATUS[status], [("Content-Type", "application/json"), ("Content-Length", str(len(body))),
                                          ("X-Request-ID", request_id), ("X-Content-Type-Options", "nosniff"),
                                          ("Cache-Control", "no-store")])
        log.info("%s %s %s %d %.1fms", request_id, method, path, status, (time.perf_counter() - t0) * 1000)
        return [body]

    app.model, app.metadata = model, meta
    return app


def serve(model, host: str = "127.0.0.1", port: int = 8000, **kw):
    """Serve the model with a threaded development server (use gunicorn/waitress in production).

    Binds to localhost by default; pass ``host="0.0.0.0"`` to accept outside connections."""
    from socketserver import ThreadingMixIn
    from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

    class _Server(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    class _Quiet(WSGIRequestHandler):
        def log_message(self, *args):
            pass

    app = make_app(model, **kw)
    httpd = make_server(host, port, app, server_class=_Server, handler_class=_Quiet)
    print(f"Serving {app.metadata.get('name') or 'model'} on http://{host}:{httpd.server_port} "
          f"(POST /predict, GET /metadata, GET /health) - Ctrl+C to stop", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
