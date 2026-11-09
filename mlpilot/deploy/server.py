"""A REST API for a model, as a standard WSGI app (no web-framework dependency).

Run it with ``serve(model)`` for development, or any WSGI server in production — e.g.
``gunicorn mlpilot.deploy.wsgi:app`` (Linux) / ``waitress-serve`` (Windows), which
is what the generated Docker image does. ``make_asgi_app`` wraps it for uvicorn/hypercorn.

Endpoints::

    GET  /health  /ping           liveness: {"status": "ok"}
    GET  /metadata                model name, task, target, classes, expected input columns, version
    GET  /metrics                 Prometheus metrics (requests, latency, rows, errors, feedback,
                                  live accuracy, reloads; drift + estimated performance with monitor=)
    POST /predict                 {"records": [{...}, ...]}  (or a bare list / a single record)
    POST /feedback                {"prediction_id": ..., "label": ..., "rating": 4, "comment": "..."}
                                  (or {"feedback": [...]}); ids come from each prediction response
    POST /admin/reload            check for a new model now (hot reload; needs the API key)
    POST /invocations             same, AWS SageMaker convention
    POST /score                   same, Azure ML convention
    $AIP_PREDICT_ROUTE / $AIP_HEALTH_ROUTE   Google Vertex AI routes (payload {"instances": [...]})

Responses: ``{"predictions": [...], "probabilities": [{class: p}], "prediction_ids": [...],
"variant": "primary", ...}``. Send outcomes back with ``POST /feedback`` and the prediction id:
they are logged (``feedback_log``), matched to the prediction, and counted as live accuracy.

Hot reload: ``app_from_registry(..., watch=30)`` follows promotions in the registry, and
``make_app(path, watch=30)`` follows a model directory being replaced - no restart.

Rollouts: ``canary=(model, 0.1)`` sends ~10% of requests (chosen by request id, so retries are
stable) to a candidate model; ``shadow=model`` scores every request with a second model in the
background and logs its output (and agreement) without affecting responses.

Security: set ``api_key`` (or the ``AC_API_KEY`` env var) to require an ``X-API-Key`` header
(compared in constant time). Bodies over ``max_body_bytes`` and requests over ``max_rows`` are
rejected. Internal errors return a generic message; details go to the server log.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import polars as pl

from ..quality.checks import jsonable
from .scoring import expected_columns, model_inputs, resolve_model

log = logging.getLogger("mlpilot.deploy")

_STATUS = {200: "200 OK", 400: "400 Bad Request", 401: "401 Unauthorized", 403: "403 Forbidden", 404: "404 Not Found",
           405: "405 Method Not Allowed", 413: "413 Payload Too Large", 415: "415 Unsupported Media Type",
           422: "422 Unprocessable Entity", 500: "500 Internal Server Error"}
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_RECORD_KEYS = ("records", "instances", "data", "inputs")


class _HTTPError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class Metrics:
    """Thread-safe Prometheus counters, gauges and a latency histogram (text exposition format)."""

    def __init__(self, labels: dict):
        self._lock = threading.Lock()
        self._base = labels
        self.requests: dict = {}          # (route, status, variant) -> count
        self.rows: dict = {}              # variant -> rows scored
        self.shadow: dict = {"agree": 0, "disagree": 0}
        self.hist = [0] * (len(_BUCKETS) + 1)
        self.latency_sum = 0.0
        self.feedback: dict = {}          # (variant, result) -> count
        self.abs_error: dict = {}         # variant -> [sum, count]   (regression feedback)
        self.live: dict = {}              # variant -> deque of recent correctness (0/1)
        self.reloads: dict = {"ok": 0, "failed": 0}
        self.gauges: dict = {}            # (name, frozen labels) -> value   (drift, estimates)
        self.gauge_help: dict = {}

    def set_labels(self, **labels):
        with self._lock:
            self._base = {**self._base, **{k: ("" if v is None else v) for k, v in labels.items()}}

    def observe(self, route: str, status: int, variant: str | None, seconds: float, rows: int = 0):
        with self._lock:
            key = (route, str(status), variant or "")
            self.requests[key] = self.requests.get(key, 0) + 1
            if rows and variant:
                self.rows[variant] = self.rows.get(variant, 0) + rows
            if route in ("predict",):
                self.latency_sum += seconds
                i = next((i for i, b in enumerate(_BUCKETS) if seconds <= b), len(_BUCKETS))
                self.hist[i] += 1

    def shadow_result(self, agree: int, disagree: int):
        with self._lock:
            self.shadow["agree"] += agree
            self.shadow["disagree"] += disagree

    def feedback_result(self, variant: str, result: str, *, correct: bool | None = None,
                        abs_error: float | None = None, window: int = 1000):
        from collections import deque
        with self._lock:
            key = (variant or "", result)
            self.feedback[key] = self.feedback.get(key, 0) + 1
            if correct is not None:
                self.live.setdefault(variant, deque(maxlen=window)).append(1 if correct else 0)
            if abs_error is not None:
                acc = self.abs_error.setdefault(variant, [0.0, 0])
                acc[0] += abs_error
                acc[1] += 1

    def reload_result(self, ok: bool):
        with self._lock:
            self.reloads["ok" if ok else "failed"] += 1

    def set_gauges(self, name: str, help_text: str, values: dict):
        """Replace every series of gauge ``name``: ``values`` maps a label dict (as a tuple of
        pairs) to a number."""
        with self._lock:
            self.gauges = {k: v for k, v in self.gauges.items() if k[0] != name}
            for labels, value in values.items():
                if value is not None and value == value:
                    self.gauges[(name, labels)] = float(value)
            self.gauge_help[name] = help_text

    def render(self) -> str:
        def lab(extra: dict) -> str:
            items = {**self._base, **extra}
            return "{" + ",".join(f'{k}="{str(v).replace(chr(34), "")}"' for k, v in items.items()) + "}"
        with self._lock:
            out = ["# HELP ac_model_info Model being served.", "# TYPE ac_model_info gauge", f"ac_model_info{lab({})} 1",
                   "# HELP ac_requests_total Requests by route, HTTP status and model variant.",
                   "# TYPE ac_requests_total counter"]
            out += [f"ac_requests_total{lab({'route': r, 'status': s, 'variant': v})} {n}"
                    for (r, s, v), n in sorted(self.requests.items())]
            out += ["# HELP ac_rows_scored_total Records scored, by model variant.", "# TYPE ac_rows_scored_total counter"]
            out += [f"ac_rows_scored_total{lab({'variant': v})} {n}" for v, n in sorted(self.rows.items())]
            out += ["# HELP ac_shadow_predictions_total Shadow-model predictions that agreed / disagreed.",
                    "# TYPE ac_shadow_predictions_total counter"]
            out += [f"ac_shadow_predictions_total{lab({'result': k})} {n}" for k, n in self.shadow.items()]
            out += ["# HELP ac_predict_duration_seconds Prediction request latency.",
                    "# TYPE ac_predict_duration_seconds histogram"]
            cum = 0
            for b, n in zip(_BUCKETS, self.hist):
                cum += n
                out.append(f"ac_predict_duration_seconds_bucket{lab({'le': b})} {cum}")
            cum += self.hist[-1]
            out += [f"ac_predict_duration_seconds_bucket{lab({'le': '+Inf'})} {cum}",
                    f"ac_predict_duration_seconds_sum{lab({})} {self.latency_sum:.6f}",
                    f"ac_predict_duration_seconds_count{lab({})} {cum}"]
            out += ["# HELP ac_feedback_total Feedback received, by variant and result "
                    "(correct, incorrect, scored, rating, unmatched).", "# TYPE ac_feedback_total counter"]
            out += [f"ac_feedback_total{lab({'variant': v, 'result': r})} {n}"
                    for (v, r), n in sorted(self.feedback.items())]
            out += ["# HELP ac_live_accuracy Accuracy over the most recent feedback (classification).",
                    "# TYPE ac_live_accuracy gauge"]
            out += [f"ac_live_accuracy{lab({'variant': v})} {sum(d) / len(d):.6f}" for v, d in sorted(self.live.items())
                    if len(d)]
            out += ["# HELP ac_feedback_abs_error Absolute error reported through feedback (regression).",
                    "# TYPE ac_feedback_abs_error summary"]
            for v, (tot, cnt) in sorted(self.abs_error.items()):
                out += [f"ac_feedback_abs_error_sum{lab({'variant': v})} {tot:.6f}",
                        f"ac_feedback_abs_error_count{lab({'variant': v})} {cnt}"]
            out += ["# HELP ac_model_reloads_total Hot reloads of the served model.",
                    "# TYPE ac_model_reloads_total counter"]
            out += [f"ac_model_reloads_total{lab({'result': k})} {n}" for k, n in self.reloads.items()]
            for name in sorted({k[0] for k in self.gauges}):
                out += [f"# HELP {name} {self.gauge_help.get(name, name)}", f"# TYPE {name} gauge"]
                out += [f"{name}{lab(dict(labels))} {v:.6g}" for (n, labels), v in sorted(self.gauges.items())
                        if n == name]
        return "\n".join(out) + "\n"


def _metadata(model, name: str | None, version) -> dict:
    from ..automl import AutoCluster
    meta: dict[str, Any] = {"name": name, "version": version, "input_columns": expected_columns(model)}
    if isinstance(model, AutoCluster):
        meta.update(task="clustering", n_clusters=model.best_.get("k"))
    else:
        meta.update(task=model.task_.task, target=model.target, classes=model.classes_,
                    algorithm=model.best_["model"], metric=model.metric_)
    try:
        from importlib.metadata import version as _v
        meta["library_version"] = _v("mlpilot")
    except Exception:
        pass
    return meta


def _scorer(model):
    """records -> output dict, for one model (serialised: preprocessing isn't re-entrant)."""
    from ..automl import AutoCluster
    lock = threading.Lock()
    clustering = isinstance(model, AutoCluster)

    def predict(records: list) -> dict:
        frame = model_inputs(model, pl.DataFrame(records, infer_schema_length=None))
        with lock:
            if clustering:
                return {"clusters": model.predict(frame).to_list()}
            labels, proba = model.predict_with_proba(frame)
            out: dict[str, Any] = {"predictions": labels.to_list()}
            if proba is not None:
                names = [str(c) for c in model.classes_]
                out["probabilities"] = [dict(zip(names, map(float, row))) for row in proba.iter_rows()]
        return out
    return predict


def _tracer(enabled: bool):
    if not enabled:
        return None
    try:
        from opentelemetry import trace
    except ImportError:
        log.warning("tracing=True but opentelemetry-api is not installed; tracing disabled")
        return None
    return trace.get_tracer("mlpilot.deploy")


class _App:
    """The WSGI callable. ``model`` / ``metadata`` always describe what is served *now*
    (they follow hot reloads); ``reload()`` checks for a new model, ``close()`` stops background
    threads."""

    def __init__(self, fn, state, **extra):
        self._fn, self._state = fn, state
        self.__dict__.update(extra)

    def __call__(self, environ, start_response):
        return self._fn(environ, start_response)

    @property
    def model(self):
        return self._state["served"].model

    @property
    def metadata(self) -> dict:
        return self._state["served"].meta


class _Served:
    """What is being served right now; replaced as a whole on hot reload."""

    def __init__(self, model, name, version, canary=None, shadow=None):
        self.model = resolve_model(model)
        self.meta = jsonable(_metadata(self.model, name, version))
        self.variants = {"primary": _scorer(self.model)}
        self.canary_weight = 0.0
        if canary is not None:
            canary_model, weight = canary
            if not 0.0 <= float(weight) <= 1.0:
                raise ValueError("canary weight must be between 0 and 1")
            self.variants["canary"] = _scorer(resolve_model(canary_model))
            self.canary_weight = float(weight)
            self.meta["canary_weight"] = self.canary_weight
        self.shadow = _scorer(resolve_model(shadow)) if shadow is not None else None
        self.classification = getattr(self.model, "classes_", None) is not None


def _watch_dir(path) -> Callable:
    """A reloader that fires when the saved model at ``path`` is replaced (manifest changes)."""
    from pathlib import Path
    manifest = Path(path) / "manifest.json"

    def stamp():
        try:
            return hashlib.sha256(manifest.read_bytes()).hexdigest()
        except OSError:
            return None
    last = {"v": stamp()}

    def check():
        cur = stamp()
        if cur and cur != last["v"]:
            last["v"] = cur
            return {"model": str(path)}
        return None
    return check


def make_app(model, *, api_key: str | None = None, name: str | None = None, version=None,
             max_rows: int = 10_000, max_body_bytes: int = 10 * 1024 * 1024, prediction_log=None,
             canary=None, shadow=None, tracing: bool = False, public_metrics: bool = True,
             feedback_log=None, monitor=None, drift_window: int = 2000, drift_every: int = 500,
             reloader: Callable | None = None, watch: float | None = None, recent_predictions: int = 100_000):
    """Build the WSGI application for ``model`` (an object or a saved-model path).

    * ``prediction_log``: JSONL file to append every request's inputs and outputs to (feed it to
      ``mlpilot.monitoring`` to watch for drift).
    * ``canary=(model, weight)``: route that share of requests to a candidate model.
    * ``shadow=model``: also score every request with this model, in the background; only logged.
    * ``tracing=True``: OpenTelemetry spans per request (needs ``opentelemetry-api`` + an SDK).
    * ``public_metrics=False``: require the API key for ``/metrics`` too.
    * ``feedback_log``: JSONL file for ``POST /feedback`` (default: next to ``prediction_log``).
    * ``monitor``: a fitted :class:`~mlpilot.monitoring.Monitor` (or a saved one's path);
      the last ``drift_window`` inputs are checked every ``drift_every`` rows and the results
      (drift share, per-feature PSI, prediction PSI, estimated performance, active alerts) are
      exported as Prometheus gauges.
    * ``watch=seconds``: hot reload. With a model *path*, a replaced model at that path is picked
      up; with ``reloader`` (see :func:`app_from_registry`), whatever it returns. ``POST
      /admin/reload`` (needs the API key) reloads immediately. In-flight requests finish on the
      old model; the swap is atomic.
    """
    from ..secretstore import resolve
    api_key = resolve(api_key if api_key is not None else os.environ.get("AC_API_KEY") or None)  # vault://, aws-sm://...
    state = {"served": _Served(model, name, version, canary, shadow)}
    served0: _Served = state["served"]
    metrics = Metrics({"model": served0.meta.get("name") or "model", "version": served0.meta.get("version") or ""})
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ac-shadow")   # threads start lazily
    tracer = _tracer(tracing)
    log_lock = threading.Lock()
    if feedback_log is None and prediction_log:
        feedback_log = f"{prediction_log}.feedback.jsonl"
    recent: OrderedDict = OrderedDict()                   # prediction id -> (prediction, variant)
    recent_lock = threading.Lock()
    predict_routes = {"/predict", "/invocations", "/score"}
    health_routes = {"/health", "/ping"}
    if os.environ.get("AIP_PREDICT_ROUTE"):
        predict_routes.add(os.environ["AIP_PREDICT_ROUTE"].rstrip("/") or "/")
    if os.environ.get("AIP_HEALTH_ROUTE"):
        health_routes.add(os.environ["AIP_HEALTH_ROUTE"].rstrip("/") or "/")

    # ── live drift ──────────────────────────────────────────────────────────
    if isinstance(monitor, (str, os.PathLike)):
        from ..monitoring import Monitor
        monitor = Monitor.load(monitor, served0.model)
    window: deque = deque(maxlen=drift_window)
    drift_state = {"since": 0, "running": False}
    drift_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ac-drift") if monitor is not None else None
    if monitor is not None and monitor.reference_metrics:
        # validation metrics, so alert rules can compare live / estimated values with them
        metrics.set_gauges("ac_reference_metric", "The model's validation metrics (from the monitor's reference data).",
                           {(("metric", k),): v for k, v in monitor.reference_metrics.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})

    def run_drift():
        try:
            batch = pl.DataFrame(list(window), infer_schema_length=None)
            rep = monitor.check(batch)
            metrics.set_gauges("ac_drift_share", "Share of input columns with major drift (PSI >= 0.25).",
                               {(): rep.data_drift.drift_share})
            cols = rep.data_drift.columns
            if cols.height and "psi" in cols.columns:
                metrics.set_gauges("ac_feature_psi", "Population stability index per input column.",
                                   {(("feature", r["column"]),): r["psi"] for r in cols.iter_rows(named=True)
                                    if r.get("psi") is not None})
            if rep.prediction_drift:
                metrics.set_gauges("ac_prediction_psi", "PSI of the prediction distribution.",
                                   {(): rep.prediction_drift["psi"]})
            perf = rep.performance or {}
            if perf:
                kind = "estimated" if perf.get("estimated") else "measured"
                metrics.set_gauges("ac_performance", "Model performance on the drift window (estimated when no labels).",
                                   {(("metric", perf["metric"]), ("kind", kind)): perf["current"]})
                metrics.set_gauges("ac_reference_performance", "Reference (validation) performance.",
                                   {(("metric", perf["metric"]),): perf["reference"]})
                metrics.set_gauges("ac_performance_change", "Current minus reference performance (relative for errors).",
                                   {(("metric", perf["metric"]), ("kind", kind)): perf["change"]})
            counts: dict = {}
            for a in rep.alerts:
                key = (("kind", a.kind), ("severity", a.severity))
                counts[key] = counts.get(key, 0) + 1
            metrics.set_gauges("ac_monitor_alerts", "Alerts raised by the latest drift-window check.", counts or {})
            metrics.set_gauges("ac_drift_window_rows", "Rows in the latest drift-window check.", {(): batch.height})
        except Exception:
            log.exception("live drift check failed")
        finally:
            drift_state["running"] = False

    def note_inputs(records):
        if monitor is None:
            return
        window.extend(records)
        drift_state["since"] += len(records)
        if drift_state["since"] >= drift_every and not drift_state["running"] and len(window) >= min(drift_every, 50):
            drift_state["since"], drift_state["running"] = 0, True
            drift_pool.submit(run_drift)

    # ── hot reload ──────────────────────────────────────────────────────────
    if reloader is None and watch and isinstance(model, (str, os.PathLike)):
        reloader = _watch_dir(model)
    reload_lock = threading.Lock()
    stop = threading.Event()

    def reload_now() -> dict:
        """Ask the reloader for a new model; swap it in if there is one."""
        if reloader is None:
            return {"reloaded": False, "reason": "no reloader configured"}
        with reload_lock:
            try:
                change = reloader()
                if not change:
                    return {"reloaded": False, "version": state["served"].meta.get("version")}
                cur = state["served"]
                new = _Served(change["model"], change.get("name", cur.meta.get("name")),
                              change.get("version", cur.meta.get("version")),
                              change.get("canary"), change.get("shadow"))
            except Exception as exc:
                metrics.reload_result(False)
                log.exception("model reload failed; still serving the previous model")
                return {"reloaded": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
            state["served"] = new
            metrics.set_labels(version=new.meta.get("version"))
            metrics.reload_result(True)
            log.info("now serving %s version %s", new.meta.get("name"), new.meta.get("version"))
            return {"reloaded": True, "version": new.meta.get("version")}

    if reloader is not None and watch:
        def loop():
            while not stop.wait(float(watch)):
                reload_now()
        threading.Thread(target=loop, name="ac-model-watch", daemon=True).start()

    # ── request handling ────────────────────────────────────────────────────
    def choose(served: _Served, request_id: str) -> str:
        if served.canary_weight <= 0:
            return "primary"
        bucket = int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "canary" if bucket < served.canary_weight else "primary"

    def write_log(records, payload, request_id, variant, shadow_out=None):
        if not prediction_log:
            return
        outputs = payload.get("predictions", payload.get("clusters"))
        shadow_preds = (shadow_out or {}).get("predictions", (shadow_out or {}).get("clusters")) or [None] * len(records)
        with log_lock, open(prediction_log, "a", encoding="utf-8") as fh:
            for i, (rec, pred) in enumerate(zip(records, outputs)):
                entry = {"time": time.time(), "request_id": request_id, "row": i, "variant": variant,
                         "prediction_id": f"{request_id}-{i}", "version": payload.get("version"),
                         "input": rec, "prediction": pred}
                if shadow_out is not None:
                    entry["shadow_prediction"] = shadow_preds[i]
                fh.write(json.dumps(jsonable(entry), default=str) + "\n")

    def run_shadow(served, records, payload, request_id, variant):
        try:
            out = served.shadow(records)
            a = out.get("predictions", out.get("clusters"))
            b = payload.get("predictions", payload.get("clusters"))
            agree = sum(1 for x, y in zip(a, b) if x == y)
            metrics.shadow_result(agree, len(b) - agree)
            write_log(records, payload, request_id, variant, out)
        except Exception:
            log.exception("shadow model failed for request %s", request_id)

    def read_json(environ):
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
            return json.loads(raw or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _HTTPError(400, "Body is not valid JSON.") from None

    def read_body(environ) -> list:
        body = read_json(environ)
        if isinstance(body, dict):
            key = next((k for k in _RECORD_KEYS if k in body), None)
            records = body[key] if key else body
        else:
            records = body
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
            raise _HTTPError(422, 'Expected {"records": [{column: value, ...}, ...]} with at least one record.')
        if len(records) > max_rows:
            raise _HTTPError(413, f"At most {max_rows} records per request.")
        return records

    def remember(ids, preds, variant, version):
        with recent_lock:
            for pid, pred in zip(ids, preds):
                recent[pid] = (pred, variant, version)
            while len(recent) > recent_predictions:
                recent.popitem(last=False)

    def handle_feedback(environ, request_id):
        body = read_json(environ)
        items = body.get("feedback") if isinstance(body, dict) and "feedback" in body else body
        items = [items] if isinstance(items, dict) else items
        if not isinstance(items, list) or not items or len(items) > max_rows:
            raise _HTTPError(422, 'Expected {"prediction_id": ..., "label": ...} (or {"feedback": [...]}), '
                                  f"at most {max_rows} items.")
        served = state["served"]
        out = {"accepted": 0, "matched": 0, "unknown_ids": []}
        entries = []
        for it in items:
            if not isinstance(it, dict) or not isinstance(it.get("prediction_id"), str) or len(it["prediction_id"]) > 100:
                raise _HTTPError(422, "Each feedback item needs a string prediction_id (from the prediction response).")
            if "label" not in it and "rating" not in it and "comment" not in it:
                raise _HTTPError(422, "Send at least one of: label (the true outcome), rating, comment.")
            rating = it.get("rating")
            if rating is not None and not isinstance(rating, (int, float)):
                raise _HTTPError(422, "rating must be a number.")
            comment = it.get("comment")
            if comment is not None and (not isinstance(comment, str) or len(comment) > 2000):
                raise _HTTPError(422, "comment must be a string of at most 2000 characters.")
            with recent_lock:
                known = recent.get(it["prediction_id"])
            entry = {"time": time.time(), "feedback_request_id": request_id, "prediction_id": it["prediction_id"],
                     "label": it.get("label"), "rating": rating, "comment": comment, "user": it.get("user")}
            variant = None
            if known is not None:
                pred, variant, ver = known
                entry.update(prediction=pred, variant=variant, version=ver)
                out["matched"] += 1
                if it.get("label") is not None:
                    if served.classification:
                        ok = str(pred) == str(it["label"])
                        entry["correct"] = ok
                        metrics.feedback_result(variant, "correct" if ok else "incorrect", correct=ok)
                    else:
                        try:
                            err = abs(float(pred) - float(it["label"]))
                        except (TypeError, ValueError):
                            raise _HTTPError(422, "label must be a number for a regression model.") from None
                        entry["abs_error"] = err
                        metrics.feedback_result(variant, "scored", abs_error=err)
                else:
                    metrics.feedback_result(variant, "rating")
            else:
                out["unknown_ids"].append(it["prediction_id"])
                metrics.feedback_result("", "unmatched")
            entries.append(entry)
            out["accepted"] += 1
        if feedback_log:
            with log_lock, open(feedback_log, "a", encoding="utf-8") as fh:
                for e in entries:
                    fh.write(json.dumps(jsonable(e), default=str) + "\n")
        out["unknown_ids"] = out["unknown_ids"][:50]
        return out

    def handle(environ, request_id, method, path):
        """Returns (status, payload_or_text, route, variant, rows)."""
        served: _Served = state["served"]
        if path in health_routes:
            if method != "GET":
                raise _HTTPError(405, "Use GET.")
            return 200, {"status": "ok"}, "health", None, 0
        if path == "/metrics" and public_metrics:
            if method != "GET":
                raise _HTTPError(405, "Use GET.")
            return 200, metrics.render(), "metrics", None, 0
        if api_key is not None:
            given = environ.get("HTTP_X_API_KEY", "")
            if not hmac.compare_digest(given.encode(), api_key.encode()):
                raise _HTTPError(401, "Missing or invalid X-API-Key header.")
        if path == "/metrics":
            return 200, metrics.render(), "metrics", None, 0
        if path == "/metadata":
            if method != "GET":
                raise _HTTPError(405, "Use GET.")
            return 200, served.meta, "metadata", None, 0
        if path == "/feedback":
            if method != "POST":
                raise _HTTPError(405, "Use POST.")
            return 200, handle_feedback(environ, request_id), "feedback", None, 0
        if path == "/admin/reload":
            if method != "POST":
                raise _HTTPError(405, "Use POST.")
            if api_key is None:
                raise _HTTPError(403, "Reloading needs an API key to be configured (api_key / AC_API_KEY).")
            return 200, reload_now(), "admin", None, 0
        if path in predict_routes:
            if method != "POST":
                raise _HTTPError(405, "Use POST.")
            records = read_body(environ)
            variant = choose(served, request_id)
            try:
                payload = served.variants[variant](records)
            except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError) as exc:
                raise _HTTPError(422, f"Could not score the records: {exc}"[:500]) from None
            payload.update(model=served.meta.get("name"), version=served.meta.get("version"), variant=variant,
                           n=len(records))
            ids = [f"{request_id}-{i}" for i in range(len(records))]
            payload["prediction_ids"] = ids
            remember(ids, payload.get("predictions", payload.get("clusters")), variant, served.meta.get("version"))
            if pool is not None and served.shadow is not None:
                pool.submit(run_shadow, served, records, dict(payload), request_id, variant)
            else:
                write_log(records, payload, request_id, variant)
            note_inputs(records)
            return 200, payload, "predict", variant, len(records)
        raise _HTTPError(404, "Unknown path. Try GET /health, GET /metadata, GET /metrics, POST /predict "
                              "or POST /feedback.")

    def app(environ, start_response):
        request_id = (environ.get("HTTP_X_REQUEST_ID") or uuid.uuid4().hex[:16])[:64]
        method, path = environ.get("REQUEST_METHOD", "GET"), environ.get("PATH_INFO", "/").rstrip("/") or "/"
        t0 = time.perf_counter()
        route, variant, rows = "other", None, 0
        span = tracer.start_as_current_span("ac.request") if tracer else None
        ctx = span.__enter__() if span else None
        try:
            status, payload, route, variant, rows = handle(environ, request_id, method, path)
        except _HTTPError as exc:
            status, payload = exc.status, {"error": exc.message}
        except Exception:
            log.exception("request %s failed", request_id)
            status, payload = 500, {"error": "Internal error.", "request_id": request_id}
        elapsed = time.perf_counter() - t0
        if ctx is not None:
            ctx.set_attribute("http.route", route)
            ctx.set_attribute("http.status_code", status)
            ctx.set_attribute("ac.variant", variant or "")
            ctx.set_attribute("ac.rows", rows)
            span.__exit__(None, None, None)
        metrics.observe(route, status, variant, elapsed, rows)
        if isinstance(payload, str):
            body, ctype = payload.encode(), "text/plain; version=0.0.4"
        else:
            if status == 200 and route == "predict":
                payload["latency_ms"] = round(elapsed * 1000, 2)
            body, ctype = json.dumps(jsonable(payload), default=str).encode(), "application/json"
        start_response(_STATUS[status], [("Content-Type", ctype), ("Content-Length", str(len(body))),
                                          ("X-Request-ID", request_id), ("X-Content-Type-Options", "nosniff"),
                                          ("Cache-Control", "no-store")])
        log.info("%s %s %s %d %.1fms", request_id, method, path, status, elapsed * 1000)
        return [body]

    def _close():
        stop.set()
        for p in (pool, drift_pool):
            if p is not None:
                p.shutdown(wait=True)

    return _App(app, state, metrics=metrics, shadow_pool=pool, drift_pool=drift_pool, reload=reload_now,
                close=_close)


def app_from_registry(registry, name: str, *, canary_stage: str | None = "staging", canary_weight: float = 0.0,
                      shadow_stage: str | None = None, watch: float | None = None, **kw):
    """Serve the production version of a registered model, optionally with the newest
    ``canary_stage`` version as a canary (``canary_weight`` share) or as a shadow.

    ``watch=seconds`` polls the registry and hot-swaps to a newly promoted production version
    (and refreshes the canary / shadow), without a restart."""
    def resolve():
        prod = registry.get(name, stage="production")
        canary = shadow = None
        if canary_weight > 0 and canary_stage:
            try:
                c = registry.get(name, stage=canary_stage)
                canary = (c.model_path, canary_weight) if c.version != prod.version else None
            except KeyError:
                canary = None
        if shadow_stage:
            try:
                shadow = registry.get(name, stage=shadow_stage).model_path
            except KeyError:
                shadow = None
        return prod, canary, shadow

    prod, canary, shadow = resolve()
    seen = {"key": (prod.version, canary[0] if canary else None, shadow)}

    def reloader():
        p, c, s = resolve()
        key = (p.version, c[0] if c else None, s)
        if key == seen["key"]:
            return None
        seen["key"] = key
        return {"model": p.model_path, "version": p.version, "name": name, "canary": c, "shadow": s}

    return make_app(prod.model_path, name=name, version=prod.version, canary=canary, shadow=shadow,
                    reloader=reloader, watch=watch, **kw)


def make_asgi_app(model=None, *, wsgi_app=None, **kw):
    """The same API as an ASGI app (for uvicorn / hypercorn). Requests run in a thread pool."""
    wsgi = wsgi_app or make_app(model, **kw)

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        import io
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        environ = {"REQUEST_METHOD": scope["method"], "PATH_INFO": scope["path"],
                   "QUERY_STRING": scope.get("query_string", b"").decode(), "CONTENT_LENGTH": str(len(body)),
                   "CONTENT_TYPE": headers.get("content-type", ""), "wsgi.input": io.BytesIO(body)}
        for k, v in headers.items():
            environ["HTTP_" + k.upper().replace("-", "_")] = v
        result: dict = {}

        def start_response(status, response_headers):
            result["status"], result["headers"] = int(status.split()[0]), response_headers

        chunks = await asyncio.to_thread(lambda: list(wsgi(environ, start_response)))
        await send({"type": "http.response.start", "status": result["status"],
                    "headers": [(k.lower().encode(), v.encode()) for k, v in result["headers"]]})
        await send({"type": "http.response.body", "body": b"".join(chunks)})

    app.wsgi = wsgi  # type: ignore[attr-defined]
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
    print(f"Serving {app.metadata.get('name') or 'model'} on http://{host}:{httpd.server_port} "  # type: ignore[attr-defined]
          f"(POST /predict, GET /metadata, GET /metrics, GET /health) - Ctrl+C to stop", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
