"""Operations around a Monitor: incident alerting, delayed ground-truth labels, result sinks.

* ``pagerduty_alert(routing_key)`` / ``opsgenie_alert(api_key)``  incident tools (Slack/Teams:
  ``webhook_alert``). Critical alerts page; warnings are sent as low priority.
* ``join_labels(prediction_log, labels, on="order_id")``  match labels that arrive days later
  to the predictions that were actually served, then ``Monitor.check_labels(joined)``.
* ``sql_sink(db, "model_monitoring")``  store every check's summary in a database table.
"""

from __future__ import annotations

import dataclasses
import json
import urllib.request
import warnings

import polars as pl

from ..core import ServiceWarning
from ..quality.checks import jsonable


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 5.0) -> bool:
    req = urllib.request.Request(url, data=json.dumps(jsonable(payload)).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        urllib.request.urlopen(req, timeout=timeout).close()      # noqa: S310 - fixed https endpoints
        return True
    except Exception as exc:              # alerting must never break the scoring job
        warnings.warn(f"Alert delivery to {url.split('?')[0]} failed: {exc}", ServiceWarning, stacklevel=3)
        return False


def pagerduty_alert(routing_key: str, *, source: str = "mlpilot", component: str | None = None,
                    url: str = "https://events.pagerduty.com/v2/enqueue", timeout: float = 5.0):
    """Alert handler for PagerDuty Events API v2. One incident per model + alert kind
    (``dedup_key``), so a recurring drift alert updates the open incident instead of paging again.
    ``routing_key`` may be a secret reference (``vault://...``)."""
    from ..secretstore import resolve
    routing_key = resolve(routing_key)
    def send(report, alerts):
        for a in alerts:
            name = getattr(report, "name", None) or component or "model"
            _post_json(url, {
                "routing_key": routing_key, "event_action": "trigger",
                "dedup_key": f"{name}:{a.kind}",
                "payload": {"summary": f"[{name}] {a.kind}: {a.message}"[:1024], "source": source,
                            "severity": "critical" if a.severity == "critical" else "warning",
                            "component": name, "group": "ml-monitoring", "class": a.kind,
                            "custom_details": {"value": a.value, "threshold": a.threshold, "rows": report.rows,
                                               "retrain_recommended": report.retrain_recommended}}}, timeout=timeout)
    return send


def opsgenie_alert(api_key: str, *, region: str = "us", team: str | None = None, timeout: float = 5.0):
    """Alert handler for Opsgenie (``region="eu"`` for EU accounts). Critical -> P1, warning -> P3.
    ``api_key`` may be a secret reference (``vault://...``)."""
    from ..secretstore import resolve
    api_key = resolve(api_key)
    url = f"https://api.{'eu.' if region == 'eu' else ''}opsgenie.com/v2/alerts"

    def send(report, alerts):
        for a in alerts:
            body = {"message": f"{a.kind}: {a.message}"[:130], "alias": f"{getattr(report, 'name', 'model')}:{a.kind}",
                    "description": report.summary()[:15000], "priority": "P1" if a.severity == "critical" else "P3",
                    "tags": ["ml-monitoring", a.kind], "details": {"value": str(a.value), "threshold": str(a.threshold)}}
            if team:
                body["responders"] = [{"name": team, "type": "team"}]
            _post_json(url, body, {"Authorization": f"GenieKey {api_key}"}, timeout=timeout)
    return send


def join_labels(prediction_log, labels: pl.DataFrame, *, on: str, label: str | None = None,
                since: float | None = None) -> pl.DataFrame:
    """Predictions that were served (from the REST API's ``prediction_log``) joined with ground
    truth that arrived later. ``on`` is an input field identifying the entity (e.g. an order id);
    ``label`` the column in ``labels`` holding the outcome (default: the one non-key column).

    Returns the logged inputs plus ``prediction``, ``label``, ``variant`` and ``time``; rows
    whose label hasn't arrived yet are dropped."""
    from ..registry._store import read_jsonl
    rows = [r for r in read_jsonl(prediction_log) if since is None or r.get("time", 0) >= since]
    if not rows:
        return pl.DataFrame()
    served = pl.DataFrame([{**r["input"], "prediction": r["prediction"], "variant": r.get("variant"),
                            "time": r.get("time")} for r in rows], infer_schema_length=None)
    if on not in served.columns:
        raise ValueError(f"'{on}' is not a field of the logged inputs")
    label = label or next(c for c in labels.columns if c != on)
    lab = labels.select(pl.col(on), pl.col(label).alias("label"))
    served = served.with_columns(pl.col(on).cast(lab.schema[on], strict=False))
    # the latest prediction per entity is the one the outcome belongs to
    served = served.sort("time").unique(subset=[on], keep="last", maintain_order=True)
    return served.join(lab, on=on, how="inner")


def join_feedback(prediction_log, feedback_log, *, since: float | None = None) -> pl.DataFrame:
    """Predictions joined with the outcomes users sent to ``POST /feedback`` (matched on the
    prediction id the API returned). Returns the logged inputs plus ``prediction``, ``label``,
    ``variant``, ``version``, ``rating``, ``comment`` and ``time`` - ready for
    ``Monitor.check_labels`` (only rows with a label) or for reviewing ratings and comments.
    The latest feedback per prediction wins."""
    from ..registry._store import read_jsonl
    preds = [r for r in read_jsonl(prediction_log) if since is None or r.get("time", 0) >= since]
    fb = read_jsonl(feedback_log)
    if not preds or not fb:
        return pl.DataFrame()
    served = pl.DataFrame([{**r["input"], "prediction_id": r.get("prediction_id") or f"{r['request_id']}-{r['row']}",
                            "prediction": r["prediction"], "variant": r.get("variant"), "version": r.get("version"),
                            "time": r.get("time")} for r in preds], infer_schema_length=None)
    feedback = (pl.DataFrame([{"prediction_id": f["prediction_id"], "label": f.get("label"), "rating": f.get("rating"),
                               "comment": f.get("comment"), "feedback_time": f.get("time")} for f in fb],
                             infer_schema_length=None)
                .sort("feedback_time").unique(subset=["prediction_id"], keep="last", maintain_order=True))
    return served.join(feedback, on="prediction_id", how="inner")


def sql_sink(db, table: str = "model_monitoring"):
    """``Monitor.on_check`` handler writing one summary row per check to ``table`` (created if
    missing) via an ``ingestion.SQLSource`` — for dashboards in the warehouse/BI tool."""
    def write(report):
        db.insert(table, [{
            "checked_at": report.time, "model": getattr(report, "name", None), "rows": report.rows,
            "drift_share": report.data_drift.drift_share, "drifted": ",".join(report.data_drift.drifted),
            "prediction_psi": (report.prediction_drift or {}).get("psi"),
            "metric": (report.performance or {}).get("metric"), "metric_value": (report.performance or {}).get("current"),
            "alerts": len(report.alerts), "critical": len(report.critical),
            "retrain_recommended": report.retrain_recommended,
            "details": json.dumps([dataclasses.asdict(a) for a in report.alerts], default=str)}])
    return write
