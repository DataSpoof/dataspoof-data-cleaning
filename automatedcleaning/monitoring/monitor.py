"""Production monitoring: data drift, prediction drift, accuracy decay, alerts and retraining."""

from __future__ import annotations

from typing import Any

import dataclasses
import datetime as _dt
import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from ..core import ServiceWarning
from ..quality.checks import jsonable
from .drift import Baseline, DriftReport, compare, compare_column, fit_column, severity_of


@dataclasses.dataclass
class RetrainPolicy:
    """When a batch should raise alerts and trigger retraining.

    * ``max_drift_share``: share of monitored columns with *major* drift (PSI ≥ 0.25) → critical
    * ``prediction_psi``: PSI of the prediction distribution → warning (critical at 2×)
    * ``max_metric_drop``: absolute drop of a higher-is-better metric (accuracy, AUC, R²) → critical
    * ``max_error_increase``: relative increase of RMSE / MAE → critical
    * ``min_rows``: smaller batches get a warning that the statistics are unreliable
    """

    max_drift_share: float = 0.3
    prediction_psi: float = 0.2
    max_metric_drop: float = 0.05
    max_error_increase: float = 0.10
    min_rows: int = 100


@dataclasses.dataclass
class Alert:
    severity: str                  # warning | critical
    kind: str                      # data_drift | prediction_drift | performance | schema | volume
    message: str
    value: float | None = None
    threshold: float | None = None


@dataclasses.dataclass
class MonitorReport:
    time: str
    rows: int
    data_drift: DriftReport
    prediction_drift: dict | None
    performance: dict | None
    alerts: list
    retrain_recommended: bool
    name: str | None = None

    @property
    def critical(self) -> list:
        return [a for a in self.alerts if a.severity == "critical"]

    def summary(self) -> str:
        status = "RETRAIN RECOMMENDED" if self.retrain_recommended else ("warnings" if self.alerts else "healthy")
        lines = [f"Monitoring check {self.time}: {self.rows:,} rows - {status}"]
        for a in self.alerts:
            lines.append(f"  [{a.severity}] {a.kind}: {a.message}")
        if self.performance:
            p = self.performance
            lines.append(f"  Performance: {p['metric']} {p['current']:.4f} (reference {p['reference']:.4f})")
        if self.prediction_drift:
            lines.append(f"  Prediction drift: PSI {self.prediction_drift['psi']:.3f} ({self.prediction_drift['severity']})")
        lines.append("  " + self.data_drift.summary().replace("\n", "\n  "))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return jsonable({"time": self.time, "rows": self.rows, "retrain_recommended": self.retrain_recommended,
                         "alerts": [dataclasses.asdict(a) for a in self.alerts],
                         "performance": self.performance, "prediction_drift": self.prediction_drift,
                         "data_drift": self.data_drift.to_dict()})

    def __repr__(self) -> str:
        return self.summary()


_HIGHER_BETTER = ("roc_auc", "accuracy", "f1_macro", "balanced_accuracy", "r2", "roc_auc_ovr")
_LOWER_BETTER = ("rmse", "mae")


def _headline(metrics: dict, task: str) -> str:
    order = ("roc_auc", "roc_auc_ovr", "f1_macro", "accuracy") if task != "regression" else ("rmse", "mae", "r2")
    return next(m for m in order if metrics.get(m) is not None)


def webhook_alert(url: str, *, timeout: float = 5.0, headers: dict | None = None):
    """An alert handler that POSTs each report (JSON) to ``url`` (Slack/Teams incoming webhooks,
    or any HTTP endpoint). For incident tools see ``ops.pagerduty_alert`` / ``ops.opsgenie_alert``."""
    import urllib.request

    def send(report: MonitorReport, alerts: list):
        payload = {"text": report.summary().splitlines()[0], "alerts": [dataclasses.asdict(a) for a in alerts],
                   "retrain_recommended": report.retrain_recommended, "time": report.time, "model": report.name}
        req = urllib.request.Request(url, data=json.dumps(jsonable(payload)).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            urllib.request.urlopen(req, timeout=timeout).close()
        except Exception as exc:          # alerting must never break the scoring job
            warnings.warn(f"Alert webhook failed: {exc}", ServiceWarning, stacklevel=2)
    return send


class Monitor:
    """Watches a deployed model.

    >>> mon = Monitor(model, reference=holdout, log_path="monitor.jsonl")
    >>> report = mon.check(todays_batch)            # labels optional: include the target column
    >>> report.retrain_recommended, report.alerts
    >>> mon.save("monitor.json")                    # baselines only - no raw data

    Data drift is checked on the raw input columns the model uses; prediction drift on its
    outputs; performance (when the batch has labels) against the reference metrics.
    """

    def __init__(self, model, reference=None, *, policy: RetrainPolicy | None = None, log_path=None,
                 columns=None, reference_metrics: dict | None = None, name: str | None = None,
                 estimate_performance: bool = True):
        from ..deploy.scoring import resolve_model
        self.model = resolve_model(model)
        self.policy = policy or RetrainPolicy()
        self.log_path = Path(log_path) if log_path else None
        self.columns, self.name = columns, name
        self.reference_metrics = reference_metrics
        self.baseline_: Baseline | None = None
        self.prediction_baseline_: Any = None
        self.estimate_performance = estimate_performance
        self.estimator_: Any = None
        self._handlers: list = []
        self._check_handlers: list = []
        if reference is not None:
            self.fit(reference)

    # ── helpers ─────────────────────────────────────────────────────────────
    @property
    def _supervised(self) -> bool:
        return hasattr(self.model, "classes_")

    @property
    def _task(self) -> str:
        return self.model.task_.task if self._supervised else "clustering"

    def _outputs(self, df: pl.DataFrame) -> pl.Series:
        """The prediction signal to watch: P(positive) for binary, the label for multiclass,
        the value for regression, the cluster for clustering."""
        if not self._supervised:
            return pl.Series("prediction", self.model.predict(df).to_numpy())
        if self.model.classes_ is not None and len(self.model.classes_) == 2:
            return self.model.predict_proba(df)[:, 1].alias("prediction")
        pred = self.model.predict(df)
        return pred.cast(pl.String).alias("prediction") if self.model.classes_ is not None else pred.alias("prediction")

    def _input_columns(self, df) -> list:
        from ..deploy.scoring import expected_columns
        cols = self.columns or expected_columns(self.model) or [c for c in df.columns]
        target = getattr(self.model, "target", None)
        return [c for c in cols if c != target]

    def _metrics(self, df: pl.DataFrame) -> dict | None:
        target = getattr(self.model, "target", None)
        if not self._supervised or target not in df.columns or df[target].drop_nulls().len() < 2:
            return None
        from ..evaluation import evaluate_model
        try:
            return evaluate_model(self.model, df, bootstrap=0).metrics
        except ValueError:
            return None

    # ── fitting / checking ──────────────────────────────────────────────────
    def fit(self, reference) -> "Monitor":
        """Learn the reference distributions (use held-out data, not the training set, so the
        reference metrics are realistic)."""
        df = reference if isinstance(reference, pl.DataFrame) else pl.from_pandas(reference)
        self.baseline_ = Baseline.fit(df, self._input_columns(df))
        self.prediction_baseline_ = fit_column(self._outputs(df))
        if self.reference_metrics is None:
            self.reference_metrics = self._metrics(df)
        self.reference_rows_ = df.height
        target = getattr(self.model, "target", None)
        if self.estimate_performance and self._supervised and target in df.columns:
            from .estimate import PerformanceEstimator
            try:                              # lets check() estimate performance before labels arrive
                self.estimator_ = PerformanceEstimator(self.model).fit(df)
            except ValueError:
                self.estimator_ = None
        return self

    def on_check(self, handler) -> "Monitor":
        """Register ``handler(report)``; called after every check (e.g. ``ops.sql_sink(db)``)."""
        self._check_handlers.append(handler)
        return self

    def check_labels(self, joined: pl.DataFrame, *, prediction: str = "prediction", label: str = "label") -> dict:
        """Live performance from predictions that were actually served and their ground truth
        (see ``monitoring.ops.join_labels``), compared with the reference metrics. Raises the
        same ``performance`` alerts as :meth:`check` and calls the alert handlers."""
        from sklearn import metrics as m
        if joined.height == 0:
            raise ValueError("No labelled predictions to evaluate yet.")
        y_pred, y_true = joined[prediction].to_list(), joined[label].to_list()
        if self._task == "regression":
            yp, yt = np.asarray(y_pred, dtype=float), np.asarray(y_true, dtype=float)
            current = {"rmse": float(np.sqrt(np.mean((yp - yt) ** 2))), "mae": float(np.mean(np.abs(yp - yt))),
                       "r2": float(m.r2_score(yt, yp))}
            metric = "rmse"
        else:
            yp_s, yt_s = [str(v) for v in y_pred], [str(v) for v in y_true]
            current = {"accuracy": float(m.accuracy_score(yt_s, yp_s)),
                       "balanced_accuracy": float(m.balanced_accuracy_score(yt_s, yp_s)),
                       "f1_macro": float(m.f1_score(yt_s, yp_s, average="macro", zero_division=0))}
            metric = "accuracy"
        ref = (self.reference_metrics or {}).get(metric)
        alerts: list = []
        out: dict = {"rows": joined.height, "metrics": current, "metric": metric, "reference": ref,
                     "current": current[metric], "alerts": alerts}
        if ref is not None:
            pol = self.policy
            if metric in _LOWER_BETTER:
                change = (current[metric] - ref) / ref if ref else 0.0
                bad, msg = change > pol.max_error_increase, f"live {metric} rose {change:.0%} ({ref:.4g} -> {current[metric]:.4g})"
            else:
                change = current[metric] - ref
                bad, msg = -change > pol.max_metric_drop, f"live {metric} fell {-change:.3f} ({ref:.4f} -> {current[metric]:.4f})"
            out["change"] = round(change, 6)
            if bad:
                alerts.append(Alert("critical", "performance", msg, change,
                                    pol.max_error_increase if metric in _LOWER_BETTER else pol.max_metric_drop))
        if "variant" in joined.columns and joined["variant"].n_unique() > 1:
            out["by_variant"] = {str(v): float(np.mean([str(a) == str(b) for a, b in zip(g[prediction], g[label])]))
                                 if self._task != "regression" else
                                 float(np.sqrt(np.mean((g[prediction].cast(pl.Float64) - g[label].cast(pl.Float64)).to_numpy() ** 2)))
                                 for (v,), g in joined.group_by("variant")}
        if alerts:
            report = MonitorReport(time=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                                   rows=joined.height, prediction_drift=None,
                                   data_drift=DriftReport(pl.DataFrame(schema={"column": pl.String, "severity": pl.String}),
                                                          [], [], {}, getattr(self, "reference_rows_", 0), 0),
                                   performance={**out, "metrics": current}, alerts=alerts, retrain_recommended=True,
                                   name=self.name)
            for handler in self._handlers:
                handler(report, alerts)
        return out

    def on_alert(self, handler) -> "Monitor":
        """Register ``handler(report, alerts)``; called whenever a check raises alerts."""
        self._handlers.append(handler)
        return self

    def check(self, batch) -> MonitorReport:
        if self.baseline_ is None:
            raise RuntimeError("Call fit(reference) first (or Monitor.load).")
        df = batch if isinstance(batch, pl.DataFrame) else pl.from_pandas(batch)
        pol, alerts = self.policy, []
        drift = compare(self.baseline_, df)
        if df.height < pol.min_rows:
            alerts.append(Alert("warning", "volume", f"only {df.height} rows; drift statistics are unreliable below "
                                                     f"{pol.min_rows}", df.height, pol.min_rows))
        if drift.missing_columns:
            alerts.append(Alert("critical", "schema", f"input columns missing: {drift.missing_columns}"))
        if drift.dtype_changes:
            alerts.append(Alert("warning", "schema", f"column types changed: {drift.dtype_changes}"))
        if drift.columns.height and drift.drift_share >= pol.max_drift_share:
            alerts.append(Alert("critical", "data_drift", f"{drift.drift_share:.0%} of columns drifted strongly: "
                                                          f"{drift.major[:8]}", drift.drift_share, pol.max_drift_share))
        elif drift.major:
            alerts.append(Alert("warning", "data_drift", f"major drift in {drift.major[:8]}",
                                drift.drift_share, pol.max_drift_share))

        pred_drift = None
        if df.height and self.prediction_baseline_ is not None:
            try:
                outputs = self._outputs(df)
            except Exception as exc:
                outputs = None
                alerts.append(Alert("critical", "schema", f"the model could not score this batch: {exc}"[:300]))
            if outputs is not None:
                row = compare_column(self.prediction_baseline_, outputs)
                pred_drift = {"psi": row["psi"], "severity": severity_of(row["psi"], pol.prediction_psi / 2,
                                                                         pol.prediction_psi),
                              "mean_ref": row["mean_ref"], "mean_cur": row["mean_cur"]}
                if row["psi"] >= pol.prediction_psi:
                    sev = "critical" if row["psi"] >= 2 * pol.prediction_psi else "warning"
                    alerts.append(Alert(sev, "prediction_drift", f"prediction distribution shifted (PSI {row['psi']:.3f})",
                                        row["psi"], pol.prediction_psi))

        perf = None
        current = self._metrics(df)
        if current is None and self.estimator_ is not None and df.height:
            perf = self._estimated(df, alerts)
        if current and self.reference_metrics:
            metric = _headline(self.reference_metrics, self._task)
            ref, cur = self.reference_metrics[metric], current.get(metric)
            if cur is not None:
                if metric in _LOWER_BETTER:
                    change = (cur - ref) / ref if ref else 0.0
                    bad, limit = change > pol.max_error_increase, pol.max_error_increase
                    msg = f"{metric} rose {change:.0%} ({ref:.4g} -> {cur:.4g})"
                else:
                    change = cur - ref
                    bad, limit = -change > pol.max_metric_drop, pol.max_metric_drop
                    msg = f"{metric} fell {-change:.3f} ({ref:.4f} -> {cur:.4f})"
                perf = {"metric": metric, "reference": ref, "current": cur, "change": round(change, 6),
                        "metrics": current}
                if bad:
                    alerts.append(Alert("critical", "performance", msg, change, limit))

        report = MonitorReport(time=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                               rows=df.height, data_drift=drift, prediction_drift=pred_drift, performance=perf,
                               alerts=alerts, retrain_recommended=any(a.severity == "critical" and
                                                                      a.kind in ("data_drift", "performance",
                                                                                 "prediction_drift")
                                                                      for a in alerts), name=self.name)
        if self.log_path is not None:
            from ..registry._store import append_jsonl
            append_jsonl(self.log_path, {"time": report.time, "name": self.name, "rows": report.rows,
                                         "drift_share": drift.drift_share, "drifted": drift.drifted,
                                         "prediction_psi": pred_drift["psi"] if pred_drift else None,
                                         "metric": (("estimated_" if perf.get("estimated") else "") + perf["metric"])
                                         if perf else None,
                                         "metric_value": perf["current"] if perf else None,
                                         "alerts": [dataclasses.asdict(a) for a in alerts],
                                         "retrain_recommended": report.retrain_recommended})
        if alerts:
            for handler in self._handlers:
                handler(report, alerts)
        for handler in self._check_handlers:
            try:
                handler(report)
            except Exception as exc:          # a sink failing must not lose the check result
                warnings.warn(f"Monitor check handler failed: {exc}", ServiceWarning, stacklevel=2)
        return report

    def _estimated(self, df: pl.DataFrame, alerts: list) -> dict | None:
        """Label-free performance estimate (CBPE / direct loss estimation); warns on a drop."""
        try:
            est = self.estimator_.estimate(df)
        except Exception as exc:
            warnings.warn(f"Performance estimation failed: {exc}", ServiceWarning, stacklevel=3)
            return None
        pol = self.policy
        use_auc = "roc_auc" in est and (self.reference_metrics or {}).get("roc_auc") is not None
        if use_auc:
            metric, value, ref = "roc_auc", est["roc_auc"]["estimated"], est["roc_auc"]["reference"]
            low, high = est["roc_auc"]["low"], est["roc_auc"]["high"]
        else:
            metric, value, ref, low, high = est["metric"], est["estimated"], est["reference"], est["low"], est["high"]
        if metric in _LOWER_BETTER:
            change = (value - ref) / ref if ref else 0.0
            bad = change > pol.max_error_increase
            msg = f"estimated {metric} rose {change:.0%} ({ref:.4g} -> {value:.4g}) - no labels yet"
        else:
            change = value - ref
            bad = -change > pol.max_metric_drop
            msg = f"estimated {metric} fell {-change:.3f} ({ref:.4f} -> {value:.4f}) - no labels yet"
        if bad:
            alerts.append(Alert("warning", "estimated_performance", msg, round(change, 6),
                                pol.max_error_increase if metric in _LOWER_BETTER else pol.max_metric_drop))
        return {"metric": metric, "reference": ref, "current": value, "change": round(change, 6),
                "interval": [low, high], "estimated": True, "details": est}

    def history(self) -> pl.DataFrame:
        """Past checks from ``log_path``: time, rows, drift share, metric, alert count."""
        from ..registry._store import read_jsonl
        rows = read_jsonl(self.log_path) if self.log_path else []
        out = [{"time": r["time"], "rows": r["rows"], "drift_share": r["drift_share"],
                "prediction_psi": r.get("prediction_psi"), "metric": r.get("metric"),
                "metric_value": r.get("metric_value"), "alerts": len(r["alerts"]),
                "retrain_recommended": r["retrain_recommended"]} for r in rows]
        schema = {"time": pl.String, "rows": pl.Int64, "drift_share": pl.Float64, "prediction_psi": pl.Float64,
                  "metric": pl.String, "metric_value": pl.Float64, "alerts": pl.Int64,
                  "retrain_recommended": pl.Boolean}
        return pl.DataFrame(out, schema=schema) if out else pl.DataFrame(schema=schema)

    # ── persistence (baselines only, never raw data) ────────────────────────
    def save(self, path) -> Path:
        from ..registry._store import write_json
        if self.baseline_ is None:
            raise RuntimeError("Nothing to save: call fit(reference) first.")
        path = Path(path)
        write_json(path, {"format": "dataspoof.monitor", "version": 1, "name": self.name,
                          "policy": dataclasses.asdict(self.policy), "columns": self.columns,
                          "reference_metrics": self.reference_metrics, "reference_rows": self.reference_rows_,
                          "baseline": self.baseline_.to_dict(),
                          "prediction_baseline": dataclasses.asdict(self.prediction_baseline_)
                          if self.prediction_baseline_ else None,
                          "estimator": self.estimator_.to_dict() if self.estimator_ is not None else None})
        return path

    @classmethod
    def load(cls, path, model, *, log_path=None) -> "Monitor":
        from .drift import ColumnBaseline
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format") != "dataspoof.monitor":
            raise ValueError(f"{path} is not a saved Monitor")
        mon = cls(model, policy=RetrainPolicy(**d["policy"]), log_path=log_path, columns=d["columns"],
                  reference_metrics=d["reference_metrics"], name=d.get("name"))
        mon.baseline_ = Baseline.from_dict(d["baseline"])
        mon.prediction_baseline_ = ColumnBaseline(**d["prediction_baseline"]) if d["prediction_baseline"] else None
        mon.reference_rows_ = d["reference_rows"]
        if d.get("estimator"):
            from .estimate import PerformanceEstimator
            mon.estimator_ = PerformanceEstimator.from_dict(mon.model, d["estimator"])
        return mon


def read_prediction_log(path, since: float | None = None) -> pl.DataFrame:
    """Inputs logged by the REST API (``prediction_log=``) as a frame, ready for ``Monitor.check``."""
    from ..registry._store import read_jsonl
    rows = [r["input"] for r in read_jsonl(path) if since is None or r.get("time", 0) >= since]
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


# ── retraining ──────────────────────────────────────────────────────────────

@dataclasses.dataclass
class RetrainResult:
    challenger: object
    metric: str
    champion_score: float | None
    challenger_score: float
    improved: bool
    version: Any = None
    promoted_to: str | None = None

    def summary(self) -> str:
        champ = "n/a" if self.champion_score is None else f"{self.champion_score:.4f}"
        line = (f"Retrained: holdout {self.metric} {champ} (current) -> {self.challenger_score:.4f} (new): "
                f"{'better' if self.improved else 'not better'}")
        if self.version is not None:
            line += f"; registered v{self.version.version}" + (f", promoted to {self.promoted_to}"
                                                                if self.promoted_to else "")
        return line

    def __repr__(self):
        return self.summary()


def retrain(model, train_df, holdout_df, *, registry=None, name: str | None = None, promote_to: str | None = "staging",
            min_improvement: float = 0.0, **automl_kwargs) -> RetrainResult:
    """Train a challenger on fresh data and compare it with the current model on ``holdout_df``.

    Settings (task, metric, leakage handling) are copied from the current model. With
    ``registry`` and ``name`` the challenger is registered, and promoted to ``promote_to``
    only if it beats the current model by at least ``min_improvement`` on the holdout."""
    from ..automl import AutoML
    from ..automl.automl import _metric_label
    from ..deploy.scoring import resolve_model
    from ..evaluation import evaluate_model
    champion = resolve_model(model)
    if not isinstance(champion, AutoML):
        raise TypeError("retrain() works with supervised AutoML models")
    settings = {"task": champion.task_.task, "metric": champion.metric_, "verbose": False,
                "time_col": champion.task_.time_col if champion.task_.time_series else None, **automl_kwargs}
    challenger = AutoML(champion.target, **settings).fit(train_df)
    new = evaluate_model(challenger, holdout_df, bootstrap=0).metrics
    try:
        old = evaluate_model(champion, holdout_df, bootstrap=0).metrics
    except Exception:                      # e.g. the current model can't score the new schema
        old = {}
    label = _metric_label(champion.metric_)
    metric = label if new.get(label) is not None else _headline(new, champion.task_.task)
    lower = metric in _LOWER_BETTER
    cur, prev = new[metric], old.get(metric)
    improved = prev is None or ((prev - cur) if lower else (cur - prev)) > min_improvement
    result = RetrainResult(challenger, metric, prev, cur, improved)
    if registry is not None and name:
        v = registry.register(challenger, name, dataset=train_df,
                              metrics={f"holdout_{k}": val for k, val in new.items() if val is not None},
                              description=f"automatic retrain: holdout {metric} {cur:.4f} vs {prev}")
        result.version = v
        if improved and promote_to:
            registry.promote(name, v.version, promote_to, reason=f"retrain improved {metric}: {prev} -> {cur:.4f}")
            result.promoted_to = promote_to
    return result


def retrain_if_needed(monitor: Monitor, batch, train_df, holdout_df, **kwargs):
    """Check ``batch``; retrain only if the policy recommends it. Returns ``(report, result | None)``."""
    report = monitor.check(batch)
    if not report.retrain_recommended:
        return report, None
    return report, retrain(monitor.model, train_df, holdout_df, **kwargs)
