"""Monitoring (module 9): data & prediction drift, accuracy decay, alerts, retraining triggers.

* ``drift_report(reference, current)``      PSI / JS distance / chi-square per column.
* ``Monitor(model, reference)``             ``check(batch)`` -> alerts + retrain recommendation;
                                            saves baselines only (no raw data).
* ``retrain(model, train, holdout, registry=..., name=...)``  champion vs challenger, with
                                            registration and gated promotion.
* ``webhook_alert(url)``                    send alerts to Slack/Teams/any webhook.
"""

from .drift import Baseline, DriftReport, drift_report, psi
from .monitor import (Alert, Monitor, MonitorReport, RetrainPolicy, RetrainResult, read_prediction_log, retrain,
                      retrain_if_needed, webhook_alert)

__all__ = ["Baseline", "DriftReport", "drift_report", "psi", "Alert", "Monitor", "MonitorReport", "RetrainPolicy",
           "RetrainResult", "read_prediction_log", "retrain", "retrain_if_needed", "webhook_alert"]
