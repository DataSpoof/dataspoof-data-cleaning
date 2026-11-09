# Monitoring


```python
mon = ac.Monitor(model, reference=holdout, log_path="monitor.jsonl")
mon.on_alert(ac.webhook_alert("https://hooks.slack.com/..."))    # or any callback(report, alerts)
report = mon.check(todays_batch)       # include the target column when labels are available
print(report)                          # alerts + "RETRAIN RECOMMENDED" when the policy says so
mon.save("monitor.json")               # binned baselines only: no raw data leaves the training side
mon.history()                          # every past check
```

Each check covers:
- **data drift** per input column: PSI, Jensen–Shannon distance, a chi-square test, missing-rate
  jumps and never-seen categories;
- **prediction drift**;
- **accuracy decay**, when the batch has labels: an AUC/accuracy drop, or an RMSE/MAE rise;
- **schema breaks** and **batches too small** to judge.

Thresholds are set with `RetrainPolicy(...)`. Data drift alone is also available as
`ac.drift_report(reference, current)`.

**Retraining:** `ac.retrain(model, new_train, holdout, registry=reg, name="titanic")` trains a
challenger with the champion's settings and compares the two on the holdout. It registers the
challenger and promotes it to staging only if it is better.
`monitoring.retrain_if_needed(mon, batch, ...)` runs the check first and retrains only when
needed.
---

### Incidents and delayed labels

```python
from mlpilot.monitoring import pagerduty_alert, opsgenie_alert, join_labels, sql_sink
mon.on_alert(pagerduty_alert(routing_key))       # one incident per model and alert kind (deduplicated)
mon.on_check(sql_sink(ac.SQLSource(url)))        # every check result -> a warehouse table for BI
joined = join_labels("preds.jsonl", outcomes, on="order_id")   # labels that arrived days later
mon.check_labels(joined)                         # live accuracy of what was actually served
```
