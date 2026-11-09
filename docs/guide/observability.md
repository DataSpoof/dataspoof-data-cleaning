# Observability


```python
app = ac.deploy.app_from_registry(registry, "churn", watch=30,          # promotions go live, no restart
                                  monitor=ac.Monitor.load("monitor.json", model))
# POST /predict  -> {"predictions": [...], "prediction_ids": ["9f2c...-0", ...]}
# POST /feedback -> {"prediction_id": "9f2c...-0", "label": 1, "rating": 4, "comment": "..."}
# GET  /metrics  -> live accuracy, drift per feature, estimated performance, reloads, latency...
```

```bash
cd deploy/monitoring && MODEL_TARGET=model:8000 GRAFANA_ADMIN_PASSWORD=... docker compose up -d
```

**What the server now does:**
- **Feedback:** users send outcomes or ratings against a prediction ID. They are logged and
  counted as live accuracy, and `monitoring.join_feedback` feeds them into `Monitor.check_labels`.
- **Hot reload:** the server follows registry promotions, or a replaced model folder, or
  `POST /admin/reload`. The swap is atomic, and a failed reload keeps the previous model serving.
- **Before labels arrive:** confidence-based performance estimation (and direct loss estimation
  for regression) warns when a model is getting worse. `domain_classifier_drift` catches
  multivariate drift. In testing, estimated accuracy landed within 3 points of the true accuracy.

**Dashboards and alerts:**
- The Grafana dashboard covers traffic, latency, errors, rollouts, live and estimated quality,
  drift, and reloads.
- 14 Prometheus alert rules come with their own `promtool` unit tests.
- In a Docker test, a real Prometheus scraped a live server, and the drift alert fired on the
  drifted column. Every dashboard query ran in Grafana.

**Evidently** (`integrations.evidently`): `evidently_report(reference, current, model=model)`. In
testing, its drifted columns agreed with ours.
