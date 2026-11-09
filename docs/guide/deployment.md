# Deployment


```python
ac.score(model, df, keep=["id"])                                  # id + prediction + proba_<class>
ac.batch_score(model, "big.csv", "scored.parquet", chunk_size=100_000)   # streams files of any size
ac.serve(model, port=8000)                                        # REST API (development server)
ac.export_model(model, "out/", format="bundle")                   # or "pickle" / "onnx"
ac.build_docker(model, "build/", name="titanic", version="3")     # ready-to-build Docker context
```

**The REST API** is a standard WSGI app with no web framework needed. Its endpoints:
- `GET /health`;
- `GET /metadata`;
- `POST /predict` with `{"records": [{...}, ...]}`, which returns predictions and class probabilities.

Set `AC_API_KEY` to require an `X-API-Key` header, which is checked in constant time. Request
sizes are capped. Errors return JSON, and internal errors never leak details. With
`prediction_log="preds.jsonl"`, every request is logged for monitoring.

In production, run `gunicorn mlpilot.deploy.wsgi:app` with `AC_MODEL_PATH` set. Use
waitress on Windows, and install either with `pip install "mlpilot[serve]"`.

**Docker.** `build_docker` writes a multi-stage Dockerfile: a non-root user, a health check,
gunicorn, and the model baked in. The generated image has been built and tested end to end.

**ONNX** (`pip install "mlpilot[onnx]"`) exports the trained estimator: linear models,
random forests, extra trees and voting ensembles. Pass `sample=df` and the export is checked
against the original model. HistGradientBoosting can't be exported yet because of an upstream
skl2onnx limitation; the error says so.

**The CLI** (also available as `python -m mlpilot.deploy`):

```bash
mlpilot serve  --model models/titanic --port 8000
mlpilot score  --model models/titanic --input big.csv --output scored.parquet
mlpilot --registry ac_registry docker --model titanic:production --out build/
mlpilot --registry ac_registry monitor --model titanic --monitor monitor.json --input today.csv
```
---

### Production serving

```python
app = ac.make_app(model, api_key=..., canary=(new_model, 0.1), shadow=challenger,
                  prediction_log="preds.jsonl", tracing=True)
app = ac.deploy.app_from_registry(registry, "churn", canary_weight=0.1)    # production + staging canary
ac.build_docker(model, "build/", target="sagemaker")                       # or "vertex", "azureml", "generic"
```

- **Rollouts:**
  - Canary rollouts send a share of traffic to a candidate model. Each request ID always goes
    to the same model, so retries stay consistent.
  - A shadow model scores every request in the background and is only logged, never returned.
- **Observability:** Prometheus metrics at `/metrics`, and optional OpenTelemetry tracing.
- **Cloud routes:** SageMaker (`/ping`, `/invocations`), Vertex AI (`instances` payloads,
  `$AIP_*` routes) and Azure ML (`/score`). There's also an ASGI app for uvicorn
  (`mlpilot.deploy.asgi:app`).
- **Speed:** a single-row prediction takes about 50 ms, down from about 500 ms in 1.x.
- **Signed models** (`mlpilot.security`):
  - `sign_model(path, private_key)` signs a saved model with Ed25519.
  - Set `AC_MODEL_PUBLIC_KEY` and every load (the API, batch scoring, the registry) refuses
    models that are unsigned or tampered with. The check runs before anything is unpickled.
  - `export_model(..., format="skops")` stores the model without pickle.
