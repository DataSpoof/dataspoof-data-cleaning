# Deployment: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Deployment guide](../guide/deployment.md).

**On this page:**

- [`mlpilot.deploy`](#mlpilotdeploy)
- [`mlpilot.deploy.asgi`](#mlpilotdeployasgi)
- [`mlpilot.deploy.aws`](#mlpilotdeployaws): `push_image`, `deploy_sagemaker`, `invoke_sagemaker`
- [`mlpilot.deploy.cli`](#mlpilotdeploycli): `container_command`, `main`
- [`mlpilot.deploy.export`](#mlpilotdeployexport): `export_model`, `load_skops`, `load_pickle`, `onnx_inputs`, `build_docker`
- [`mlpilot.deploy.quantize`](#mlpilotdeployquantize): `quantizable_weights`, `quantize_onnx`, `QuantizedVariant`, `QuantizationReport`, `quantize_model`
- [`mlpilot.deploy.scoring`](#mlpilotdeployscoring): `resolve_model`, `expected_columns`, `model_inputs`, `score`, `batch_score`, `to_records`
- [`mlpilot.deploy.server`](#mlpilotdeployserver): `Metrics`, `make_app`, `app_from_registry`, `make_asgi_app`, `serve`
- [`mlpilot.deploy.wsgi`](#mlpilotdeploywsgi): `build_app`

---

## `mlpilot.deploy`

<sub>Source: [`mlpilot/deploy/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/deploy/__init__.py)</sub>

Deployment (module 8): scoring, a REST API, batch jobs, exports and Docker images.

* `score(model, df)` / `batch_score(model, "in.csv", "out.parquet")`  predictions,
  streaming files of any size in chunks.
* `make_app(model)` / `serve(model)`   a REST API (WSGI; gunicorn/waitress-ready), with
  `POST /feedback`, hot reload (`watch=`) and live drift gauges (`monitor=`).
* `deploy.aws`                          ECR push + blue/green SageMaker endpoints (needs boto3).
* `export_model(model, path, format="bundle" | "pickle" | "onnx", quantize=None | "int8" | "fp16")`.
* `quantize_model(model, test_df)`   int8 / fp16 ONNX variants, measured (size, speed, metric).
* `build_docker(model, "build/")`   a ready-to-build Docker context (non-root, health check).
* CLI: `python -m mlpilot.deploy serve|score|docker|export|monitor`.

---

## `mlpilot.deploy.asgi`

<sub>Source: [`mlpilot/deploy/asgi.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/deploy/asgi.py)</sub>

ASGI entry point: `uvicorn mlpilot.deploy.asgi:app` (same environment variables
as `wsgi`).

---

## `mlpilot.deploy.aws`

<sub>Source: [`mlpilot/deploy/aws.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/deploy/aws.py)</sub>

Deploy a model to AWS: push the container to ECR and run it as a SageMaker endpoint.

```python
from mlpilot.deploy import aws
ac.build_docker(model, "build/churn", target="sagemaker")          # then: docker build -t churn:4 build/churn
uri = aws.push_image("churn:4", "ml/churn")                         # ECR repo created if missing
aws.deploy_sagemaker(uri, "churn-prod", role_arn="arn:aws:iam::…:role/sagemaker-churn",
                     instance_type="ml.m5.large", instance_count=2, autoscale=(2, 8),
                     environment={"AC_API_KEY": "aws-sm://prod/churn#api_key"},
                     data_capture_s3="s3://ml-capture/churn/")
aws.invoke_sagemaker("churn-prod", [{"tenure": 3, "plan": "basic"}])
```

* **Blue/green updates**: every deploy creates a new model + endpoint configuration and updates the
  endpoint in place; SageMaker shifts traffic only once the new fleet is healthy, and rolls back if
  it isn't. Old configurations are pruned (`keep_configs`).
* **Serverless** (`serverless={"memory_mb": 2048, "max_concurrency": 10}`) or real-time instances,
  with target-tracking autoscaling on invocations per instance.
* **Data capture** of requests and responses to S3 for monitoring.
* Secrets: environment values may be `aws-sm://` references; the container resolves them at
  start-up (give the role `secretsmanager:GetSecretValue`).

The Terraform modules in `deploy/terraform/aws-sagemaker` and `deploy/terraform/aws-ecs` create
the same resources declaratively (plus IAM, alarms and, for ECS, a load balancer).

### `push_image()`

```python
def push_image(local_image: str, repository: str, *, tag: str | None=None, ecr=None, docker: str='docker') -> str
```

Tag and push a local image to ECR (creating the repository, scan-on-push, if needed).
Returns the pushed image URI.

### `deploy_sagemaker()`

```python
def deploy_sagemaker(image_uri: str, endpoint_name: str, *, role_arn: str, instance_type: str='ml.m5.large', instance_count: int=1, serverless: dict | None=None, autoscale: tuple | None=None, target_invocations: int=300, environment: dict | None=None, data_capture_s3: str | None=None, capture_percentage: int=100, model_name: str | None=None, model_version: str | None=None, tags: dict | None=None, wait: bool=True, timeout: float=1800, keep_configs: int=3, vpc: dict | None=None, rollback_alarms: list | None=None, bake_seconds: int=300, sagemaker=None, autoscaling=None) -> dict
```

Create or blue/green-update `endpoint_name` to serve `image_uri`. Returns the endpoint
description (`EndpointStatus` `InService` when `wait`). `rollback_alarms`: CloudWatch alarm
names that roll a blue/green update back if they fire during `bake_seconds`.

### `invoke_sagemaker()`

```python
def invoke_sagemaker(endpoint_name: str, records, *, runtime=None) -> dict
```

Score records through the endpoint (the same JSON contract as the REST API).

---

## `mlpilot.deploy.cli`

<sub>Source: [`mlpilot/deploy/cli.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/deploy/cli.py)</sub>

Command line: `mlpilot serve | score | docker | export | monitor | hub` and the
model CI/CD steps `train | gate | register | promote | release | latest | deploy-sagemaker | cicd init`.

Also available as `python -m mlpilot.deploy ...`.

### `container_command()`

```python
def container_command() -> list
```

The gunicorn command line for the current environment (Vertex sets AIP_HTTP_PORT).

### `main()`

```python
def main(argv=None)
```

---

## `mlpilot.deploy.export`

<sub>Source: [`mlpilot/deploy/export.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/deploy/export.py)</sub>

Exporting models: a portable bundle, a single pickle file, ONNX, or a Docker image context.

### `export_model()`

```python
def export_model(model, path, format: str='bundle', *, sample=None, quantize: str | None=None) -> Path
```

Export a fitted model.

* `"bundle"` (default): a directory with `manifest.json`, `model.joblib` and
  `preprocess.json` — the format `load_model`/`AutoML.load` read; checksum-verified.
* `"pickle"`: one `.joblib` file holding the whole model object, plus a `.sha256`
  file next to it. Convenient, but only load pickles you trust.
* `"skops"`: like `"bundle"` but the estimator is stored with skops, which loads only
  explicitly trusted types (no arbitrary code on load); read it with `load_skops`.
* `"onnx"`: the trained estimator as an ONNX graph (needs `pip install skl2onnx`),
  for ONNX Runtime / other languages. The graph takes the *model-ready* columns (after
  AutoML's cleaning & feature steps); `preprocess.json` is written next to it, and
  `io.json` lists the input names and types. Pass `sample` (raw rows) to check the
  ONNX predictions match (needs `onnxruntime`). `quantize="int8"` / `"uint8"` /
  `"fp16"` also writes `model.<mode>.onnx` with 8- or 16-bit weights and returns its path
  (neural-network models only; see `mlpilot.deploy.quantize`, and `quantize_model` to
  compare every mode on held-out data first).

### `load_skops()`

```python
def load_skops(path, *, trusted: list | None=None)
```

Load a `format="skops"` export. scikit-learn / numpy / scipy types are trusted; any other
type stored in the file must be listed in `trusted` (see `manifest.json["skops_types"]`),
otherwise loading stops with the list of types it refused.

### `load_pickle()`

```python
def load_pickle(path)
```

Load a `format="pickle"` export after checking its `.sha256` file.

### `onnx_inputs()`

```python
def onnx_inputs(model, df) -> dict
```

Feed dict for ONNX Runtime from raw rows (applies the model's preprocessing).

### `build_docker()`

```python
def build_docker(model, out_dir, *, name: str='model', version: str='1', port: int | None=None, workers: int=2, base_image: str='python:3.11-slim', library_spec: str | None=None, extra_requirements: list | None=None, target: str='generic') -> Path
```

Write a Docker build context that serves the model over REST (gunicorn + /predict).

`out_dir` gets `Dockerfile`, `requirements.txt`, `.dockerignore`, `README.md` and
the saved `model/`. Build with `docker build -t <name> <out_dir>`. The image runs as a
non-root user, has a health check, and reads `AC_API_KEY` to require an API key.
`target`: `"generic"` (port 8000), `"sagemaker"` (AWS SageMaker: port 8080,
`/ping` + `/invocations`), `"vertex"` (Google Vertex AI: `$AIP_HTTP_PORT` and its
routes) or `"azureml"` (Azure ML: `/score`). Every target serves every route.
`library_spec` pins how the library is installed: default this version from PyPI; or a
requirement string; or the path of a local wheel/sdist, which is copied into the context.

---

## `mlpilot.deploy.quantize`

<sub>Source: [`mlpilot/deploy/quantize.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/deploy/quantize.py)</sub>

Model quantization: smaller ONNX models with 8-bit or 16-bit weights, measured before you ship.

```python
rep = quantize_model(model, test_df, out="build/quantized")     # AutoML model + held-out rows
print(rep.summary())
```
fp32   155.3 KB  1.00x  roc_auc 0.9412           agreement 100.00%
int8    43.6 KB  3.56x  roc_auc 0.9405 (-0.0007)  agreement  99.70%   <- recommended
fp16    78.9 KB  1.97x  roc_auc 0.9412 (-0.0000)  agreement 100.00%
```python
rep.recommended.path                                           # build/quantized/model.int8.onnx
quantize_onnx("my_model.onnx", "my_model.int8.onnx")           # any ONNX model (vision, NLP, ...)
```

Modes:

* `int8` / `uint8`: dynamic quantization (ONNX Runtime). Weights of matrix-multiply and
  convolution layers are stored as 8-bit integers (about 4x smaller); activations are
  quantized on the fly. The biggest saving, with a small accuracy cost.
* `fp16`: float32 weight tensors are *stored* as float16 (2x smaller) and cast back to
  float32 when the model loads, so it runs everywhere at full float32 speed. Almost lossless.

What it helps: neural networks: AutoML's opt-in `mlp` model (`AutoML(..., include=["mlp"])`),
and your own ONNX models (CNNs, transformers, sentence embedders). Tree ensembles, linear models
and k-NN export to ONNX operators that keep their own float32 parameters, which ONNX Runtime
cannot quantize; for those the report says there is nothing to quantize instead of producing
a file that is no smaller.

### `quantizable_weights()`

```python
def quantizable_weights(model) -> dict
```

How many float32 weights an ONNX model (path or `ModelProto`) has, and how many sit in
layers that quantization can shrink. `{"weights", "int8_weights", "fp16_weights"}`.

### `quantize_onnx()`

```python
def quantize_onnx(src, dst, mode: str='int8', *, per_channel: bool=False) -> Path
```

Quantize any ONNX model file. `mode`: `"int8"`, `"uint8"` or `"fp16"` (see the
module docstring). Raises `ValueError` when the model has no weights the mode can shrink.

### class `QuantizedVariant`

```python
class QuantizedVariant:
```

| Field | Type | Default |
|---|---|---|
| `mode` | `str` | required |
| `path` | `Path \| None` | required |
| `size_bytes` | `int \| None` | required |
| `compression` | `float \| None` | required |
| `latency_ms` | `float \| None` | required |
| `score` | `float \| None` | required |
| `score_change` | `float \| None` | required |
| `agreement` | `float \| None` | required |
| `note` | `str` | `''` |

#### `QuantizedVariant.as_dict()`

```python
as_dict() -> dict
```

### class `QuantizationReport`

```python
class QuantizationReport:
```

| Field | Type | Default |
|---|---|---|
| `metric` | `str` | required |
| `higher_is_better` | `bool` | required |
| `weights` | `dict` | required |
| `variants` | `list` | required |
| `max_drop` | `float` | required |
| `min_agreement` | `float` | required |
| `recommended` | `QuantizedVariant \| None` | required |

#### `QuantizationReport.variant()`

```python
variant(mode: str) -> QuantizedVariant | None
```

#### `QuantizationReport.drop()`

```python
drop(v: QuantizedVariant) -> float | None
```

How much worse `v` scores than fp32 (positive = worse), whatever the metric direction.

#### `QuantizationReport.ok()`

```python
ok(v: QuantizedVariant) -> bool
```

#### `QuantizationReport.passed()`

```python
passed(mode: str | None=None) -> bool
```

`mode` (default: the recommended one) is within `max_drop` and `min_agreement`.

#### `QuantizationReport.to_frame()`

```python
to_frame() -> pl.DataFrame
```

#### `QuantizationReport.summary()`

```python
summary() -> str
```

#### `QuantizationReport.to_json()`

```python
to_json(path) -> Path
```

### `quantize_model()`

```python
def quantize_model(model, data, *, modes=('int8', 'fp16'), out=None, metric: str | None=None, max_drop: float=0.01, min_agreement: float=0.99) -> QuantizationReport
```

Export `model` (an AutoML model, a saved model directory or a registry model) to ONNX,
quantize it in each of `modes`, and measure every variant on held-out `data` (raw rows
including the target): file size, latency, the metric and agreement with the original model.

The `recommended` variant is the smallest one whose metric is at most `max_drop` worse
than the float32 export and whose predictions agree with the original on at least
`min_agreement` of rows. Files go to `out` (default: a temporary directory), together
with `quantization.json`.

---

## `mlpilot.deploy.scoring`

<sub>Source: [`mlpilot/deploy/scoring.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/deploy/scoring.py)</sub>

Scoring: predictions for a frame, and batch jobs that stream files of any size.

### `resolve_model()`

```python
def resolve_model(model)
```

A fitted model object, or a path to a saved model directory / registry spec.

### `expected_columns()`

```python
def expected_columns(model) -> list
```

Raw input columns the model was trained on (before its own preprocessing).

### `model_inputs()`

```python
def model_inputs(model, df: pl.DataFrame) -> pl.DataFrame
```

Only the columns the model was trained on (extra ids/metadata columns are normal when
scoring and shouldn't trigger schema warnings); missing ones are left to the model.

### `score()`

```python
def score(model, df, *, proba: bool=True, keep: list | None=None, prefix: str='') -> pl.DataFrame
```

Predictions for every row of `df`.

Returns the `keep` columns of `df` (default: none) followed by `prediction` and, for
classifiers, `proba_<class>` columns (clustering models return `cluster`).

### `batch_score()`

```python
def batch_score(model, source, output, *, chunk_size: int=100000, keep: list | str | None='all', proba: bool=True, log_every: int | None=None) -> dict
```

Score a large file chunk by chunk and write the results (`.parquet` or `.csv`).

`keep`: `"all"` input columns (default), a list of id columns, or `None`.
Memory use is bounded by `chunk_size`. Returns a summary (rows, seconds, rows/second,
prediction distribution, columns the model expected but the file lacked).

### `to_records()`

```python
def to_records(df: pl.DataFrame) -> list
```

JSON-safe list of row dicts (NaN → None).

---

## `mlpilot.deploy.server`

<sub>Source: [`mlpilot/deploy/server.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/deploy/server.py)</sub>

A REST API for a model, as a standard WSGI app (no web-framework dependency).

Run it with `serve(model)` for development, or any WSGI server in production — e.g.
`gunicorn mlpilot.deploy.wsgi:app` (Linux) / `waitress-serve` (Windows), which
is what the generated Docker image does. `make_asgi_app` wraps it for uvicorn/hypercorn.

Endpoints:
```
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
```

Responses: `{"predictions": [...], "probabilities": [{class: p}], "prediction_ids": [...],
"variant": "primary", ...}`. Send outcomes back with `POST /feedback` and the prediction id:
they are logged (`feedback_log`), matched to the prediction, and counted as live accuracy.

Hot reload: `app_from_registry(..., watch=30)` follows promotions in the registry, and
`make_app(path, watch=30)` follows a model directory being replaced - no restart.

Rollouts: `canary=(model, 0.1)` sends ~10% of requests (chosen by request id, so retries are
stable) to a candidate model; `shadow=model` scores every request with a second model in the
background and logs its output (and agreement) without affecting responses.

Security: set `api_key` (or the `AC_API_KEY` env var) to require an `X-API-Key` header
(compared in constant time). Bodies over `max_body_bytes` and requests over `max_rows` are
rejected. Internal errors return a generic message; details go to the server log.

### class `Metrics`

```python
class Metrics:
```

Thread-safe Prometheus counters, gauges and a latency histogram (text exposition format).

**Constructor**

```python
Metrics(labels: dict)
```

#### `Metrics.set_labels()`

```python
set_labels(**labels)
```

#### `Metrics.observe()`

```python
observe(route: str, status: int, variant: str | None, seconds: float, rows: int=0)
```

#### `Metrics.shadow_result()`

```python
shadow_result(agree: int, disagree: int)
```

#### `Metrics.feedback_result()`

```python
feedback_result(variant: str, result: str, *, correct: bool | None=None, abs_error: float | None=None, window: int=1000)
```

#### `Metrics.reload_result()`

```python
reload_result(ok: bool)
```

#### `Metrics.set_gauges()`

```python
set_gauges(name: str, help_text: str, values: dict)
```

Replace every series of gauge `name`: `values` maps a label dict (as a tuple of
pairs) to a number.

#### `Metrics.render()`

```python
render() -> str
```

### `make_app()`

```python
def make_app(model, *, api_key: str | None=None, name: str | None=None, version=None, max_rows: int=10000, max_body_bytes: int=10 * 1024 * 1024, prediction_log=None, canary=None, shadow=None, tracing: bool=False, public_metrics: bool=True, feedback_log=None, monitor=None, drift_window: int=2000, drift_every: int=500, reloader: Callable | None=None, watch: float | None=None, recent_predictions: int=100000)
```

Build the WSGI application for `model` (an object or a saved-model path).

* `prediction_log`: JSONL file to append every request's inputs and outputs to (feed it to
  `mlpilot.monitoring` to watch for drift).
* `canary=(model, weight)`: route that share of requests to a candidate model.
* `shadow=model`: also score every request with this model, in the background; only logged.
* `tracing=True`: OpenTelemetry spans per request (needs `opentelemetry-api` + an SDK).
* `public_metrics=False`: require the API key for `/metrics` too.
* `feedback_log`: JSONL file for `POST /feedback` (default: next to `prediction_log`).
* `monitor`: a fitted `Monitor` (or a saved one's path);
  the last `drift_window` inputs are checked every `drift_every` rows and the results
  (drift share, per-feature PSI, prediction PSI, estimated performance, active alerts) are
  exported as Prometheus gauges.
* `watch=seconds`: hot reload. With a model *path*, a replaced model at that path is picked
  up; with `reloader` (see `app_from_registry`), whatever it returns. `POST
  /admin/reload` (needs the API key) reloads immediately. In-flight requests finish on the
  old model; the swap is atomic.

### `app_from_registry()`

```python
def app_from_registry(registry, name: str, *, canary_stage: str | None='staging', canary_weight: float=0.0, shadow_stage: str | None=None, watch: float | None=None, **kw)
```

Serve the production version of a registered model, optionally with the newest
`canary_stage` version as a canary (`canary_weight` share) or as a shadow.

`watch=seconds` polls the registry and hot-swaps to a newly promoted production version
(and refreshes the canary / shadow), without a restart.

### `make_asgi_app()`

```python
def make_asgi_app(model=None, *, wsgi_app=None, **kw)
```

The same API as an ASGI app (for uvicorn / hypercorn). Requests run in a thread pool.

### `serve()`

```python
def serve(model, host: str='127.0.0.1', port: int=8000, **kw)
```

Serve the model with a threaded development server (use gunicorn/waitress in production).

Binds to localhost by default; pass `host="0.0.0.0"` to accept outside connections.

---

## `mlpilot.deploy.wsgi`

<sub>Source: [`mlpilot/deploy/wsgi.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/deploy/wsgi.py)</sub>

WSGI entry point for production servers: `gunicorn mlpilot.deploy.wsgi:app`.

Configured by environment variables:

* `AC_MODEL_PATH`        saved model directory (required)
* `AC_API_KEY`           if set, requests need a matching `X-API-Key` header
* `AC_MODEL_NAME` / `AC_MODEL_VERSION`   shown in /metadata, responses and metrics
* `AC_MAX_ROWS`          records per request (default 10000)
* `AC_PREDICTION_LOG`    JSONL file to log inputs and predictions to (for monitoring)
* `AC_CANARY_MODEL_PATH` + `AC_CANARY_WEIGHT`   canary rollout (e.g. 0.1 = 10% of requests)
* `AC_SHADOW_MODEL_PATH` shadow model: scored in the background, only logged
* `AC_TRACING=1`         OpenTelemetry spans (needs opentelemetry-api + an SDK/exporter)
* `AC_PUBLIC_METRICS=0`  require the API key for /metrics
* `AC_FEEDBACK_LOG`      JSONL file for POST /feedback (default: next to the prediction log)
* `AC_MONITOR_PATH`      a saved Monitor: live drift + estimated performance as Prometheus gauges
* `AC_WATCH_SECONDS`     hot reload: re-check the model (or the registry) every N seconds
* `AC_REGISTRY` + `AC_MODEL_NAME`   serve the registry's production model instead of
  `AC_MODEL_PATH` (with AC_WATCH_SECONDS, promotions go live by themselves)

### `build_app()`

```python
def build_app()
```
