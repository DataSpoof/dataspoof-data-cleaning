# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/)
(see the [versioning policy](https://dataspoof.github.io/dataspoof-mlpilot/policies/)).

## [2.2.0] — 2026-11-12

### Added: images taken in hard conditions
- **Weather and lighting** (`media.conditions`):
  - detects fog, rain, snow, low light, glare, motion blur, sensor noise and JPEG damage
    (each scored 0-1 and tagged at 0.5);
  - `condition_report` shows which conditions a dataset covers and flags the rare or missing ones;
  - `apply_condition` simulates all of them, plus shadow and cutout, at severity 1-5;
  - `enhance` restores images: dehaze, low-light brightening, denoise, sharpen.
- **Faces** (`media.faces`, `[vision]` extra):
  - detection;
  - how much of a face is covered, and by what (hair, mask, sunglasses or object), in a per-image
    `occlusion_report`;
  - simulated coverings for training;
  - blurring or pixelating for privacy (`anonymize`, `blur_regions`).
- **Licence plates** (`media.plates`, `[ocr]` extra):
  - detection and OCR, with plate patterns (`IN`, `IN_BH`, `UK`, `EU`, `US`) that fix OCR
    look-alikes;
  - optional restoration before reading;
  - `plate_report`, which pairs each read with the conditions in the image.
- **Robustness benchmark** (`media.benchmark`):
  - `condition_benchmark` re-scores any image model per condition and severity, with face
    coverings included;
  - `condition_augment` builds training sets from the weakest conditions;
  - ready-made predictors for AutoML image classifiers, plate reading and face counting.
- New extras: `vision` (OpenCV 4: OpenCV 5 no longer ships the detector models) and `ocr`
  (RapidOCR; its models are bundled, so nothing is downloaded at run time).
- Image functions also accept numpy arrays, alongside PIL images and paths.

### Added: quantization
- **Models** (`mlpilot.deploy.quantize`, `[onnx]` extra):
  - `quantize_model` writes int8 and fp16 ONNX variants of a model and measures each on held-out
    data (file size, speed, metric change and agreement with the original), then recommends
    one;
  - `export_model(..., format="onnx", quantize="int8" | "uint8" | "fp16")`, `quantize_onnx`
    for any ONNX model, and a `mlpilot quantize` CLI command;
  - models with nothing to quantize (tree ensembles, linear models, k-NN) are reported as
    such rather than exported unchanged.
- **AutoML:** a new opt-in `mlp` model family (scikit-learn neural network), used only when
  named in `include=`. It is the AutoML model family that quantization shrinks.
- **CI/CD:** a `quantization` gate in `gates.yml` (`mode`, `max_drop`, `min_agreement`,
  `required`).
- **Embeddings:**
  - `TextEmbeddings(dtype="float32" | "float16" | "int8")`;
  - `quantize_vectors` / `dequantize_vectors` with a per-dimension int8 scale learned on
    training data.
- **Tables:**
  - `optimize_dtypes` and the `OptimizeDtypes` pipeline step shrink column types without
    changing values (integers by range, floats only when they survive the round trip, text
    to Categorical);
  - the step learns the types on training data and never overflows on new data.

### Fixed
- `save_cleaned_data(quantize=True)` no longer fails on integers above 2.1 billion (it cast
  every `Int64` column to `Int32`).
- `ClipOutliers` no longer fails on small integer columns (`Int8`, `Int16`, `UInt8`, …)
  whose learned fences lie outside the type's range.

## [2.1.0] — 2026-11-09

### Renamed: AutomatedCleaning is now MLPilot
- **Names:**
  - the distribution is `dataspoof-mlpilot` (PyPI refused `mlpilot` as too similar to an existing project);
  - the import is `import mlpilot`, and the CLI is `mlpilot`;
  - `MLPilotWarning` replaces `AutomatedCleaningWarning`, which remains as an alias.
- **Compatibility:**
  - `import automatedcleaning` and pickles that refer to `automatedcleaning.*` classes resolve to
    the same `mlpilot` modules, so models saved by AutomatedCleaning keep loading with the same
    classes;
  - the `AutomatedCleaning` project on PyPI gets a final 2.0.1 that installs `dataspoof-mlpilot`.
- **Unchanged:** the `AC_*` environment variables, the `ac_*` Prometheus metrics and the saved-file
  formats (`dataspoof.*`).
- **Renamed with the library:**
  - the Helm chart (`mlpilot-hub`), the OPA policy package (`data.mlpilot`) and the Prometheus job;
  - the Grafana dashboard file;
  - the MLflow tag prefix and the default OpenLineage namespace.

## [2.0.0] — 2026-11-09

### Breaking
- **Lean core install.** `pip install AutomatedCleaning` now installs only the core
  dependencies. The heavy ones moved to extras:
  - `[nlp]`: nltk
  - `[dashboard]`: plotly, matplotlib, missingno, pyfiglet
  - `[pii]`: presidio
  - `[llm]`: anthropic, langchain-anthropic

  To get what 1.x installed, use `pip install "AutomatedCleaning[all]"`. A feature whose extra
  is missing raises an `ImportError` naming the extra to install.
- `setup.py` was removed. The build is defined only in `pyproject.toml` (maturin).

### Changed
- **No side effects at import.**
  - Importing the library no longer downloads NLTK data, changes the application's warning
    filters, or reconfigures logging.
  - NLTK corpora are downloaded on first use, and never when `AC_OFFLINE` is set. Text cleaning
    falls back to a built-in stopword list when NLTK isn't installed.
- **Typed package.** It ships a `py.typed` marker and stubs for the Rust extension, and passes
  `mypy`.
- **CI** now runs:
  - tests across Linux, macOS and Windows × Python 3.11–3.13 × Polars 1/2;
  - a core-only install test;
  - mypy;
  - pip-audit and cargo-audit.

  Releases publish with PyPI Trusted Publishing, signed build-provenance attestations and a
  CycloneDX SBOM.
- Interactive AI corrections use `ANTHROPIC_API_KEY` when it is set, and the current Claude model.

### Added
- **Hub, the multi-user platform server** (`automatedcleaning.hub`, `[server]` extra):
  - workspaces, with SSO through OIDC and identity-provider group mapping;
  - API tokens and service accounts;
  - four roles (viewer, data scientist, approver, admin);
  - runs and a model registry with immutable, checksummed artifacts on disk, S3, GCS or Azure;
  - two-person production approvals with policy gates;
  - monitoring results;
  - a hash-chained, verifiable and exportable audit log;
  - a web UI, the Python client `HubClient`, and `automatedcleaning hub serve | create-admin`.
- **Hub deployment:** a Docker image with compose (plus PostgreSQL), a Helm chart, and a
  Terraform module for AWS (S3, RDS, Secrets Manager, IAM role, Helm release).
- **Forecasting** (`Forecaster`):
  - multiple series and multi-step horizons;
  - rolling-origin backtests against seasonal-naive;
  - conformal intervals and hierarchical reconciliation.
- **Causal inference** (`estimate_effect`): doubly-robust AIPW with overlap and balance
  diagnostics. **Uplift:** `UpliftModel` and `qini`.
- **Anomaly detection** (`AutoAnomaly`): an ensemble of three detectors with plain-language
  explanations.
- **AutoML:**
  - conformal `predict_interval` and `predict_set`;
  - `search="optuna"`, `imbalance="auto"`, `n_jobs`;
  - `predict_with_proba` and `set_inference_threads`.
- **Serving:**
  - Prometheus `/metrics` and OpenTelemetry tracing;
  - canary and shadow rollouts, and `app_from_registry`;
  - SageMaker, Vertex AI and Azure ML routes and Docker targets;
  - an ASGI app and the `container` entry point;
  - single-row latency down from about 500 ms to about 50 ms.
- **Security:**
  - Ed25519 model signing, verified before unpickling whenever `AC_MODEL_PUBLIC_KEY` is set;
  - `skops` export;
  - signing on registration with `register(..., sign_with=...)`.
- **Monitoring:**
  - PagerDuty and Opsgenie alert handlers;
  - `join_labels` and `Monitor.check_labels` for labels that arrive later;
  - `sql_sink` and `Monitor.on_check`;
  - `SQLSource.insert` and `execute`.
- **Scale:**
  - `transform_file` for files of any size;
  - Spark, Ray and Dask scoring;
  - `to_sql` SQL pushdown of fitted cleaning pipelines.
- **Integrations:** Airflow operators, Prefect tasks and flow, Dagster assets, Kafka stream
  scoring, and plain `integrations.tasks`.
- **MLflow** (`integrations.mlflow`, `[mlflow]` extra):
  - batched, idempotent export of runs and whole trackers;
  - AutoML models logged as pyfunc models with a signature, so they serve with `mlflow models serve`;
  - `export_registry` (stages become aliases) and `load_automl`.
- **DVC** (`integrations.dvc`, `[dvc]` extra): `DVCRepo` (init, remotes, track, push and pull,
  read any revision, checkout, `log_to_run`) and `export_repository` for `DataRepository`
  history.
- **Feast** (`integrations.feast`, `[feast]` extra): `export_to_feast` writes, applies and
  materializes a Feast repository.
- **Data splitting** (`split_data`, `cv_folds`, `check_split`): stratified, grouped and time-ordered
  splits with gaps; every split is leak-checked (shared groups, time overlap, duplicate rows, unseen classes).
- **Model security:** `robustness_report` (noise stability, a minimal adversarial attack,
  feature sensitivity, extreme inputs) and `poisoning_report` (confident-learning label errors,
  conflicting duplicates, backdoor triggers, outliers).
- **Environments** (`Environments`, `environments.yml`): isolated dev / QA / prod registries;
  `release` enforces order, source stage, metric gates, four-eyes approvals and signatures, and
  records provenance on both sides. The registry and Hub gain a `qa` stage.
- **Model CI/CD:** the CLI commands `train`, `gate`, `register`, `promote`, `release`, `latest`,
  `deploy-sagemaker` and `cicd init` (GitHub Actions / GitLab CI pipelines, checked with actionlint).
- **Secrets** (`secretstore`): `vault://` (token, AppRole, Kubernetes; KV v1 and v2), `aws-sm://`,
  `env://` and `file://` references, resolved wherever credentials are read.
- **Open Policy Agent for the Hub** (`hub.policy`): deny-overrides access rules, promotion rules
  and fail-closed behaviour, plus Rego policies with unit tests in `deploy/opa`.
- **Serving:**
  - `POST /feedback` with prediction IDs, live-accuracy metrics and `join_feedback`;
  - hot reload (`watch=`, `POST /admin/reload`, registry-following);
  - live drift, estimated-performance and alert gauges (`monitor=`).
- **Monitoring before labels:** `PerformanceEstimator` (CBPE, direct loss estimation),
  `domain_classifier_drift`, and `estimated_performance` alerts in `Monitor.check`.
- **Grafana + Prometheus** (`deploy/monitoring`): a dashboard, 14 alert rules with `promtool`
  unit tests, and a compose stack.
- **Lineage** (`lineage`): OpenLineage run events with schema, quality and version facets; HTTP
  and file transports; `emit_tracked_run`, `emit_dataset_version`, `LineageGraph`.
- **Catalogs and orchestration:** AWS Glue (catalog tables, partitions, reads, job runs) and Azure
  Data Factory pipeline runs.
- **AWS deployment** (`deploy.aws`): ECR push and blue/green SageMaker endpoints (serverless or
  instances, autoscaling, data capture, rollback alarms); Terraform for SageMaker and ECS Fargate.
- **Evidently** (`integrations.evidently`): reports plus agreement with the built-in drift report.
- **Tests against real infrastructure:** Prometheus, Grafana, Vault, OPA and Marquez in Docker
  (`AC_DOCKER_TESTS=1`); moto for AWS; dedicated CI jobs.
- These three are tested against the real tools (MLflow server and model server, git + DVC
  with a remote, Feast), including in a dedicated CI job.
- **Image and audio data** (`media`):
  - quality audits (corrupt files, exact and near duplicates, blur, exposure, clipping, silence,
    mixed formats);
  - preprocessors (EXIF-aware image resize and normalise; audio resample, trim and level);
  - augmentation;
  - features for AutoML (colour and edge-orientation histograms; MFCC and spectral features).
- **Statistical testing** (`stats`):
  - automatic test selection with assumption checks, effect sizes and confidence intervals;
  - chi-square and Fisher's exact test, correlations;
  - A/B tests, power analysis, multiple-testing correction, `test_all`.
- **Data version control** (`DataRepository`): content-addressed commits, tags, row-level
  diffs, checksum-verified checkout, links to experiment runs, garbage collection.
- **Feature store** (`FeatureStore`, `Entity`, `FeatureView`): point-in-time-correct
  historical joins with TTL, an SQLite online store, `materialize`, schema guards.
- **Documentation site** (MkDocs Material with an API reference), plus `SECURITY.md`,
  `CONTRIBUTING.md` and a versioning policy.

### Removed
- `automatedcleaning/cleaning_update.py`: dead code that nothing imported.

## [1.14.0] — 2026-11-08
- **Ingestion & connectors:**
  - files, globs, HTTP, and S3 / GCS / Azure;
  - any SQL database (bound parameters);
  - Snowflake and BigQuery;
  - REST APIs with five pagination styles;
  - reusable schema inference and incremental loads.
- **Experiment tracking & model registry:** runs, dataset fingerprints, dev → staging →
  production promotion with metric gates, audit history and rollback.
- **Deployment:**
  - batch scoring and a REST API (API keys, prediction log);
  - bundle, pickle and ONNX export;
  - Docker build contexts and a CLI.
- **Monitoring:** data and prediction drift (PSI), accuracy decay, alerts and webhooks, and
  champion-vs-challenger retraining.
- **Insights & reporting:** significance-tested findings, executive reports, and optional
  narratives written by Claude.

## [1.13.0] — 2026-11-06
- **Feature engineering:**
  - dates, out-of-fold target encoding, interactions, aggregations, lags, text embeddings;
  - feature selection and leakage detection.
- **AutoML:** task detection, time-budgeted model search, an ensemble, and AutoCluster.
- **Evaluation & explainability:** metrics with confidence intervals, built-in SHAP, error
  analysis, fairness checks and model cards.

## [1.12.0] — 2026-11-06
- Profiling & data quality: automated EDA, the 0–100 quality score, YAML data contracts, schema
  diffs and quality gates.

## [1.11.0] — 2026-11-05
- Shared `Step` / `Pipeline` core and the stateful cleaning pipeline (JSON save/load, audit
  reports, schema checks).

## [1.10.0] and earlier — 2026-07
- 1.10.0: 8 more data-quality checks in the dashboard.
- 1.9.0: column pickers in the dashboard.
- 1.8.x: the dashboard data-quality panel.
- 1.7.0: the enterprise dashboard redesign.
- 1.6.x: 43 additional cleaning steps and a dependency cleanup.
- 1.5.x: the Rust compute backend.
