# MLPilot

!!! note "AutomatedCleaning is now MLPilot"
    Install `mlpilot` and `import mlpilot`. Existing code and saved models keep working: see [Moving from AutomatedCleaning](installation.md#moving-from-automatedcleaning).


**From raw data to a monitored production model: automated, explainable, auditable.**

MLPilot automates the whole data-science lifecycle in one Python library. It covers
ingestion and cleaning, data quality, feature engineering, AutoML, explainability and fairness,
the model registry, deployment, monitoring, and plain-English insights. A **Rust engine**
handles the heavy computation, and every stage shares one `fit` / `transform` / `save` contract.

## Quick start


```bash
pip install mlpilot
```

```python
import mlpilot as ac

df = ac.read("titanic.csv")                                       # 1. ingest
train, test = df.head(700), df.tail(191)
print(ac.check_quality(train).grade)                              # 2. data-quality score
model = ac.AutoML("Survived", time_budget=60).fit(train)          # 3. clean + features + AutoML
print(model.report(test))                                         # 4. held-out metrics
model.model_card(test, sensitive=["Sex"]).save("model_card.html") # 5. model card + fairness
registry = ac.ModelRegistry("registry")
version = registry.register(model, "titanic")                     # 6. version it...
registry.promote("titanic", version.version, "production")        #    ...and promote it
predictions = ac.score(registry.load("titanic"), test)            # 7. score (or ac.serve / build_docker)
monitor = ac.Monitor(model, reference=test)                       # 8. watch for drift & decay
print(ac.insights_report(df, "Survived").narrative)               # 9. plain-English insights
```

What that gives you:
- **Held-out metrics** with 95% confidence intervals.
- **A model card** (HTML) with SHAP drivers, error analysis and a fairness review.
- **A versioned production model** whose checksum is verified when it loads.
- **A drift monitor** for new data.
- **Plain-English findings**, such as *"The rate of Survived = 1 is 19% when Sex is male, vs 74%
  otherwise"*.

Prefer to clean interactively? `ac.clean_data(ac.load_data("data.csv"))` walks you through it
and builds an EDA dashboard.

![MLPilot workflow](assets/workflow.svg)

See [Workflow](workflow.md) for how to read it.

## The platform at a glance


| | Stage | Module | What it automates |
|:-:|---|---|---|
| 📥 | **Ingestion & connectors** | `ingestion` | Files, globs, URLs, S3 / GCS / Azure, any SQL database, Snowflake, BigQuery and REST APIs. Includes schema inference and incremental loads. |
| 🧹 | **Data cleaning** | `cleaners`, `steps` | 50+ cleaning steps: missing values, outliers, text, dates, PII, duplicates, encodings. Rules are learned on training data and replayed exactly. |
| 📊 | **Profiling & quality** | `quality` | Automated EDA, a 0–100 quality score, YAML data contracts, schema-drift checks and quality gates. |
| 🧬 | **Feature engineering** | `features` | Dates, out-of-fold target encoding, interactions, aggregations, lags, text embeddings, selection and **leakage detection**. |
| 🤖 | **AutoML** | `automl` | Detects the task, runs a time-budgeted model search, builds an ensemble, and also does clustering. |
| 🔍 | **Evaluation & explainability** | `evaluation` | Metrics with confidence intervals, calibration, **SHAP**, error analysis, **fairness and bias** checks, and **model cards**. |
| 🗂️ | **Tracking & registry** | `registry` | Experiment runs and dataset fingerprints. Models move dev → staging → production, with metric gates and rollback. |
| 🚀 | **Deployment** | `deploy` | Batch scoring, a REST API, ONNX and pickle export, **Docker images**, and a CLI. |
| 📡 | **Monitoring** | `monitoring` | Data and prediction drift, accuracy decay, alerts and webhooks, and champion-vs-challenger **retraining**. |
| 💡 | **Insights & reporting** | `insights` | Significance-tested findings, executive reports, and an optional narrative written by **Claude**. |
| 🏛️ | **Hub (platform server)** | `hub` | Multi-user server: SSO (OIDC) and RBAC, a shared registry with two-person production approvals and policy gates, a tamper-evident audit log, a web UI, Helm and Terraform. |
| 📈 | **Forecasting** | `forecast` | Multi-series, multi-step forecasts with backtests, calibrated intervals and hierarchical reconciliation. |
| 🧪 | **Causal & uplift** | `causal` | Doubly-robust treatment effects with diagnostics, and per-customer uplift models. |
| 🚨 | **Anomaly detection** | `anomaly` | An ensemble of three detectors with a plain-language reason for every flagged row. |
| 🖼️ | **Image & audio data** | `media` | Quality audits (corrupt, duplicate, blurry, clipped, silent…), consistent preprocessing, and features that let AutoML classify images and sounds. |
| 📐 | **Statistical testing** | `stats` | Chooses the right test for you (t-test, ANOVA, chi-square, …), with effect sizes, confidence intervals, A/B tests and power analysis. |
| 🗃️ | **Data versioning** | `versioning` | Commit, tag, diff and restore exact dataset versions, linked to experiment runs. |
| 🏪 | **Feature store** | `feature_store` | Define features once; leak-free point-in-time training data and low-latency online lookups. |
| ✂️ | **Data splitting** | `splitting` | Train / validation / test splits and CV folds (stratified, grouped, by time) that are checked for leakage. |
| 🛡️ | **Model security** | `robustness`, `poisoning` | Adversarial-robustness tests; detection of mislabelled, poisoned or backdoored training data. |
| 🚦 | **Environments & CI/CD** | `environments`, `cicd` | Isolated dev → QA → prod registries with gated, approved releases; generated train → gate → deploy pipelines. |
| 🔐 | **Secrets & policy** | `secretstore`, `hub.policy` | Credentials from Vault / AWS Secrets Manager; Open Policy Agent rules for access and promotions. |
| 📟 | **Observability** | `deploy`, `monitoring` | Grafana dashboard + Prometheus alerts, a feedback endpoint, hot reload, drift and accuracy *before* labels arrive. |
| 🧾 | **Lineage & catalogs** | `lineage`, `integrations` | OpenLineage events (Marquez, DataHub…), AWS Glue Data Catalog, Glue jobs and Azure Data Factory runs. |
| ☁️ | **Deploy on AWS** | `deploy.aws` | ECR push, blue/green SageMaker endpoints, Terraform for SageMaker and ECS Fargate. |
| 🔌 | **Integrations & scale** | `integrations`, `scale` | MLflow, DVC, Feast, Airflow, Prefect, Dagster, Kafka; Spark, Ray and Dask scoring; SQL pushdown into the warehouse. |

```
   ingest ─▶ clean ─▶ profile & quality ─▶ features ─▶ AutoML ─▶ evaluate & explain
                                                                        │
   insights ◀─ monitor ◀──── deploy ◀──── registry & tracking ◀─────────┘
                  │                                             ▲
                  └──── drift / accuracy decay ─▶ retrain ──────┘
```

**Built for production, not just notebooks:**
- **Reproducible:** pipelines learn on training data and replay exactly on new data. They save
  as reviewable **JSON**, never pickle.
- **Auditable:** every run records its parameters, data fingerprints, git commit, user and
  timings. Every promotion is logged.
- **Safe by default:**
  - SQL values are always bound as parameters, so inputs can't inject SQL.
  - Secrets are redacted everywhere.
  - Model files are checksummed.
  - The REST API supports API keys.
  - Docker images run as a non-root user.
- **Private:** drift monitors store binned statistics, not your rows. The LLM only ever sees
  aggregated findings.
