<a id="top"></a>

<div align="center">

![DataSpoof MLPilot](https://raw.githubusercontent.com/DataSpoof/dataspoof-data-cleaning/main/images/DataSpoof%20MLPilot%20Neon%20Banner.png)

# MLPilot

### From raw data to a monitored production model — automated, explainable, auditable.

[![PyPI](https://img.shields.io/pypi/v/dataspoof-mlpilot?color=2f6fde&label=PyPI)](https://pypi.org/project/dataspoof-mlpilot/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Rust engine](https://img.shields.io/badge/engine-Rust-B7410E?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Polars](https://img.shields.io/badge/dataframes-Polars-CD792C?logo=polars&logoColor=white)](https://pola.rs/)
[![Tests](https://img.shields.io/badge/tests-298%20passing-2ea44f)](#%EF%B8%8F-building--contributing)
[![License: MIT](https://img.shields.io/badge/license-MIT-2ea44f)](#-license)

**[Install](#-installation)** ·
**[Quick start](#-quick-start)** ·
**[Modules](#-the-platform-at-a-glance)** ·
**[Contents](#-contents)** ·
**[Why Rust?](#-why-rust--python)**

</div>

---

MLPilot automates the whole data-science lifecycle in one Python library. It covers
ingestion and cleaning, data quality, feature engineering, AutoML, explainability and fairness,
the model registry, deployment, monitoring, and plain-English insights. A **Rust engine**
handles the heavy computation, and every stage shares one `fit` / `transform` / `save` contract.

> **AutomatedCleaning is now MLPilot.** Install `dataspoof-mlpilot` and `import mlpilot`. Existing code and saved models keep working: see [Moving from AutomatedCleaning](#moving-from-automatedcleaning).

## ⚡ Quick start

```bash
pip install dataspoof-mlpilot
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

---

## 🧭 The platform at a glance

| | Stage | Module | What it automates |
|:-:|---|---|---|
| 📥 | **Ingestion & connectors** | `ingestion` | Files, globs, URLs, S3 / GCS / Azure, any SQL database, Snowflake, BigQuery and REST APIs. Includes schema inference and incremental loads. |
| 🧹 | **Data cleaning** | `cleaners`, `steps` | 50+ cleaning steps: missing values, outliers, text, dates, PII, duplicates, encodings. Rules are learned on training data and replayed exactly. |
| 📊 | **Profiling & quality** | `quality` | Automated EDA, a 0–100 quality score, YAML data contracts, schema-drift checks and quality gates. |
| 🧬 | **Feature engineering** | `features` | Dates, out-of-fold target encoding, interactions, aggregations, lags, text embeddings, selection and **leakage detection**. |
| 🤖 | **AutoML** | `automl` | Detects the task, runs a time-budgeted model search, builds an ensemble, and also does clustering. |
| 🔍 | **Evaluation & explainability** | `evaluation` | Metrics with confidence intervals, calibration, **SHAP**, error analysis, **fairness and bias** checks, and **model cards**. |
| 🗂️ | **Tracking & registry** | `registry` | Experiment runs and dataset fingerprints. Models move dev → staging → production, with metric gates and rollback. |
| 🚀 | **Deployment** | `deploy` | Batch scoring, a REST API, ONNX and pickle export, **int8 / fp16 quantization**, **Docker images**, and a CLI. |
| 📡 | **Monitoring** | `monitoring` | Data and prediction drift, accuracy decay, alerts and webhooks, and champion-vs-challenger **retraining**. |
| 💡 | **Insights & reporting** | `insights` | Significance-tested findings, executive reports, and an optional narrative written by **Claude**. |
| 🏛️ | **Hub (platform server)** | `hub` | Multi-user server: SSO (OIDC) and RBAC, a shared registry with two-person production approvals and policy gates, a tamper-evident audit log, a web UI, Helm and Terraform. |
| 📈 | **Forecasting** | `forecast` | Multi-series, multi-step forecasts with backtests, calibrated intervals and hierarchical reconciliation. |
| 🧪 | **Causal & uplift** | `causal` | Doubly-robust treatment effects with diagnostics, and per-customer uplift models. |
| 🚨 | **Anomaly detection** | `anomaly` | An ensemble of three detectors with a plain-language reason for every flagged row. |
| 🖼️ | **Image & audio data** | `media` | Quality audits (corrupt, duplicate, blurry, clipped, silent…), consistent preprocessing, and features that let AutoML classify images and sounds. Weather / low-light detection, simulation and restoration, faces covered by hair, masks or sunglasses, licence-plate reading, face and plate blurring, and per-condition robustness benchmarks. |
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

---

## 🗺️ How it all fits together

<p align="center"><img src="https://raw.githubusercontent.com/DataSpoof/dataspoof-data-cleaning/main/docs/assets/workflow.svg" alt="MLPilot workflow: data in, build and validate, release through isolated dev, QA and prod registries, and operate with feedback, monitoring and a retraining loop. Five gates can stop a run." width="100%"></p>

Read lane 1 left to right, lane 2 right to left, and so on: the flow snakes down the page.

- **Gates** (amber) can stop a run: the quality gate, the poisoning check, the CI gate, and the two
  releases between environments.
- **Dashed lines** are side paths. The held-out test split skips feature engineering and AutoML and
  goes straight to evaluation. Served inputs reach the monitor at once, so drift and estimated
  accuracy are known before any labels exist.
- **The orange loop** is retraining. A challenger re-enters at AutoML, so it passes the same gates
  and releases as the model it may replace.
- **Across every stage:** secrets come from Vault or AWS Secrets Manager, OPA policies and the Hub's
  audit log govern changes, and every run reports lineage and experiment tracking.

| Lane | Stages | Read more |
|---|---|---|
| 1 · Data in | sources → ingest → version → clean → quality gate → split | [1](#-1--ingestion--connectors) · [18](#%EF%B8%8F-18--data-versioning) · [2](#-2--data-cleaning) · [3](#-3--profiling--data-quality) · [20](#%EF%B8%8F-20--data-splitting) |
| 2 · Build & validate | features → poisoning check → AutoML → evaluate → CI gate → register | [4](#-4--feature-engineering) · [19](#-19--feature-store) · [21](#%EF%B8%8F-21--model-security) · [5](#-5--automl) · [6](#-6--evaluation--explainability) · [22](#-22--environments--cicd) |
| 3 · Release | dev → QA → prod registries, approvals, OPA, signatures → deploy | [7](#%EF%B8%8F-7--experiment-tracking--model-registry) · [23](#-23--secrets--policy) · [8](#-8--deployment) · [26](#%EF%B8%8F-26--deploy-on-aws) |
| 4 · Operate | serve → feedback → monitor → alerts → retrain | [24](#-24--observability) · [9](#-9--monitoring) · [25](#-25--lineage--catalogs) |

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 📚 Contents

- [How it all fits together](#%EF%B8%8F-how-it-all-fits-together)

1. [Ingestion & connectors](#-1--ingestion--connectors)
2. [Data cleaning](#-2--data-cleaning)
3. [Profiling & data quality](#-3--profiling--data-quality)
4. [Feature engineering](#-4--feature-engineering)
5. [AutoML](#-5--automl)
6. [Evaluation & explainability](#-6--evaluation--explainability)
7. [Experiment tracking & model registry](#%EF%B8%8F-7--experiment-tracking--model-registry)
8. [Deployment](#-8--deployment)
9. [Monitoring](#-9--monitoring)
10. [Insights & reporting](#-10--insights--reporting)
11. [Hub: the platform server](#%EF%B8%8F-11--hub-the-platform-server)
12. [Forecasting](#-12--forecasting)
13. [Causal inference & uplift](#-13--causal-inference--uplift)
14. [Anomaly detection](#-14--anomaly-detection)
15. [Integrations & scale](#-15--integrations--scale)
16. [Image & audio data](#%EF%B8%8F-16--image--audio-data)
17. [Statistical testing](#-17--statistical-testing)
18. [Data versioning](#%EF%B8%8F-18--data-versioning)
19. [Feature store](#-19--feature-store)
20. [Data splitting](#%EF%B8%8F-20--data-splitting)
21. [Model security](#%EF%B8%8F-21--model-security)
22. [Environments & CI/CD](#-22--environments--cicd)
23. [Secrets & policy](#-23--secrets--policy)
24. [Observability](#-24--observability)
25. [Lineage & catalogs](#-25--lineage--catalogs)
26. [Deploy on AWS](#%EF%B8%8F-26--deploy-on-aws)

Also: [Installation](#-installation) · [Why Rust + Python?](#-why-rust--python) ·
[Requirements](#-requirements) · [Building & contributing](#%EF%B8%8F-building--contributing) ·
[License](#-license)

---

## 📦 Installation

```bash
pip install dataspoof-mlpilot                    # lean core (pipelines, quality, features, AutoML, …)
pip install "dataspoof-mlpilot[sql,excel]"       # + the extras you need (see below)
pip install "dataspoof-mlpilot[all]"             # everything, like 1.x installed by default
```

Since 2.0, the core install no longer pulls in the heavy dependencies (NLTK, Plotly, Presidio,
LangChain). Those features ask for their extra when you use them. Importing the library never
downloads data, writes files or changes your logging. Set `AC_OFFLINE=1` to guarantee no
network use at all.

The import name is `mlpilot` (`import mlpilot as ac`). It needs
**Python 3.11+**.

> **Platform note:** PyPI currently has a prebuilt wheel for **Windows**. On Linux and macOS,
> including Colab, `pip` builds from source and needs the Rust toolchain
> (`curl https://sh.rustup.rs -sSf | sh`). The included GitHub Actions workflow builds wheels
> for every platform.

| Extra | Adds | For |
|---|---|---|
| `dashboard` / `nlp` / `pii` | Plotly…, NLTK, Presidio | the EDA dashboard, text lemmatisation, PII masking (`clean_data` uses all three) |
| `server` | FastAPI, SQLAlchemy, PyJWT, psycopg | the Hub platform server |
| `image` / `audio` | Pillow / soundfile | image data; non-WAV audio (WAV needs nothing extra) |
| `vision` / `ocr` | OpenCV 4 / + RapidOCR | faces (coverage by hair, mask, sunglasses; blurring) / licence-plate reading |
| `mlflow` / `dvc` / `feast` / `evidently` | MLflow, DVC, Feast, Evidently | bridges to those tools (not in `all`: each is large) |
| `aws` / `azure` | boto3 / azure-identity | Glue, SageMaker + ECR deploys, `aws-sm://` secrets; Azure Data Factory sign-in |
| `optuna` / `skops` / `signing` | Optuna, skops, cryptography | Bayesian search, safe model export, model signing |
| `sql` | SQLAlchemy (plus your database's driver) | PostgreSQL, MySQL, SQL Server, Oracle, Redshift… |
| `excel` | fastexcel | `.xlsx` / `.xls` files |
| `snowflake` / `bigquery` | native warehouse connectors | Snowflake, BigQuery |
| `automl` | LightGBM, XGBoost | extra AutoML model families |
| `embeddings` | sentence-transformers | neural text embeddings |
| `explain` | shap | shap's own plots from `Explanation.to_shap()` |
| `onnx` | skl2onnx, onnxruntime | ONNX export |
| `serve` | gunicorn / waitress | production REST serving |
| `llm` | anthropic, langchain-anthropic | Claude-written narratives and AI categorical fixes |
| `text` | langdetect, better-profanity, pyspellchecker | language / profanity / spelling steps |

### Moving from AutomatedCleaning

MLPilot is the new name of **AutomatedCleaning** (renamed in 2.1.0, when the library had grown from
data cleaning into the whole model lifecycle). Nothing else changed.

On PyPI it is **`dataspoof-mlpilot`**: PyPI treats `mlpilot` as too close to an existing project. You
install `dataspoof-mlpilot`, and import and run `mlpilot`, the same way `pip install scikit-learn`
gives you `import sklearn`.

```bash
pip uninstall AutomatedCleaning && pip install dataspoof-mlpilot
```

| Before | Now |
|---|---|
| `pip install AutomatedCleaning` | `pip install dataspoof-mlpilot` |
| `import automatedcleaning as ac` | `import mlpilot as ac` |
| `automatedcleaning serve ...` (CLI) | `mlpilot serve ...` |
| `AutomatedCleaningWarning` | `MLPilotWarning` (the old name still works) |

**What keeps working without changes:**
- **Saved artifacts:** models, registries, monitors and pipelines saved by AutomatedCleaning load
  as before, including raw pickles that refer to `automatedcleaning.*` classes.
- **Old imports:** once `mlpilot` is installed, `import automatedcleaning` still works and gives
  the same library.
- **Running services:** the `AC_*` environment variables and the `ac_*` Prometheus metrics keep
  their names, so deployments and dashboards are unaffected.
- **The old package name:** `pip install AutomatedCleaning` now installs MLPilot (`dataspoof-mlpilot`) through a small
  forwarding package (2.0.1).

**Renamed with the library:**
- the OPA policy package (`data.mlpilot`) and the Prometheus job (`mlpilot`);
- the Helm chart (`mlpilot-hub`);
- the MLflow tag prefix (`mlpilot.`) and the default OpenLineage namespace.

Update these if you deployed them from 2.0.0.

<details>
<summary><b>Install from source</b></summary>

```bash
pip install maturin                               # needs Rust: https://rustup.rs
git clone https://github.com/DataSpoof/dataspoof-data-cleaning.git
cd dataspoof-data-cleaning
pip install .                                     # or: maturin develop --release (in a virtualenv)
```

</details>

---

## 📥 1 · Ingestion & connectors

`ac.read()` reads from any supported source and returns a Polars DataFrame. `ac.ingest()` also
infers types, validates the data and records where it came from.

```python
ac.read("sales.csv")                                   # csv, tsv, parquet, json, ndjson, excel, arrow, avro, .csv.gz
ac.read("data/2024-*.parquet", include_source=True)    # globs -> one table (+ _source_file column)
ac.read("https://example.com/export.csv")
ac.read("s3://bucket/sales/*.parquet", storage_options={"aws_region": "eu-west-1"})   # also gs://, az://, abfss://
ac.read("postgresql+psycopg://user:pass@host/db", query="SELECT * FROM orders WHERE region = :r",
        params={"r": "EU"})
ac.read("sqlite:///shop.db", table="orders")

result = ac.ingest("sales.csv", name="sales", contract="sales.contract.yaml", quality=True)
result.data                 # typed DataFrame: "1,200" -> 1200.0, "2024-01-05" -> date, "yes" -> True
print(result)               # rows, inferred types, contract result, quality score
result.fingerprint          # shape + schema + content hash (also logged with ingest(..., run=run))
```

**Connectors:**
- **Cloud storage:** S3, GCS and Azure are read natively by Polars, with no extra packages.
  Credentials come from `storage_options` or the standard environment variables.
- **Databases:** `SQLSource` takes any SQLAlchemy URL (PostgreSQL, MySQL, SQL Server, Oracle,
  Redshift, DuckDB, …) or any DB-API connection. Install with
  `pip install "dataspoof-mlpilot[sql]"` plus your database's driver.
- **Warehouses:** `SnowflakeSource` (`[snowflake]`) and `BigQuerySource` (`[bigquery]`) use the
  native connectors and fetch results as Arrow.
- **APIs:** `RESTSource` handles token auth (pass the token, or read it from an environment
  variable), and every pagination style: page, offset, cursor, next-URL and `Link` header. It
  retries 429 and 5xx responses with back-off and `Retry-After`, supports rate limiting, and
  flattens nested JSON into columns.

**Safety:**
- Values are always bound as query parameters (write `:name`), and table and column names are
  quoted, so inputs can't inject SQL.
- Passwords and tokens are redacted from every repr and report.
- An API never follows a next-page link to another host, so your token can't leak there.

```python
api = ac.RESTSource("https://api.example.com/v1", token_env="EXAMPLE_TOKEN", rate_limit=5)
orders = api.fetch("orders", records="data", pagination="cursor",
                   cursor_param="after", cursor_path="meta.next_cursor")
```

**Incremental loads** read only what is new since the last successful load. The state is saved
only after your code has processed the batch, so a crash re-delivers the rows instead of
dropping them. If two runners try to commit, the second gets an error rather than overwriting.

```python
loader = ac.IncrementalLoader(ac.SQLSource(url), table="orders", cursor="updated_at",
                              key="order_id", state="state/orders.json")
with loader.batch() as new_rows:          # rows past the saved watermark; late rows with the
    append_to_warehouse(new_rows)         # same timestamp are caught and de-duplicated by key

files = ac.IncrementalLoader(files="landing/**/*.csv", state="state/landing.json")
files.run(lambda df: df.write_parquet("bronze/batch.parquet"))   # only new or changed files
```

Types are learned from the first batch and then reused, so every later batch has the same
schema.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🧹 2 · Data cleaning

### Interactive cleaning

```python
import mlpilot as ac

# Load any CSV / TSV / JSON / Parquet file (via Polars)
df = ac.load_data("dataset.csv")

# Run the full interactive cleaning pipeline
df_cleaned = ac.clean_data(df, background_image_path="assets/gradient.png")
```

`clean_data()` walks the dataset through the whole pipeline and asks a few interactive
questions (column spelling fixes, categorical corrections, target column for imbalance).
It writes `cleaned_data.csv` and an EDA dashboard to `output/eda/dashboard.html`.

### Pipelines: fit once, apply anywhere (recommended for production)

`clean_data()` is interactive and re-learns everything from whatever data it is given. For
scheduled jobs, train/test splits and production scoring, use a **pipeline** instead. It asks
no questions, learns its cleaning rules on training data, and replays exactly those rules on
new data, so a test set or a production batch never influences how it is cleaned.

```python
import mlpilot as ac

pipe = ac.default_cleaning_pipeline()
train_clean = pipe.fit_transform(train_df)   # learn the rules on training data
test_clean = pipe.transform(test_df)         # same rules, nothing re-learned

print(pipe.report_)                          # what each step did (rows, columns, nulls, time)
pipe.save("cleaning_pipeline.json")          # everything it learned, as plain JSON (no pickle)

# later, in a scheduled job or an API
pipe = ac.Pipeline.load("cleaning_pipeline.json")
batch_clean = pipe.transform(batch_df)
```

The default pipeline: fix missing-value placeholders → normalise text → repair numbers and
dates stored as text → drop duplicate rows (training only) → drop useless columns (mostly
missing, constant, ID-like, duplicated) → impute, adding a `<col>_was_missing` flag for
columns ≥5% missing → group rare categories → cap extreme outliers (3×IQR) → drop
near-duplicate numeric columns. Change any step by name:

```python
pipe = ac.default_cleaning_pipeline(clip_outliers=False, impute={"numeric": "mean"})
```

### Build your own pipeline

Mix the stateful steps (which learn in `fit`) with any of the functions above:

```python
pipe = ac.Pipeline([
    "replace_disguised_missing",                       # any library function, by name
    ac.ConvertNumericText(),
    ac.Impute(numeric="median", add_indicators=True),
    ac.EncodeCategorical(method="onehot"),
    ac.Scale(method="standard"),
], name="churn-features")
```

Stateful steps: `drop_columns`, `drop_collinear`, `convert_numeric_text`, `parse_dates`,
`impute`, `clip_outliers`, `remove_outliers`, `log_transform_skewed`, `scale`,
`group_rare_categories`, `encode_categorical`, `drop_duplicates`. Run
`ac.available_steps()` to list every step name with a description.

### Config files

Keep pipelines in version control as YAML or JSON:

```yaml
# cleaning.yaml
name: churn-cleaning
on_schema_mismatch: raise        # warn (default) | raise | ignore
steps:
  - step: replace_disguised_missing
    params: {numeric_sentinels: [-999]}
  - step: drop_columns
    params: {max_missing: 0.5}
  - step: impute
    params: {numeric: median, add_indicators: true}
  - step: remove_outliers          # drops rows during fit only
```

```python
pipe = ac.Pipeline.from_config("cleaning.yaml")
```

### Built for production

- **Schema checks:** `transform` compares incoming columns and types with the training data
  and warns, or raises with `on_schema_mismatch="raise"`.
- **Training-only steps:** row-removing steps (`drop_duplicates`, `remove_outliers`) run during
  `fit` and are skipped by `transform`, so scoring never silently drops records.
- **Audit trail:** every run produces a `RunReport` with a run ID, timings, per-step changes,
  data fingerprints, the config hash and library versions. Pass `audit_log="runs.jsonl"`
  to append each run to a file.
- **Safe persistence:** saved pipelines are plain JSON you can review, diff and load
  safely. Nothing is pickled.
- **Extensible:** subclass `ac.Step` and decorate it with `@ac.register_step` to add your own
  steps; they work in configs and in saved pipelines like the built-in ones.

### Individual cleaning functions

Every stage is also a standalone function you can call directly — useful for scripting or
building your own non-interactive pipeline. These delegate to the Rust backend:

```python
import polars as pl
import mlpilot as ac

df = ac.load_data("dataset.csv")

df = ac.detect_column_types_and_process_text(df)  # classify + clean text columns (Rust)
df = ac.handle_negative_values(df)                # negatives -> absolute values (Rust)
df = ac.replace_symbols(df)                       # strip $ ₹ , - •  (Rust)
df = ac.handle_missing_values(df)                 # KNN impute + mode fill (Python/sklearn)
df = ac.handle_duplicates(df)                     # drop duplicate rows (Polars)
df = ac.remove_outliers(df)                       # IQR outlier removal (Rust)
df = ac.fix_skewness(df)                          # log-transform skewed columns (Rust)
df = ac.check_multicollinearity(df, threshold=0.7)# drop correlated features (Rust)
df, cardinality = ac.check_cardinality(df)        # report + drop constant columns
df = ac.fix_json_columns(df)                      # expand JSON columns (Rust detection)
df = ac.detect_and_mask_pii_polars(df)            # detect + mask PII (Presidio)
ac.generate_dashboard(df)                         # Plotly EDA dashboard
ac.save_cleaned_data(df, "cleaned_data.csv")
```

#### Text preprocessing on its own

```python
ac.preprocess_text("I can't wait!! Visit https://x.co @bob 😀 it's GR8")
# -> 'cannot wait visit'
```

<details>
<summary><b>All additional cleaning steps (<code>mlpilot.steps</code>)</b></summary>


Beyond the core pipeline, ~45 focused, composable steps are available directly
on `ac.*`. String-heavy ones (whitespace, Unicode NFKC, mojibake, disguised-missing,
booleans) run in the Rust backend.

```python
df = ac.standardize_column_names(df)          # trim/lowercase/snake_case/de-dupe
df = ac.drop_empty_rows_and_columns(df)
df = ac.drop_high_missing_columns(df, 0.5)
df = ac.remove_duplicate_columns(df)
df = ac.drop_id_like_columns(df)
df = ac.replace_disguised_missing(df, numeric_sentinels=[999, -1])  # "NA","?"… -> null (Rust)
df = ac.add_missing_indicators(df)
df = ac.impute_missing(df, numeric_strategy="mice")   # mean|median|knn|mice|interpolate|ffill|bfill
df = ac.normalize_whitespace(df)              # Rust
df = ac.normalize_unicode(df)                 # NFKC + control-char strip (Rust)
df = ac.fix_mojibake(df)                       # "Ã©" -> "é" (Rust)
df = ac.standardize_casing(df, {"name": "title"})
df = ac.parse_dates(df, ["signup"])           # -> native datetime
df = ac.extract_date_parts(df, ["signup"])    # year/month/day/weekday/quarter
df = ac.normalize_booleans(df, ["active"])    # yes/no/Y/N/1/0 -> bool (Rust)
df = ac.downcast_numeric(df)                  # int64->int32, etc.
df = ac.group_rare_categories(df, 0.01)
df = ac.fuzzy_cluster_categories(df)          # merge "New York"/"newyork" (rapidfuzz)
df = ac.encode_categorical(df, "onehot")      # onehot|label|frequency|ordinal|target
df = ac.remove_outliers_zscore(df)            # + _mad, _isolation_forest, _by_group
df = ac.winsorize(df)                          # cap instead of drop
df = ac.fuzzy_deduplicate(df, ["name","email"])
df = ac.scale_numeric(df, "standard")         # minmax|standard|robust
df = ac.bin_numeric(df, ["age"], bins=5)
df = ac.apply_range_rules(df, {"age": (0, 120)}, action="clip")
df = ac.validate_formats(df, {"email": "email", "phone": "phone"})
df = ac.normalize_phone_numbers(df, ["phone"], region="US")   # -> E.164
df = ac.normalize_emails(df, ["email"]); df = ac.canonicalize_urls(df, ["site"])
```

Steps that need optional libraries (`detect_language`, `redact_profanity`,
`correct_spelling_text`) are installed via the extra:

```bash
pip install "dataspoof-mlpilot[text]"
```

</details>

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 📊 3 · Profiling & data quality

### Automated EDA

```python
p = ac.profile(df)
print(p)                 # overview, every column's type and meaning, plain-English insights
p["Fare"].numeric        # min, max, mean, median, std, p01…p99, skew, outliers, …
p.to_frame()             # one row per column, ready to filter or export
p.to_json("profile.json")
```

Each column gets a **semantic type** describing what it holds, not just how it's stored:
`identifier`, `binary`, `categorical`, `numeric`, `discrete`, `text`, `email`, `phone`, `url`,
`datetime`, `numeric_text` (numbers stored as text), `datetime_text` (dates stored as text),
`constant` or `empty`. Insights are generated for findings such as:

- columns that are mostly empty
- identifier and constant columns
- strong correlations
- heavy skew
- columns dominated by a single value

### Quality score

```python
report = ac.check_quality(df)
print(report)                  # score, grade, dimension bars, weakest columns, top issues
report.score, report.grade     # e.g. 97.2, "A"
report.dimensions              # completeness / uniqueness / validity / consistency, each 0-100
report.weakest_columns()       # e.g. [("Cabin", 50.9), ("Age", 89.3)]
report.to_frame()              # every issue with severity, column and cells affected
```

The score combines four standard data-quality dimensions:

- **completeness:** real values, counting `"NA"`/`"?"` placeholders and `-999` codes as missing
- **uniqueness:** rows that aren't duplicates
- **validity:** values pass range, email, phone and calendar-date rules
- **consistency:** no stray whitespace, case variants or mixed date formats

The default weights are 0.35 / 0.15 / 0.30 / 0.20 and can be changed with `weights=`. Grades run
A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, otherwise F. The overall score averages over every cell, so one
broken column barely moves it; `weakest_columns()` is there to catch that.

### Data contracts

Declare what valid data looks like and check every batch against it:

```yaml
# customers.contract.yaml
name: customers
extra_columns: warn            # allow | warn | error
no_duplicate_rows: true
unique: [customer_id]
columns:
  customer_id: {type: integer, required: true, unique: true}
  email:       {type: string, format: email, max_null_fraction: 0.01}
  age:         {type: number, min: 0, max: 120, warn_only: [min, max]}
  plan:        {type: string, allowed: [basic, pro, enterprise]}
```

```python
contract = ac.Contract.from_file("customers.contract.yaml")
result = contract.validate(batch_df)
print(result)                  # PASSED/FAILED with every broken rule and example values
result.raise_if_failed()       # raise ContractViolation in a job or CI

# No contract yet? Infer one from data you trust, review it, then commit it:
ac.Contract.infer(train_df).to_yaml("customers.contract.yaml")
```

Inferred contracts make structural rules errors (columns, types, required, unique) and
statistical rules warnings (ranges, allowed values, lengths, null fractions), because new
data legitimately drifts.

### Schema checks

```python
diff = ac.compare_schemas(train_df, new_df)
diff.removed, diff.added, diff.changed      # e.g. {"age": ["Float64", "String"]}
diff.compatible                             # False if a column vanished or changed type
```

### Enforce quality inside pipelines

```python
pipe = ac.Pipeline([
    ac.ValidateContract(),                       # infers a contract from training data on fit
    *ac.default_cleaning_pipeline().steps,
    ac.QualityGate(min_score=85, min_dimensions={"completeness": 90}),
])
pipe.fit(train_df)
pipe.save("pipeline.json")                       # the inferred contract is saved inside it
ac.Pipeline.load("pipeline.json").transform(new_batch)   # raises if the batch breaks it
```

Use `on_fail="warn"` on either step to log a warning instead of stopping. The EDA dashboard's
Overview tab is built on this module, so it shows the same score, weakest columns and issues.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🧬 4 · Feature engineering

Feature steps work like the cleaning steps: they learn in `fit`, replay in `transform`, and save
as JSON. Give the pipeline a `target` so steps that need the label can use it:

```python
pipe = ac.Pipeline(
    [*ac.default_cleaning_pipeline().steps, *ac.default_feature_pipeline("churned").steps],
    target="churned",
)
train_features = pipe.fit_transform(train_df)       # model-ready numeric features + the label
new_features = pipe.transform(new_df)               # the label isn't needed to predict
```

| Step | What it adds |
|---|---|
| `DateTimeFeatures` | year, month, day, weekday, quarter, weekend/month-start/month-end flags, hour, sine/cosine encodings, days elapsed |
| `TargetEncode` | each category's average target value. Training rows are encoded **out-of-fold**, so a row never sees its own label |
| `Interactions` | products and ratios of the most informative numeric columns |
| `GroupAggregations` | per-group statistics such as `fare_mean_by_class` (never of the target) |
| `LagFeatures` | time-series lags and rolling means, per entity, in time order |
| `TextEmbeddings` | TF-IDF + SVD vectors (built in), or `method="sentence_transformers"` |
| `SelectFeatures` | keep features by random-forest importance, mutual information or variance |
| `DropLeakage` | detect and drop columns that leak the target |

The default pipeline runs these steps in order:

1. date/time features
2. drop leaky columns
3. text embeddings
4. target encoding for columns with 16+ categories
5. one-hot encoding for the rest
6. feature selection

`time_col="..."` adds lag features and turns off target encoding.

**Target-aware pipelines:** with `target=`, ordinary steps never see the label. That means
imputation can't invent labels, outlier clipping can't change them, and encoders can't encode
them. Row-removing steps drop the same rows from the label, and the label can be absent at
prediction time.

**Leakage detection:** it scores every column on its own against the target. A column that
predicts almost perfectly is usually recorded *after* the outcome (`refund_issued` when
predicting churn), and such columns are flagged:

```python
report = ac.detect_leakage(df, "churned")
print(report)            # [high] refund_issued: on its own it predicts 'churned' with cv roc_auc = 1.000 ...
```

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🤖 5 · AutoML

```python
model = ac.AutoML("churned", time_budget=120).fit(train_df)
print(model.summary())               # task, metric, CV scheme, best model, best score per model
model.leaderboard_                   # every trial with its score, std, time and parameters
model.evaluate(test_df)              # accuracy, balanced accuracy, F1, ROC-AUC, log-loss (RMSE, MAE, R² for regression)
model.predict(new_df)                # labels in their original form
model.predict_proba(new_df)          # one proba_<class> column per class
model.feature_importance()           # permutation importance
model.save("churn_model/")           # directory: manifest.json + model.joblib + preprocess.json
model = ac.AutoML.load("churn_model/")
```

`AutoML(target).fit(df)` works through these steps:

1. **Detect the task.** Two values means binary; text, or up to 10 whole-number values, means
   multiclass; anything else is regression. Rows sorted by a date column are treated as a time
   series.
2. **Pick a metric:** ROC-AUC, macro-F1 or RMSE.
3. **Drop leaky columns.**
4. **Clean and engineer features** with the target-aware pipelines above.
5. **Cross-validate a baseline and every model.** The defaults are logistic/ridge regression,
   histogram gradient boosting, random forest, extra trees, k-NN, and LightGBM and XGBoost if
   installed. Folds are stratified, or time-ordered for time series.
6. **Spend the rest of `time_budget` on random hyperparameter search** around the best model
   families.
7. **Try a soft-voting ensemble** of the top three families.
8. **Refit the winner on all rows.**

| Option | Default | Meaning |
|---|---|---|
| `time_budget` | 60 | seconds for the search (75% models, the rest ensemble + refit) |
| `max_trials` | None | cap on model trials |
| `metric` | auto | any scikit-learn scorer name, e.g. `"average_precision"` |
| `cv` | 5 | number of folds |
| `include` / `exclude` | all | model names, see `ac.available_models("binary")` |
| `preprocess` | `"auto"` | `"auto"`, your own `Pipeline`, or `None` |
| `ensemble` | True | try the voting ensemble |
| `on_leakage` | `"drop"` | `drop`, `warn`, `raise` or `ignore` |
| `time_col` | `"auto"` | a date column for time-ordered CV, or None to switch detection off |

On Titanic, with the defaults and a 40-second budget, AutoML reaches 0.918 ROC-AUC and 83% accuracy
on held-out rows.

**Saved models:** a trained model can't be stored as JSON, so `model.joblib` uses joblib, the
scikit-learn standard. Its SHA-256 is recorded in `manifest.json` and verified before loading,
which catches corrupted or swapped files. Unpickling still runs code, so only load models from
people you trust.

### Uncertainty, imbalance and smarter search

```python
model = ac.AutoML("price", search="optuna").fit(train)   # Bayesian (TPE) search: pip install "...[optuna]"
model.predict_interval(test, coverage=0.9)                # prediction, lower, upper (conformal)
clf.predict_set(test, coverage=0.9)                       # label sets that contain the truth 90% of the time
```

- **Conformal intervals and sets** are calibrated on out-of-fold errors, so the stated
  coverage holds on new data (90% intervals covered 90.2% of held-out values in testing).
- **Imbalanced classes:** with `imbalance="auto"` (the default), training uses balanced class
  weights when the smallest class is under 20%.
- **Parallel cross-validation:** `n_jobs=-1` runs the folds in parallel.
- **Fast scoring:** predictions run single-threaded by default, which keeps single-row API
  latency low. Use `set_inference_threads(-1)` for large batch jobs.

### Clustering

```python
clusters = ac.AutoCluster(k_range=(2, 8)).fit(df)     # k-means and Gaussian mixtures, chosen by silhouette
clusters.labels_                                      # one label per input row
clusters.cluster_profiles(df)                         # size, share and averages per cluster
ac.automl(df)                                         # no target -> AutoCluster; with target= -> AutoML
```

Optional extras: `pip install "dataspoof-mlpilot[automl]"` (LightGBM, XGBoost) and
`pip install "dataspoof-mlpilot[embeddings]"` (sentence-transformers).

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🔍 6 · Evaluation & explainability

Every tool takes a fitted `AutoML` model (or any scikit-learn estimator plus `target=...`) and
labelled data. Each is also a method on `AutoML`.

```python
model = ac.AutoML("Survived").fit(train)

report = model.report(test)              # == ac.evaluate_model(model, test)
print(report)                            # accuracy, F1, ROC-AUC, PR-AUC, MCC, log loss, Brier, ECE ...
report.confusion_matrix                  # also: per_class, calibration, roc_curve, pr_curve
report.confidence_intervals              # 95% bootstrap intervals for the headline metrics
report.best_threshold("f1")              # threshold table for binary models -> best cut-off
ac.evaluate_model(model, test, threshold=0.3)   # metrics at a different decision threshold
```

Regression reports give RMSE, MAE, median/max error, R², bias, MAPE/sMAPE, the residual
distribution and the error for each range of the actual value.

**SHAP explanations.** Shapley values are computed by the library itself, with no `shap` dependency. They work for any
model, including AutoML's ensembles and preprocessing. With up to 8 features every
coalition is evaluated (exact SHAP); otherwise antithetic permutation sampling is used. For every
row `base_value + sum(values)` equals the model's output exactly.

```python
exp = model.explain(test, rows=100)      # == ac.explain(model, test)
print(exp)                               # global importance (mean |SHAP|) + direction of effect
exp.global_importance()                  # feature, mean_abs_shap, share, direction
exp.explain_row(0)                       # "P(Survived = 1) = 0.27 (average 0.42): Pclass = 3 (-0.174), ..."
exp.for_row(0)                           # contributions for one row, largest first
exp.to_shap()                            # a shap.Explanation for shap's plots: pip install "dataspoof-mlpilot[explain]"
ac.partial_dependence(model, test, "Fare")   # average prediction as one feature varies
```

**Error analysis** finds the worst predictions and the segments of the data where the model
fails. A segment is a category, a quantile bin, missing values, or a pair of these. Each
segment is tested against the rest of the data, with a Benjamini–Hochberg correction so random
noise isn't flagged.

```python
errors = model.error_analysis(test)
errors.problem_slices        # e.g. "region = east": error rate 31% (2.4x), q = 0.001
errors.worst                 # most confident mistakes / largest errors, with their inputs
errors.confusions            # most common actual -> predicted mistakes
```

**Fairness & bias checks.** These cover each sensitive attribute, intersections such as sex × age
band, and *proxy* features that encode a sensitive attribute.

```python
fair = model.fairness(test, ["Sex", "Age"])      # numeric attributes are cut into quantile bands
print(fair)            # selection rate, TPR, FPR, precision and accuracy per group, plus flags
fair.passed            # False if any limit is exceeded
fair["Sex"].groups     # the per-group table
fair.proxies           # Cramér's V between every feature and each sensitive attribute
```

The checks cover demographic parity, disparate impact (four-fifths rule), equal opportunity,
equalized odds, predictive parity and accuracy gaps. For regression, they compare error and bias
across groups. The default limits are in `ac.evaluation.DEFAULT_LIMITS`; override them with
`limits={...}`.

**Model cards.** One call generates a full card in the standard layout (Mitchell et al.):
- model details and training data;
- the model-selection leaderboard;
- held-out metrics with confidence intervals and calibration;
- SHAP importance, error analysis and fairness;
- the quality score of the evaluation data;
- auto-written limitations and ethical considerations;
- versions.

```python
card = model.model_card(test, sensitive=["Sex"], name="Titanic survival",
                        intended_use="Teaching example; not for real decisions.", owners="Data team")
card.save("model_card.html")     # self-contained page (light/dark); also .md and .json
card.limitations                 # e.g. "Performs significantly worse on some segments: ..."
```

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🗂️ 7 · Experiment tracking & model registry

Both are plain directories of JSON files. There is no server to run, and they are easy to back
up, diff or keep on a shared drive.

```python
tracker = ac.Tracker("ac_runs", experiment="titanic")
with tracker.start_run("baseline") as run:           # ends "finished", or "failed" with the error
    model = ac.AutoML("Survived").fit(train)
    run.log_automl(model, train, test)               # params, CV + test metrics, dataset fingerprints,
                                                     # leaderboard, the model and its model card
    run.log_metric("custom", 0.9, step=1)            # anything else: log_params / log_artifact / set_tag

tracker.runs()                                       # every run as a table (params, latest metrics)
tracker.best_run("test_roc_auc")
tracker.compare([run_a.id, run_b.id])
```

Each run records the git commit, the user and its timings. A dataset is recorded by its shape,
schema and content hash, never by its rows.

```python
reg = ac.ModelRegistry("ac_registry")
v = reg.register(model, "titanic", run=run)          # -> v1, stage "dev"
reg.promote("titanic", v.version, "staging")
reg.promote("titanic", v.version, "production",
            require={"test_roc_auc": 0.85},          # metric gates; failing ones raise PromotionError
            must_beat_production="test_roc_auc")     # ...and it must beat the current production model
model = reg.load("titanic")                          # the production model (checksum verified)
reg.history("titanic")                               # who moved what, when and why
reg.rollback("titanic")                              # restore the previous production version
```

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🚀 8 · Deployment

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
waitress on Windows, and install either with `pip install "dataspoof-mlpilot[serve]"`.

**Docker.** `build_docker` writes a multi-stage Dockerfile: a non-root user, a health check,
gunicorn, and the model baked in. The generated image has been built and tested end to end.

**ONNX** (`pip install "dataspoof-mlpilot[onnx]"`) exports the trained estimator: linear models,
random forests, extra trees and voting ensembles. Pass `sample=df` and the export is checked
against the original model. HistGradientBoosting can't be exported yet because of an upstream
skl2onnx limitation; the error says so.

**Quantization** shrinks neural-network models to 8-bit or 16-bit weights, and measures the
cost on held-out data first. See the [quantization guide](docs/guide/quantization.md).

```python
rep = ac.quantize_model(model, test, out="build/quantized")   # int8 + fp16 variants: size, speed, metric, agreement
ac.export_model(model, "build/onnx", "onnx", quantize="int8") # model trained with AutoML(..., include=["mlp"])
```

**The CLI** (also available as `python -m mlpilot.deploy`):

```bash
mlpilot serve  --model models/titanic --port 8000
mlpilot score  --model models/titanic --input big.csv --output scored.parquet
mlpilot --registry ac_registry docker --model titanic:production --out build/
mlpilot --registry ac_registry monitor --model titanic --monitor monitor.json --input today.csv
```

<p align="right"><a href="#top">back to top ↑</a></p>

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

## 📡 9 · Monitoring

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

<p align="right"><a href="#top">back to top ↑</a></p>

---

### Incidents and delayed labels

```python
from mlpilot.monitoring import pagerduty_alert, opsgenie_alert, join_labels, sql_sink
mon.on_alert(pagerduty_alert(routing_key))       # one incident per model and alert kind (deduplicated)
mon.on_check(sql_sink(ac.SQLSource(url)))        # every check result -> a warehouse table for BI
joined = join_labels("preds.jsonl", outcomes, on="order_id")   # labels that arrived days later
mon.check_labels(joined)                         # live accuracy of what was actually served
```

## 💡 10 · Insights & reporting

```python
for insight in ac.find_insights(sales_df, "sales"):
    print(insight)                     # e.g.
# Average sales rose about 50% from Jan 2024 to Dec 2024 (steady monthly trend, R² 1.00).
# sales is 19% lower when stock < 24 (average 104 vs 128 otherwise; 499 rows).

report = ac.insights_report(sales_df, "sales", title="Q4 sales review")      # offline template summary
report = ac.insights_report(sales_df, "sales", llm=True)                     # Claude writes the narrative
report.save("sales_review.html")                                             # also .md / .json
```

**What it finds:**
- segments that move the target up or down;
- trends and sudden changes over time;
- strong correlations;
- concentration, such as "20% of stores produce 80% of revenue";
- outliers and data-quality caveats;
- with `model=...`, the model's own top drivers.

Segment findings must pass a significance test corrected for multiple comparisons, and must cover
enough rows. Every finding carries the numbers behind it.

**The LLM narrative.** With `llm=True`, Claude (`claude-opus-5-5` by default) writes the executive
summary:
- It receives only the findings and their aggregate numbers, never raw rows.
- It is told not to introduce numbers that aren't in the findings, and to describe associations
  rather than causes.
- It uses `ANTHROPIC_API_KEY` or an `ant auth login` profile.
- Without credentials, or on any API error, the report falls back to the template summary with a
  warning.
- `llm=False` (the default) makes no network calls.

Install the SDK with `pip install "dataspoof-mlpilot[llm]"`.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🏛️ 11 · Hub: the platform server

A multi-user server for teams, installed with `pip install "dataspoof-mlpilot[server]"`:

```bash
mlpilot hub create-admin --name admin      # prints an API token once
mlpilot hub serve --host 0.0.0.0 --port 8080
# or: docker compose -f deploy/hub/docker-compose.yml up -d   (Hub + PostgreSQL)
```

```python
from mlpilot.hub import HubClient
hub = HubClient("https://hub.example.com", token=os.environ["AC_HUB_TOKEN"])
run = hub.start_run("churn", experiment="weekly", name="automl")
v = hub.register("churn", "churn-model", model, run_id=run["id"], metrics={"test_roc_auc": 0.91})
hub.promote("churn", "churn-model", v["version"], "production", reason="beats v3")   # -> pending approval
model = hub.load_model("churn", "churn-model", stage="production")
```

**Identity**
- **Single sign-on (OIDC):** works with Okta, Entra ID / Azure AD, Google, Keycloak and Auth0.
- **Group mapping:** identity-provider groups map to workspace roles, synced on every login.
- **API tokens** for service accounts. Tokens are stored only as hashes, can expire and can be
  revoked.

**Roles, per workspace**

| Role | Can |
|---|---|
| viewer | read |
| data_scientist | log runs, register models, request promotion |
| approver | approve or reject promotions |
| admin | manage members |

People outside a workspace can't tell that it exists.

**Governance**
- **Two-person rule:** promoting a model to production needs approval from someone other than
  the requester, with the evidence attached.
- **Policy gates:** each workspace can require minimum metrics or signed models.
- **Tamper-evident audit log:**
  - It records every action, including refused attempts.
  - Each entry is hash-chained to the one before it; `GET /api/audit/verify` checks the chain.
  - `GET /api/audit/export` exports JSONL for your security tools (SIEM).

**Storage:** metadata goes to PostgreSQL. Artifacts are immutable and checksummed, stored on
disk, S3, GCS or Azure. The server inspects uploads without ever unpickling them, and refuses
path-traversal ("zip-slip") archives.

**Operations**
- **Web UI** for models, approvals, runs, monitoring, members and the audit log.
- **OpenAPI docs** at `/docs`.
- **Deployment:**
  - a Helm chart (`deploy/helm`): non-root, read-only filesystem, health probes, pod
    disruption budget;
  - a Terraform module for AWS (`deploy/terraform/aws`): S3, RDS PostgreSQL, Secrets Manager,
    an IAM role scoped to the Hub's service account, and the Helm release.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 📈 12 · Forecasting

```python
fc = ac.Forecaster("sales", "date", horizon=14, group="store").fit(history)
print(fc.backtest_)        # rolling-origin backtest: MAE, RMSE, sMAPE, MASE vs seasonal-naive
fc.predict()               # store, date, step, forecast, lower, upper
ac.reconcile_bottom_up(fc.predict(), {"s1": ["north", "total"], "s2": ["north", "total"]},
                       group="store", time_col="date")      # forecasts that add up
```

- **Inputs:** the frequency and seasonality are inferred. It can forecast many series at once,
  and takes `known_future` inputs such as planned promotions.
- **Models:** it picks between a seasonal-naive baseline and gradient boosting on lags,
  rolling means and calendar features, whichever scores better in the backtest.
- **Intervals:** these come from the backtest errors at each step ahead. In testing, the 90%
  intervals covered 88% of later actuals.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🧪 13 · Causal inference & uplift

```python
est = ac.estimate_effect(df, treatment="got_discount", outcome="revenue")
print(est)     # effect +5.1 (95% CI +4.1 to +6.1), naive +13.5 - confounding changes the answer ...
up = ac.UpliftModel("got_offer", "converted").fit(campaign)
up.predict_uplift(customers)                     # who actually responds to the offer
ac.qini(up.predict_uplift(test), test["got_offer"], test["converted"])
```

**What `estimate_effect` does:** it estimates what a treatment, such as a discount, actually
*causes*, not just what it's associated with:
- **Method:** a doubly-robust estimate (AIPW) with cross-fitting.
- **Comparison:** it reports the naive difference next to the corrected estimate.
- **Diagnostics:** it checks overlap (whether treated and untreated customers are comparable)
  and covariate balance, and warns when the estimate shouldn't be trusted.

It states the assumption it rests on: every common cause of treatment and outcome is among the
confounders.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🚨 14 · Anomaly detection

```python
det = ac.AutoAnomaly().fit(transactions)
det.score(new)            # anomaly_score (0-1) and is_anomaly
det.explain(new)          # "amount = 9,800 is unusually high (8.9 robust SDs from typical 120)"
```

It combines three detectors, each catching a different kind of anomaly:
- **Isolation Forest:** rare combinations of values.
- **Local Outlier Factor:** points far from their neighbours.
- **Robust z-score:** extreme single values.

The detectors are combined on robustly standardised scores. The share of rows flagged is
chosen automatically, capped at 5%.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🔌 15 · Integrations & scale

| | |
|---|---|
| **Airflow** | `AutoMLTrainOperator`, `BatchScoreOperator`, `MonitorOperator` (`integrations.airflow`) |
| **Prefect / Dagster** | tasks and a lifecycle flow; software-defined assets |
| **MLflow** | export runs, models and registry stages to an MLflow server (see below) |
| **DVC** | version datasets in a git + DVC repository and read back any old revision (see below) |
| **Feast** | export the feature store as a ready-to-apply Feast repository (see below) |
| **Kafka** | `score_stream(model, KafkaSource(...), KafkaSink(...))`: offsets are committed only after delivery, so every record is processed at least once |
| **Spark / Ray / Dask** | `ac.scale.score_spark(model_path, sdf)`, `score_ray`, `score_dask` |
| **Files of any size** | `ac.transform_file(pipeline, "big.csv", "clean.parquet")` (memory-bounded chunks) |
| **SQL pushdown** | `ac.to_sql(pipeline, "raw.orders", dialect="snowflake")`: a fitted cleaning pipeline becomes one `SELECT`, so the warehouse does the work and the data never leaves it. In testing, results matched Python exactly. Any step that has no exact SQL equivalent raises an error. |

The orchestrator wrappers all call `integrations.tasks` (train & register, batch score, monitor
& retrain). Those are plain functions, so they also work from cron, Argo or Step Functions.

### Already running MLflow, DVC or Feast?

The built-in tracker, data repository and feature store need no servers. If your team already
runs one of these tools, these bridges move the same objects into it:

```python
from mlpilot.integrations import mlflow as acm
acm.export_tracker(tracker, tracking_uri="http://mlflow:5000")       # every run: params, metric history, artifacts
info = acm.log_automl(model, test, experiment="churn", register_as="churn")   # a pyfunc model with a signature
acm.export_registry(registry, "churn")                                # versions; stages become aliases
model = acm.load_automl("models:/churn@production")                   # back to an AutoML object
#   $ mlflow models serve -m models:/churn@production --env-manager local

from mlpilot.integrations.dvc import DVCRepo, export_repository
repo = DVCRepo.init("ml-data")                                        # git init + dvc init
repo.add_remote("storage", "s3://bucket/dvc")
repo.track(df, "data/customers.parquet", message="Q2 refresh", tag="customers-v2", push=True)
old = repo.read("data/customers.parquet", rev="customers-v1")         # any commit, tag or branch
export_repository(data_repo, "customers", repo)                       # DataRepository history -> git commits + tags

from mlpilot.integrations.feast import export_to_feast, feast_features
fs = export_to_feast(store, "feast_repo")             # feature_store.yaml + features.py, applied and materialized
fs.get_historical_features(entity_df, feast_features(store, ["customer_stats:*"])).to_df()
```

**MLflow** (`[mlflow]`):
- Models are logged as **pyfunc** models with a signature and an input example, so
  `mlflow.pyfunc.load_model` and `mlflow models serve` work on them directly.
- Exports are batched. Running an export again skips runs and versions already copied, so a
  scheduled sync is safe.

**DVC** (`[dvc]`): it calls the real `git` and `dvc` commands, so the result is an ordinary DVC
repository that `dvc pull`, `dvc diff` and CI already understand.
- Re-committing unchanged data creates no commit.
- A version that is missing from the local cache is fetched from the remote.
- `log_to_run` records in a run exactly which data version it used.

**Feast** (`[feast]`):
- The export writes `feature_store.yaml`, generated Entity / FileSource / FeatureView
  definitions (types, TTL, owner, tags) and each view's history.
- To use your production online store (Redis, DynamoDB, Postgres…), change one block of
  `feature_store.yaml`.
- An entity with several join keys becomes one Feast entity per key.

**Tested against the real tools, not stand-ins:**

| Tool | Setup | What the tests check |
|---|---|---|
| MLflow 3.18 | a real `mlflow server`, plus `mlflow models serve` answering HTTP | predictions over HTTP match predictions in Python |
| DVC 3.67 | git, plus a DVC remote | push, then delete the cache and pull, then read old revisions |
| Feast 0.67 | `apply` and `materialize` | point-in-time results and online lookups match ours value for value |

There is one behaviour difference. For entity rows that have no value inside the TTL window,
Feast's file offline store can leave the row out. `FeatureStore` keeps the row, with null
features.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🖼️ 16 · Image & audio data

```python
from mlpilot import media

scan = media.scan_images("photos/", label_from_folder=True)      # one row per file
print(media.image_quality_report(scan))   # corrupt, exact & near duplicates, tiny, dark, blurry, mixed modes, imbalance
X = media.ImagePreprocessor((224, 224), resize="pad").fit_transform(scan["path"])   # (n, 224, 224, 3), normalised
feats = media.image_features(scan["path"])                        # colour, edge-orientation, texture features
model = ac.AutoML("label").fit(feats.join(scan.select("path", "label"), on="path").drop("path"))

audio = media.scan_audio("calls/**/*.wav")
print(media.audio_quality_report(audio))  # corrupt, silent, clipped, too short, DC offset, mixed sample rates
clips = media.AudioPreprocessor(16000, duration=3.0).transform(audio["path"])   # mono, resampled, trimmed, levelled
feats = media.audio_features(audio["path"])                       # MFCCs + spectral features for AutoML
```

**Images** (`pip install "dataspoof-mlpilot[image]"`):
- Phone-photo rotation (EXIF) is fixed automatically.
- Near-duplicates are found by perceptual hashing, and blur is measured by the variance of the
  Laplacian.
- Preprocessing settings save as JSON, and augmentation is reproducible.
- In testing, features from synthetic image classes reached ≥ 90% accuracy with a linear model.

**Audio:**
- WAV is built in; FLAC, OGG and MP3 need the `[audio]` extra.
- Resampling, silence trimming, loudness normalisation and MFCCs are implemented here, so
  librosa isn't needed.
- In testing, audio features separated two classes of tones with 100% accuracy.

**Robustness:** a corrupt file is reported, never fatal to a batch.

**Hard conditions** (`[image]`; faces need `[vision]`, plates need `[ocr]`). See the
[guide](docs/guide/vision.md).

```python
from mlpilot.media import conditions, faces, plates, benchmark

conditions.detect_conditions(img).tags            # ['fog', 'low_light']: also rain, snow, glare, blur, noise, jpeg
per_image, coverage = conditions.condition_report("frames/")   # which conditions your data is missing
fixed, steps = conditions.enhance(img)            # dehaze / brighten / denoise / sharpen as needed
faces.face_occlusion(img)                         # coverage 0-1, likely='hair' | 'mask' | 'sunglasses' | 'object'
faces.add_occluder(img, "hair", severity=3)       # simulate coverings for training
plates.detect_plates(img, pattern="IN", enhance=True)   # [Plate(box, text='MH12AB1234', confidence=0.99)]
out, found = faces.anonymize(img)                 # blur every face and licence plate
rep = benchmark.condition_benchmark(predict, images, labels, faces=True)   # score per condition x severity
```

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 📐 17 · Statistical testing

```python
from mlpilot import stats

stats.compare_groups(df, value="revenue", group="region")
# -> one-way ANOVA (or Welch / Kruskal-Wallis), omega-squared, Tukey pairwise comparisons, plain English
stats.chi_square(df, "plan", "churned")            # Cramér's V; Fisher's exact for small 2x2 tables
stats.correlation_test(df, "price", "demand", method="spearman")   # with a 95% CI
stats.ab_test(df, variant="variant", metric="converted", control="control")   # lift, CI, p-value
stats.required_sample_size(baseline_rate=0.10, mde=0.02)          # -> 3,841 per variant
stats.test_all(df, target="churned")               # every column vs the target, FDR-corrected
```

**How the test is chosen:** `compare_groups` checks each group's normality and the equality
of variances (Levene) first, then chooses between Welch's t-test, Mann-Whitney, ANOVA,
Alexander-Govern and Kruskal-Wallis.

**What every result includes:**
- the test statistic and p-value;
- an effect size with a size label (negligible / small / medium / large);
- a confidence interval where one applies;
- the assumptions that were checked;
- a one-line interpretation.

The tests check the results against reference values (Fisher's exact test, Holm and BH
corrections, sample-size formulas).

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🗃️ 18 · Data versioning

```python
repo = ac.DataRepository("data_repo")
v1 = repo.commit(df, "customers", message="initial extract", source="s3://raw/customers.csv")
v2 = repo.commit(df2, "customers", message="Q2 refresh", tags=["gold"])
repo.log("customers")                                   # who, when, rows, hash, message
repo.diff("customers", 1, 2, key="customer_id")         # +added / -removed / ~changed rows, column changes
old = repo.checkout("customers", version=1)             # the exact data back
repo.checkout("customers", tag="gold", run=run)         # ...and record which version a run used
```

- **Storage:** content-addressed, zstd-compressed Parquet. Committing identical data again
  creates no new copy.
- **Integrity:** every checkout is verified against the stored file's SHA-256, so modified files
  are detected.
- **Tags** move between versions the way git tags do.
- **Lineage:** `commit(..., run=...)` and `checkout(..., run=...)` link dataset versions to
  experiment runs, in both directions.
- **Already using DVC?** `integrations.dvc` versions data in a git + DVC repository, with
  remotes, old revisions and lineage, and `export_repository` replays this history into it
  ([section 15](#-15--integrations--scale)).

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🏪 19 · Feature store

```python
fs = ac.FeatureStore("feature_repo")
fs.apply(ac.Entity("customer", ["customer_id"]),
         ac.FeatureView("customer_stats", entities=["customer"], timestamp="as_of",
                        features=["orders_30d", "spend_30d"], ttl_days=30, owner="growth"))
fs.ingest("customer_stats", daily_snapshot)                           # history + online latest
train = fs.get_historical_features(labels, ["customer_stats:*"])      # point-in-time correct
fs.get_online_features(["customer_stats:spend_30d"], [{"customer_id": 42}])   # serving lookup
```

**No leakage in training data:** each training row gets feature values *as they were at that
row's timestamp*, never later. Values older than `ttl_days` are dropped.

**Storage:**
- The **offline** store is append-only Parquet, holding the full history for training.
- The **online** store is SQLite, holding the latest value per entity for serving. Late-arriving
  old rows never overwrite newer values. `materialize()` rebuilds it as of any date.

**Safety:** changing a feature's type, or removing a feature that already has data, is refused.

**Moving to Feast?** `integrations.feast.export_to_feast(store, "feast_repo")` writes and applies
an equivalent Feast repository. In testing, Feast's point-in-time joins and online lookups
matched these ones ([section 15](#-15--integrations--scale)).

<p align="right"><a href="#top">back to top ↑</a></p>

---

## ✂️ 20 · Data splitting

```python
s = ac.split_data(df, target="churned", test_size=0.2, val_size=0.1)    # stratified for classification
s = ac.split_data(df, target="churned", group="customer_id")            # a customer is never on both sides
s = ac.split_data(df, target="sales", time="date", gap="7d")            # past -> train, future -> test
print(s.report())                       # sizes, class balance, time ranges, leakage checks
ac.check_split(my_train, my_test, group="patient_id", time="visit")     # audit a split made elsewhere
for train, test in ac.cv_folds(df, target="churned", n_splits=5, group="customer_id"): ...
```

**How the strategy is chosen:**
- A `time=` column gives a chronological split. `gap` drops the rows just before each boundary,
  so window features can't leak forward.
- A `group=` column keeps every row of a group in one split. With a classification target, the
  groups are also balanced by class.
- Otherwise the split is random, and stratified by the target for classification.

**Leakage checks:** every split is checked afterwards for
- groups appearing on both sides;
- test rows older than the newest training row;
- identical rows in train and test;
- test classes never seen in training.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🛡️ 21 · Model security

```python
rep = ac.robustness_report(model, test_df)      # any AutoML or scikit-learn model (target=...)
print(rep.summary())                            # noise stability, adversarial distance, sensitive features
rep.passed(max_noise_flip=0.05, min_median_distance=0.25)

pois = ac.poisoning_report(train_df, target="fraud")
print(pois.summary())                           # label errors, conflicting duplicates, triggers, outliers
clean = pois.clean(train_df)
```

**Robustness** uses black-box tests, so the model's own preprocessing is part of what is attacked:
- **Noise stability** at 1%, 5% and 10% of each feature's standard deviation.
- **Minimal adversarial change:** a greedy evasion attack measures, in standard deviations, how
  small a change flips each prediction, and which feature it exploits.
- **Per-feature sensitivity:** each feature is shifted, swapped or set to missing.
- **Extreme inputs** must not crash the model or produce NaN.

**Poisoning detection:**
- **Label errors:** confident learning on out-of-fold probabilities finds mislabelled rows.
- **Conflicts:** identical rows that carry different labels.
- **Backdoor triggers:** a rare value whose rows almost all share one label, tested with a
  Holm-corrected binomial test.
- **Outliers**, found with Isolation Forest.

**In testing:**
- At least 70% of deliberately flipped labels were found, with at least half of the flags correct.
- Both planted triggers were found: a categorical code and a repeated numeric value.
- Clean data produced no false triggers.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🚦 22 · Environments & CI/CD

```bash
mlpilot cicd init --provider github --name churn --data data/train.parquet --target churned
#  -> .github/workflows/model-churn.yml, gates.yml, environments.yml
mlpilot train --data data/train.parquet --target churned --out build/model --group customer_id
mlpilot gate  --model build/model --data build/test.parquet --gates gates.yml     # exit 1 = blocked
mlpilot register --model build/model --registry registry/dev --name churn --metrics build/metrics.json
mlpilot release --config environments.yml --name churn --version 7 --from qa --to prod \
                          --released-by ana --approved-by bob,carol
```

**Environments:**
- dev, QA and prod each have their **own registry**, and two environments may not share one.
- Models move forward only through `release`, one environment at a time, so nothing skips QA.
- The target environment's rules are enforced: metric floors, the number of approvers (never
  counting the person releasing), and a signature check against its public key.
- Every release, and every refusal, is recorded on both sides, so production always knows where
  a model came from.
- The registry gains a `qa` stage: `dev → qa → staging → production`.

**The pipeline** has five stages: train, gate, register, release to QA, then a release to
production approved through GitHub environments and a blue/green deploy.
- It is generated for GitHub Actions or GitLab CI.
- Every stage is one CLI command, so it runs the same way locally and in CI.
- **Gates** cover metric floors, fairness limits and robustness. A model that fails them never
  gets registered.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🔐 23 · Secrets & policy

```bash
export AC_API_KEY="vault://secret/ml/churn#api_key"          # resolved at start-up, never stored
export AC_HUB_DATABASE_URL="aws-sm://prod/hub#database_url"
export AC_HUB_OPA_URL="http://opa:8181"                        # policy as code for the Hub
```

**Secret references** work anywhere the library reads a credential: the model API key, the Hub's
database / OIDC / role settings, SQL and Snowflake connections, and PagerDuty and Opsgenie keys.
- **Vault** (KV v1 and v2) signs in with a token, AppRole or Kubernetes auth, and no extra
  package is needed.
- Also supported: AWS Secrets Manager, environment variables and mounted files.
- Values are cached, and error messages never contain them.

**Open Policy Agent** adds rules on top of the Hub's own roles. OPA can only take access away,
never grant it.
- The shipped Rego policies (`deploy/opa`, with 10 unit tests) enforce that:
  - nothing skips QA;
  - production needs a description and the organisation's metric floors;
  - the four-fifths fairness rule applies;
  - models must be signed;
  - service accounts can promote at most to QA and can never approve;
  - frozen workspaces are read-only;
  - restricted workspaces only accept identities from the listed e-mail domains.
- Refusals go into the tamper-evident audit log.
- If OPA is unreachable, requests are refused: the Hub fails closed.

**Tested against the real servers in Docker:**
- **Vault:** AppRole login with a read-only policy, refused paths, and a model server, SQL source
  and Hub configured purely by `vault://` references.
- **OPA:** the Hub enforced every rule, including a change freeze pushed to OPA while it was
  running.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 📟 24 · Observability

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

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🧾 25 · Lineage & catalogs

```python
from mlpilot.lineage import dataset
lin = ac.Lineage.from_env()                       # OPENLINEAGE_URL (Marquez, DataHub, Atlan...) and/or a file
with lin.run("train_churn", inputs=[dataset("warehouse.customers", train_df)],
             outputs=[dataset("models/churn/v7")]):
    model = ac.AutoML("churned").fit(train_df)    # START, then COMPLETE (or FAIL with the stack trace)
lin.emit_tracked_run(run)                          # existing tracker runs and dataset versions, too
ac.LineageGraph.from_file("lineage.jsonl").upstream("models/churn/v7")

from mlpilot.integrations import aws_glue
aws_glue.write_table(df, "analytics", "customers", "s3://lake/customers/", partition_by=["country"])
aws_glue.read_table("analytics", "customers"); aws_glue.run_job("nightly-etl", {"--date": "2026-11-08"})

from mlpilot.integrations.azure_data_factory import DataFactory
DataFactory(sub, "data-rg", "corp-adf").run_pipeline("CopyOrders", {"date": "2026-11-08"})
```

**What the events carry:** OpenLineage events include each dataset's schema, row and null counts,
and versions.

**What was tested against what:**
- **Marquez** (the OpenLineage reference server): real, in Docker. Jobs, run states, schemas and
  the lineage graph all appeared as expected.
- **Glue:** a moto server speaking the real S3 protocol. This covered Parquet, partitions, the
  catalog schema and job runs.
- **Data Factory:** a fake Azure Resource Manager endpoint, testing the REST contract.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## ☁️ 26 · Deploy on AWS

```python
from mlpilot.deploy import aws
uri = aws.push_image("churn:7", "ml/churn")                    # ECR, scan-on-push, immutable tags
aws.deploy_sagemaker(uri, "churn-prod", role_arn=ROLE, instance_count=2, autoscale=(2, 8),
                     environment={"AC_API_KEY": "aws-sm://prod/churn#api_key"},
                     data_capture_s3="s3://ml-capture/churn/", rollback_alarms=["churn-prod-5xx"])
```

```bash
terraform -chdir=deploy/terraform/aws-sagemaker apply -var name=churn -var image_uri=$URI
terraform -chdir=deploy/terraform/aws-ecs apply -var name=churn -var image_uri=$URI -var vpc_id=...
```

**SageMaker:**
- Blue/green updates, rolled back automatically if an alarm fires while the new fleet is baking.
- Serverless or instance-based variants, with autoscaling and request/response data capture.
- IAM with least privilege.

**ECS Fargate:** a load balancer with HTTPS, tasks in private subnets, the API key from Secrets
Manager, a deployment circuit breaker with rollback, and CPU plus request-count autoscaling.

**Tested:**
- Both Terraform modules pass `terraform validate` and `fmt` against AWS provider 6.x.
- The Python deploy path is tested against moto, with requests validated against the real AWS
  API schemas.

> Nothing here has been run against a real AWS account: that needs your account, credentials and
> budget. The modules and the boto3 code are ready to point at one.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🦀 Why Rust + Python?

The library is a **hybrid**: a friendly Python API on top, a fast Rust engine underneath.

- 🐍 **Python frontend** — the API you call, plus everything that leans on the Python
  ecosystem (Polars I/O, scikit-learn imputation, Plotly dashboards, Presidio PII, Claude AI).
- 🦀 **Rust backend** (`mlpilot._rustcore`) — the number-crunching and per-row
  string work: text preprocessing, symbol stripping, IQR outliers, skewness, correlation,
  multicollinearity and JSON detection.

You install one wheel and write normal Python — the Rust is invisible.

```
┌─────────────────────────── your code ───────────────────────────┐
│  import mlpilot as ac                                  │
│  df = ac.load_data("data.csv"); ac.clean_data(df)                │
└──────────────────────────────┬──────────────────────────────────┘
                               │ calls
┌──────────────────────────────▼──────────────────────────────────┐
│  Python frontend  (mlpilot/cleaning.py)                │
│    Polars • scikit-learn • Plotly • Presidio • LangChain         │
└──────────────────────────────┬──────────────────────────────────┘
                               │ delegates hot loops
┌──────────────────────────────▼──────────────────────────────────┐
│  🦀 Rust backend  (_rustcore, built with PyO3 + rayon)           │
│    text cleaning • skewness • IQR • correlation • JSON detect     │
└──────────────────────────────────────────────────────────────────┘
```

**Benchmark** — text preprocessing over 50,000 rows: **~3.8× faster** than the pure-Python
path, with the GIL released so it scales across cores.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🧾 Requirements

- **Python** ≥ 3.11.
- **Installed automatically (the lean core):**
  - data: `polars`, `pandas`, `numpy`, `pyarrow`
  - modelling: `scikit-learn`
  - validation and parsing: `rapidfuzz`, `phonenumbers`, `email-validator`, `python-dateutil`
  - configs and contracts: `pyyaml`
- **Optional extras** add everything else; see [Installation](#-installation).
  `pip install "dataspoof-mlpilot[all]"` installs them all.
- **Claude-powered features** are optional: AI categorical corrections and insight narratives.
  They use your `ANTHROPIC_API_KEY`, and without it fall back to offline behaviour.

---

## 🛠️ Building & contributing

The project uses **maturin** as its build backend (`pyproject.toml`). See
[BUILD.md](https://github.com/DataSpoof/dataspoof-data-cleaning/blob/main/BUILD.md) for the full architecture and build/publish guide.

```
Cargo.toml            # Rust crate config
src/
  lib.rs              # PyO3 module — functions exposed to Python
  text.rs             # text preprocessing (contractions, regex, stopwords)
  stats.rs            # skewness, IQR, correlation, multicollinearity
  clean.rs            # whitespace, Unicode, mojibake, booleans, placeholders
mlpilot/
  core/               # shared Step / Pipeline core (no imports from the rest of the package)
  ingestion/          # files, cloud, SQL, warehouses, APIs, incremental loads
  cleaners.py         # stateful cleaning steps + default_cleaning_pipeline()
  steps.py            # 43 standalone cleaning functions
  cleaning.py         # interactive clean_data() pipeline and EDA dashboard
  quality/            # profiling, quality score, contracts, schema checks
  features/           # feature engineering + leakage detection
  automl/             # AutoML + AutoCluster
  evaluation/         # metrics, SHAP, error analysis, fairness, model cards
  registry/           # experiment tracking + model registry
  deploy/             # scoring, REST API, export, Docker, CLI
  monitoring/         # drift, alerts, retraining
  insights/           # findings, executive reports, LLM narratives
tests/                # pytest suite
```

Run the tests (needs the Rust toolchain to build the extension):

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin
maturin develop --release --extras test
pytest
```

The GitHub Actions workflow (`.github/workflows/CI.yml`) builds Linux, macOS and Windows wheels.
Publishing them from CI needs PyPI Trusted Publishing to be set up for the repository.

---

## 📄 License

MIT © Abhishek Kumar Singh / [DataSpoof](https://github.com/DataSpoof)
