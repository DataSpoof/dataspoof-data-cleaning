<a id="top"></a>

<div align="center">

<img src="https://raw.githubusercontent.com/DataSpoof/dataspoof-data-cleaning/main/images/logo.png" alt="AutomatedCleaning logo" width="260">

# AutomatedCleaning

### From raw data to a monitored production model — automated, explainable, auditable.

[![PyPI](https://img.shields.io/pypi/v/AutomatedCleaning?color=2f6fde&label=PyPI)](https://pypi.org/project/AutomatedCleaning/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Rust engine](https://img.shields.io/badge/engine-Rust-B7410E?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Polars](https://img.shields.io/badge/dataframes-Polars-CD792C?logo=polars&logoColor=white)](https://pola.rs/)
[![Tests](https://img.shields.io/badge/tests-183%20passing-2ea44f)](#%EF%B8%8F-building--contributing)
[![License: MIT](https://img.shields.io/badge/license-MIT-2ea44f)](#-license)

**[Install](#-installation)** ·
**[Quick start](#-quick-start)** ·
**[Modules](#-the-platform-at-a-glance)** ·
**[Contents](#-contents)** ·
**[Why Rust?](#-why-rust--python)**

</div>

---

AutomatedCleaning automates the whole data-science lifecycle in one Python library. It covers
ingestion and cleaning, data quality, feature engineering, AutoML, explainability and fairness,
the model registry, deployment, monitoring, and plain-English insights. A **Rust engine**
handles the heavy computation, and every stage shares one `fit` / `transform` / `save` contract.

## ⚡ Quick start

```bash
pip install AutomatedCleaning
```

```python
import automatedcleaning as ac

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
| 🚀 | **Deployment** | `deploy` | Batch scoring, a REST API, ONNX and pickle export, **Docker images**, and a CLI. |
| 📡 | **Monitoring** | `monitoring` | Data and prediction drift, accuracy decay, alerts and webhooks, and champion-vs-challenger **retraining**. |
| 💡 | **Insights & reporting** | `insights` | Significance-tested findings, executive reports, and an optional narrative written by **Claude**. |

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

## 📚 Contents

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

Also: [Installation](#-installation) · [Why Rust + Python?](#-why-rust--python) ·
[Requirements](#-requirements) · [Building & contributing](#%EF%B8%8F-building--contributing) ·
[License](#-license)

---

## 📦 Installation

```bash
pip install AutomatedCleaning                    # core library
pip install "AutomatedCleaning[sql,excel]"       # + any extras you need (see below)
```

The import name is `automatedcleaning` (`import automatedcleaning as ac`). It needs
**Python 3.11+**.

> **Platform note:** PyPI currently has a prebuilt wheel for **Windows**. On Linux and macOS,
> including Colab, `pip` builds from source and needs the Rust toolchain
> (`curl https://sh.rustup.rs -sSf | sh`). The included GitHub Actions workflow builds wheels
> for every platform.

| Extra | Adds | For |
|---|---|---|
| `sql` | SQLAlchemy (plus your database's driver) | PostgreSQL, MySQL, SQL Server, Oracle, Redshift… |
| `excel` | fastexcel | `.xlsx` / `.xls` files |
| `snowflake` / `bigquery` | native warehouse connectors | Snowflake, BigQuery |
| `automl` | LightGBM, XGBoost | extra AutoML model families |
| `embeddings` | sentence-transformers | neural text embeddings |
| `explain` | shap | shap's own plots from `Explanation.to_shap()` |
| `onnx` | skl2onnx, onnxruntime | ONNX export |
| `serve` | gunicorn / waitress | production REST serving |
| `llm` | anthropic | Claude-written executive narratives |
| `text` | langdetect, better-profanity, pyspellchecker | language / profanity / spelling steps |

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
  `pip install "AutomatedCleaning[sql]"` plus your database's driver.
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
import automatedcleaning as ac

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
import automatedcleaning as ac

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
import automatedcleaning as ac

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
<summary><b>All additional cleaning steps (<code>automatedcleaning.steps</code>)</b></summary>


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
pip install "AutomatedCleaning[text]"
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

### Clustering

```python
clusters = ac.AutoCluster(k_range=(2, 8)).fit(df)     # k-means and Gaussian mixtures, chosen by silhouette
clusters.labels_                                      # one label per input row
clusters.cluster_profiles(df)                         # size, share and averages per cluster
ac.automl(df)                                         # no target -> AutoCluster; with target= -> AutoML
```

Optional extras: `pip install "AutomatedCleaning[automl]"` (LightGBM, XGBoost) and
`pip install "AutomatedCleaning[embeddings]"` (sentence-transformers).

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
exp.to_shap()                            # a shap.Explanation for shap's plots: pip install "AutomatedCleaning[explain]"
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

In production, run `gunicorn automatedcleaning.deploy.wsgi:app` with `AC_MODEL_PATH` set. Use
waitress on Windows, and install either with `pip install "AutomatedCleaning[serve]"`.

**Docker.** `build_docker` writes a multi-stage Dockerfile: a non-root user, a health check,
gunicorn, and the model baked in. The generated image has been built and tested end to end.

**ONNX** (`pip install "AutomatedCleaning[onnx]"`) exports the trained estimator: linear models,
random forests, extra trees and voting ensembles. Pass `sample=df` and the export is checked
against the original model. HistGradientBoosting can't be exported yet because of an upstream
skl2onnx limitation; the error says so.

**The CLI** (also available as `python -m automatedcleaning.deploy`):

```bash
automatedcleaning serve  --model models/titanic --port 8000
automatedcleaning score  --model models/titanic --input big.csv --output scored.parquet
automatedcleaning --registry ac_registry docker --model titanic:production --out build/
automatedcleaning --registry ac_registry monitor --model titanic --monitor monitor.json --input today.csv
```

<p align="right"><a href="#top">back to top ↑</a></p>

---

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

Install the SDK with `pip install "AutomatedCleaning[llm]"`.

<p align="right"><a href="#top">back to top ↑</a></p>

---

## 🦀 Why Rust + Python?

The library is a **hybrid**: a friendly Python API on top, a fast Rust engine underneath.

- 🐍 **Python frontend** — the API you call, plus everything that leans on the Python
  ecosystem (Polars I/O, scikit-learn imputation, Plotly dashboards, Presidio PII, Claude AI).
- 🦀 **Rust backend** (`automatedcleaning._rustcore`) — the number-crunching and per-row
  string work: text preprocessing, symbol stripping, IQR outliers, skewness, correlation,
  multicollinearity and JSON detection.

You install one wheel and write normal Python — the Rust is invisible.

```
┌─────────────────────────── your code ───────────────────────────┐
│  import automatedcleaning as ac                                  │
│  df = ac.load_data("data.csv"); ac.clean_data(df)                │
└──────────────────────────────┬──────────────────────────────────┘
                               │ calls
┌──────────────────────────────▼──────────────────────────────────┐
│  Python frontend  (automatedcleaning/cleaning.py)                │
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

- **Python** ≥ 3.11
- These are installed automatically:
  - data: `polars`, `pandas`, `numpy`, `pyarrow`
  - modelling: `scikit-learn`, `nltk`
  - charts: `plotly`, `matplotlib`, `missingno`, `pyfiglet`
  - PII masking: `presidio-analyzer`, `presidio-anonymizer`
  - validation and parsing: `rapidfuzz`, `phonenumbers`, `email-validator`, `python-dateutil`,
    `pyyaml`
  - LLM features: `langchain-anthropic`
- Claude-powered features are optional: interactive categorical spell-correction and insight
  narratives. They use your `ANTHROPIC_API_KEY`. Without it, both fall back to offline behaviour.

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
automatedcleaning/
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
