# Installation


```bash
pip install AutomatedCleaning                    # lean core (pipelines, quality, features, AutoML, …)
pip install "AutomatedCleaning[sql,excel]"       # + the extras you need (see below)
pip install "AutomatedCleaning[all]"             # everything, like 1.x installed by default
```

Since 2.0, the core install no longer pulls in the heavy dependencies (NLTK, Plotly, Presidio,
LangChain). Those features ask for their extra when you use them. Importing the library never
downloads data, writes files or changes your logging. Set `AC_OFFLINE=1` to guarantee no
network use at all.

The import name is `automatedcleaning` (`import automatedcleaning as ac`). It needs
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

<details>
<summary><b>Install from source</b></summary>

```bash
pip install maturin                               # needs Rust: https://rustup.rs
git clone https://github.com/DataSpoof/dataspoof-data-cleaning.git
cd dataspoof-data-cleaning
pip install .                                     # or: maturin develop --release (in a virtualenv)
```

</details>
