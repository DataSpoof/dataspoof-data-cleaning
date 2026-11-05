# AutomatedCleaning

**AutomatedCleaning** is a Python library for automated data cleaning, now powered by a
**Rust compute backend**. It preprocesses and analyzes datasets — handling missing values,
outliers, spelling corrections, text cleaning, PII masking and more — while the CPU-heavy
work runs in compiled Rust for speed and true (GIL-free) parallelism.

![Logo](https://raw.githubusercontent.com/DataSpoof/dataspoof-data-cleaning/main/images/logo.png)

---

## Why Rust + Python?

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

---

## Features

- Supports both large (100+ GB) and small datasets
- Detects and handles missing values and duplicate records
- Identifies and corrects spelling errors in categorical values
- Detects and removes outliers (IQR)
- Detects and fixes data imbalance
- Identifies and corrects skewness in numerical data
- Checks for correlation and detects multicollinearity
- Analyzes cardinality in categorical columns
- Identifies and cleans text columns
- Detects JSON-type columns
- Detects and masks PII columns
- Performs univariate, bivariate, and multivariate analysis (interactive dashboard)

---

## Installation

### From PyPI (recommended)

```bash
pip install AutomatedCleaning
```

Prebuilt **abi3 wheels** are published for Linux, macOS and Windows and work on
**CPython 3.11+** — no Rust toolchain needed to install. The import name is also
`automatedcleaning` (`import automatedcleaning as ac`).

### From source

Building from source requires the **Rust toolchain** (because the backend is compiled):

```bash
# 1. Install Rust (https://rustup.rs) and maturin
pip install maturin

# 2. Clone and build
git clone https://github.com/DataSpoof/dataspoof-data-cleaning.git
cd dataspoof-data-cleaning

# Build & install into the current environment
pip install .

# --- or, for development (needs an active virtualenv) ---
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
maturin develop --release
```

---

## Quick start

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

---

## Pipelines: fit once, apply anywhere (recommended for production)

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

---

## Using individual steps

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

### Text preprocessing on its own

```python
ac.preprocess_text("I can't wait!! Visit https://x.co @bob 😀 it's GR8")
# -> 'cannot wait visit'
```

## Additional cleaning steps (`automatedcleaning.steps`)

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

---

## Supported file formats

`load_data()` reads `.csv`, `.tsv`, `.json`, and `.parquet` (Excel must be converted first).

---

## Requirements

- **Python** ≥ 3.11
- Key runtime deps (installed automatically): `polars`, `pandas`, `numpy`, `pyarrow`,
  `nltk`, `scikit-learn`, `matplotlib`, `seaborn`, `missingno`, `plotly`, `pyfiglet`,
  `langchain-anthropic`, `presidio-analyzer`, `presidio-anonymizer`
- The automatic categorical spell-correction step optionally uses the **Anthropic Claude API**
  (you'll be prompted for a key); you can skip it or correct values manually.

---

## Building & contributing

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
  core/               # shared Step/Pipeline core (no imports from the rest of the package)
  cleaners.py         # stateful cleaning steps + default_cleaning_pipeline()
  steps.py            # 43 standalone cleaning functions
  cleaning.py         # interactive clean_data() pipeline and EDA dashboard
tests/                # pytest suite
```

Run the tests (needs the Rust toolchain to build the extension):

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin
maturin develop --release --extras test
pytest
```

Cross-platform wheels are built and published to PyPI automatically via GitHub Actions
(`.github/workflows/CI.yml`) on every tagged release (`vX.Y.Z`).

---

## License

MIT © Abhishek Kumar Singh / [DataSpoof](https://github.com/DataSpoof)
