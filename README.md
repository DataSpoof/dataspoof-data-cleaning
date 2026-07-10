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
automatedcleaning/    # Python frontend package
```

Cross-platform wheels are built and published to PyPI automatically via GitHub Actions
(`.github/workflows/CI.yml`) on every tagged release (`vX.Y.Z`).

---

## License

MIT © Abhishek Kumar Singh / [DataSpoof](https://github.com/DataSpoof)
