# Build & Architecture

This library uses a **Rust compute backend** with a **Python frontend**, packaged
as a single `pip`-installable wheel via [maturin](https://www.maturin.rs/) + [PyO3](https://pyo3.rs/).

## Layout

```
Cargo.toml                     # Rust crate (cdylib -> _rustcore)
pyproject.toml                 # maturin build backend + Python deps
src/
  lib.rs                       # PyO3 module: exposes functions to Python
  text.rs                      # text preprocessing (contractions, regex, stopwords)
  stats.rs                     # skewness, IQR, correlation, multicollinearity
automatedcleaning/
  __init__.py                  # from .cleaning import *
  cleaning.py                  # Python frontend (I/O, ML, dashboard, PII, AI)
  _rustcore.*.pyd              # compiled Rust extension (built artifact)
```

## What runs where

| Concern | Location | Why |
|---|---|---|
| Text cleaning, symbol stripping, IQR outliers, skewness, correlation / multicollinearity, negatives, JSON detection | **Rust** (`_rustcore`) | CPU-bound, parallelised with `rayon`, GIL released |
| WordNet lemmatization | Python | preserves exact NLTK output |
| CSV/Parquet I/O, KNN imputation, class balancing | Python (Polars / scikit-learn) | mature, no benefit to reimplementing |
| Plotly dashboard, Presidio PII masking, Claude spell-correction | Python | ecosystem-only |

The Python functions delegate the heavy loops to `_rustcore`; results are verified
to match Polars to machine precision (skewness, IQR bounds, correlation drop-list).

## Build

Prereqs: Rust toolchain (`rustup`) and `maturin` (`pip install maturin`).

```bash
# Build a release wheel (dist-ready, abi3 -> works on CPython 3.11+)
maturin build --release
# -> target/wheels/AutomatedCleaning-2.0.0-cp311-abi3-win_amd64.whl

# Install it
pip install target/wheels/AutomatedCleaning-2.0.0-cp311-abi3-win_amd64.whl

# --- OR, for development (requires an active virtualenv) ---
python -m venv .venv && .venv/Scripts/activate   # Windows
maturin develop --release
```

## Publish to PyPI

`maturin build --release` produces the wheel; upload with `twine upload target/wheels/*`
(or `maturin publish`). For cross-platform wheels, build in CI with
`PyO3/maturin-action`.
