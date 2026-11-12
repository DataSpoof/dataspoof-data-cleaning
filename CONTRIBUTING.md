# Contributing

Thanks for helping! The basics:

## Set up

```bash
# needs the Rust toolchain: https://rustup.rs
git clone https://github.com/DataSpoof/dataspoof-mlpilot.git
cd dataspoof-mlpilot
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install maturin
maturin develop --release --extras dev                  # builds the Rust extension in place
```

## Before you open a pull request

```bash
pytest                 # the whole suite; must pass on Polars 1.x and 2.x
mypy                   # type-checks the package (config in pyproject.toml)
mkdocs build --strict  # if you changed the docs
```

- **Tests:** add tests for every change. Tests must run offline; set `AC_OFFLINE=1` to check.
  Use `pytest.importorskip` for optional dependencies.
- **New pipeline steps:** subclass `mlpilot.Step` and register it with
  `@register_step`. Learn in `_fit` (return JSON-serialisable state) and replay in
  `_transform`. Steps that drop rows must implement `row_mask()`.
- **`core/` stays independent:** it must not import from the rest of the package. A test
  enforces this.
- **Imports stay light:** don't import optional dependencies at module import time; import
  them where they're used, and raise an `ImportError` that names the extra.
- **Persistence:** nothing new may be pickled except trained estimators, which are checksummed.
- **Changelog:** add an entry to `CHANGELOG.md` under the unreleased version.

## Releases (maintainers)

1. Bump the version in `pyproject.toml` and `Cargo.toml`, and date the changelog entry.
2. Tag `vX.Y.Z` and push the tag. CI tests, builds and attests the wheels for every platform,
   then publishes them to PyPI with Trusted Publishing.
