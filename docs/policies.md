# Versioning, deprecation & support policy

## Versioning

AutomatedCleaning follows [Semantic Versioning](https://semver.org/):

| Change | Version bump |
|---|---|
| Incompatible API change, removed feature, or a dependency moved out of the core install | **major** (2.0 → 3.0) |
| New features that are backwards compatible | **minor** (2.0 → 2.1) |
| Bug and security fixes | **patch** (2.0.0 → 2.0.1) |

**The public API** is everything exported from `automatedcleaning` and the documented module
namespaces. Names that start with `_` are private and can change at any time.

**Saved artifacts** (pipelines, contracts, monitors, registry metadata, model bundles) carry a
format version. A release can always load artifacts written by earlier releases of the same
major version.

## Deprecation

1. A feature is first marked deprecated: it issues a `DeprecationWarning` and the changelog
   names its replacement.
2. It keeps working for **at least two minor releases**, and at least 6 months.
3. It is removed only in the next major release.

## Support

| Release | Status |
|---|---|
| Latest minor of the current major | bug and security fixes |
| Previous major (latest minor) | security fixes for 6 months after the next major ships |
| Python | every CPython version that is still upstream-supported, from 3.11 |
| Polars | 1.x and 2.x, both tested in CI |

## Platforms

Wheels are published for Linux (x86-64, ARM64), macOS (Intel, Apple Silicon) and Windows
(x86-64) once Trusted Publishing is enabled for the repository. Any other platform builds from
the source package and needs the Rust toolchain.
