# Security policy

## Supported versions

| Version | Security fixes |
|---|---|
| 2.x (latest minor) | ✅ |
| 1.14.x | ✅ until 2027-05-08 (six months after 2.0) |
| < 1.14 | ❌ |

## Reporting a vulnerability

Please **don't open a public issue**. Report it privately in either of two ways:
- through GitHub's
  [private vulnerability reporting](https://github.com/DataSpoof/dataspoof-data-cleaning/security/advisories/new);
- by email to **dataspoof007@gmail.com**, with "SECURITY" in the subject.

Include the affected version, how to reproduce the problem, and its impact. You'll get:
- an acknowledgement within 3 working days;
- an assessment within 10;
- a fix and advisory, with credit if you want it, for confirmed issues. Critical issues are
  usually fixed within 30 days.

## How releases are protected

- **Signed releases:** wheels are published with PyPI Trusted Publishing (OIDC; no long-lived
  tokens) and carry signed build-provenance attestations. To check one, run
  `gh attestation verify <file> --repo DataSpoof/dataspoof-data-cleaning`.
- **Software bill of materials:** every GitHub release includes a CycloneDX SBOM.
- **Dependency audits:** every build checks Python dependencies with `pip-audit` and Rust
  dependencies with `cargo audit`.

## Security properties of the library

- **No unsafe pipeline files:** saved pipelines, data contracts, monitors and registry metadata
  are plain JSON, never pickle.
- **Checked model files:** trained models (joblib) are verified against a SHA-256 checksum
  before loading. Unpickling still runs code, so only load models from sources you trust, or
  export to ONNX.
- **SQL:** values are always bound as parameters, and identifiers are quoted.
- **Credentials:** passwords and tokens are redacted from reprs, logs and reports.
- **REST API:**
  - optional `X-API-Key` authentication, checked in constant time;
  - request-size limits;
  - internal errors never return details to the caller.
- **API connector:** it never follows a next-page link to a different host, so tokens can't
  leak there.
- **Docker:** generated images run as a non-root user.
- **Privacy:** drift monitors store binned statistics, not rows. LLM features send only
  aggregated findings, never raw data.
