# Secrets & model signing: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Secrets & model signing guide](../guide/secrets-policy.md).

**On this page:**

- [`mlpilot.secretstore`](#mlpilotsecretstore): `SecretError`, `is_reference`, `resolve`, `get_secret`, `clear_cache`, `VaultClient`
- [`mlpilot.security`](#mlpilotsecurity): `SignatureError`, `generate_keypair`, `sign_model`, `verify_model`, `verify_if_required`

---

## `mlpilot.secretstore`

<sub>Source: [`mlpilot/secretstore.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/secretstore.py)</sub>

Secrets from a secret manager instead of plain environment variables or config files.

Anywhere the library reads a credential (the model server's API key, the Hub's database URL and
OIDC settings, SQL / Snowflake connection strings, alert-routing keys) you can give a
*reference* instead of the value:

    vault://secret/ml/hub#database_url        HashiCorp Vault, KV v2 (mount "secret", path "ml/hub", key)
  vault://kv1:ml/hub#password               a KV v1 mount (`<mount>:` prefix)
  aws-sm://prod/hub#database_url            AWS Secrets Manager (JSON secret + key, or the whole string)
  env://HUB_DB_URL                          another environment variable
  file:///run/secrets/hub_db_url            a mounted file (Docker / Kubernetes secrets)

```python
from mlpilot.secretstore import get_secret, resolve
get_secret("vault://secret/ml/hub#database_url")
resolve(os.environ.get("AC_API_KEY"))          # a reference is resolved; a plain value is returned as is
```

**Vault authentication** (first that applies): `VAULT_TOKEN`; AppRole with `VAULT_ROLE_ID` +
`VAULT_SECRET_ID`; Kubernetes with `VAULT_K8S_ROLE` (service-account token from
`VAULT_K8S_TOKEN_PATH`, default `/var/run/secrets/kubernetes.io/serviceaccount/token`).
Also `VAULT_ADDR`, `VAULT_NAMESPACE`, `VAULT_CACERT`, `VAULT_AUTH_MOUNT`. Login tokens are
renewed before they expire; secret values are cached for `AC_SECRET_CACHE_SECONDS` (default 300).
No extra package is needed for Vault (plain HTTPS); AWS needs `boto3`.

### class `SecretError`

```python
class SecretError(RuntimeError):
```

A secret reference could not be resolved (the message never contains the secret).

### `is_reference()`

```python
def is_reference(value) -> bool
```

### `resolve()`

```python
def resolve(value, *, default=None)
```

`value` itself, or the secret it refers to. `None` -> `default`.

### `get_secret()`

```python
def get_secret(ref: str, *, ttl: float | None=None) -> str
```

Resolve one secret reference (see the module docstring for the formats).

### `clear_cache()`

```python
def clear_cache() -> None
```

### class `VaultClient`

```python
class VaultClient:
```

A minimal Vault HTTP client: token / AppRole / Kubernetes login, KV v1 + v2 read & write.

**Constructor**

```python
VaultClient(addr: str | None=None, *, token: str | None=None, namespace: str | None=None, cacert: str | None=None, timeout: float=10.0)
```

#### `VaultClient.token()`

```python
token() -> str
```

#### `VaultClient.read()`

```python
read(spec: str) -> str
```

#### `VaultClient.write()`

```python
write(spec: str, values: dict) -> None
```

---

## `mlpilot.security`

<sub>Source: [`mlpilot/security.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/security.py)</sub>

Model signing: prove a saved model came from your training pipeline and wasn't altered.

A saved model directory is signed with an Ed25519 private key (kept by the training job / CI);
deployments verify it with the public key **before** anything is unpickled:
```
from mlpilot.security import generate_keypair, sign_model, verify_model
generate_keypair("keys/model_signing.pem", "keys/model_signing.pub")
sign_model("models/churn", "keys/model_signing.pem")          # writes models/churn/signature.json
verify_model("models/churn", "keys/model_signing.pub")        # raises SignatureError if tampered
```

Set `AC_MODEL_PUBLIC_KEY` (a path or PEM text) and every `load_model` — the REST API, batch
scoring, the registry — refuses models that aren't signed by the matching key.
Needs the `cryptography` package.

### class `SignatureError`

```python
class SignatureError(Exception):
```

The model is unsigned, signed by another key, or was modified after signing.

### `generate_keypair()`

```python
def generate_keypair(private_path, public_path, *, password: str | None=None) -> tuple
```

Write a new Ed25519 key pair as PEM files (the private key optionally encrypted).

### `sign_model()`

```python
def sign_model(path, private_key, *, password: str | None=None, signer: str | None=None) -> Path
```

Sign every file in a saved model directory; writes `signature.json` next to them.

### `verify_model()`

```python
def verify_model(path, public_key) -> dict
```

Check `signature.json` against `public_key` and every file's hash. Returns the
signature metadata; raises `SignatureError` on any problem (missing, wrong key,
modified, added or removed files).

### `verify_if_required()`

```python
def verify_if_required(path, public_key=None) -> dict | None
```

Verify when a key is given or `AC_MODEL_PUBLIC_KEY` is set (used by `load_model`).
