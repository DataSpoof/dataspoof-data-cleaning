"""Secrets from a secret manager instead of plain environment variables or config files.

Anywhere the library reads a credential (the model server's API key, the Hub's database URL and
OIDC settings, SQL / Snowflake connection strings, alert-routing keys) you can give a
*reference* instead of the value:

    vault://secret/ml/hub#database_url        HashiCorp Vault, KV v2 (mount "secret", path "ml/hub", key)
    vault://kv1:ml/hub#password               a KV v1 mount (``<mount>:`` prefix)
    aws-sm://prod/hub#database_url            AWS Secrets Manager (JSON secret + key, or the whole string)
    env://HUB_DB_URL                          another environment variable
    file:///run/secrets/hub_db_url            a mounted file (Docker / Kubernetes secrets)

>>> from mlpilot.secretstore import get_secret, resolve
>>> get_secret("vault://secret/ml/hub#database_url")
>>> resolve(os.environ.get("AC_API_KEY"))          # a reference is resolved; a plain value is returned as is

**Vault authentication** (first that applies): ``VAULT_TOKEN``; AppRole with ``VAULT_ROLE_ID`` +
``VAULT_SECRET_ID``; Kubernetes with ``VAULT_K8S_ROLE`` (service-account token from
``VAULT_K8S_TOKEN_PATH``, default ``/var/run/secrets/kubernetes.io/serviceaccount/token``).
Also ``VAULT_ADDR``, ``VAULT_NAMESPACE``, ``VAULT_CACERT``, ``VAULT_AUTH_MOUNT``. Login tokens are
renewed before they expire; secret values are cached for ``AC_SECRET_CACHE_SECONDS`` (default 300).
No extra package is needed for Vault (plain HTTPS); AWS needs ``boto3``.
"""

from __future__ import annotations

import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SCHEMES = ("vault://", "aws-sm://", "env://", "file://")
_cache: dict = {}
_lock = threading.Lock()


class SecretError(RuntimeError):
    """A secret reference could not be resolved (the message never contains the secret)."""


def is_reference(value) -> bool:
    return isinstance(value, str) and value.startswith(SCHEMES)


def resolve(value, *, default=None):
    """``value`` itself, or the secret it refers to. ``None`` -> ``default``."""
    if value is None:
        return default
    return get_secret(value) if is_reference(value) else value


def get_secret(ref: str, *, ttl: float | None = None) -> str:
    """Resolve one secret reference (see the module docstring for the formats)."""
    if not is_reference(ref):
        raise SecretError(f"Not a secret reference (expected one of {SCHEMES}): {ref.split('#')[0][:60]!r}")
    ttl = float(os.environ.get("AC_SECRET_CACHE_SECONDS", "300")) if ttl is None else ttl
    if ttl > 0:                                   # ttl=0: always fetch fresh (e.g. right after a rotation)
        with _lock:
            hit = _cache.get(ref)
            if hit and hit[1] > time.time():
                return hit[0]
    if ref.startswith("env://"):
        name = ref[6:]
        if name not in os.environ:
            raise SecretError(f"Environment variable {name} is not set")
        value = os.environ[name]
    elif ref.startswith("file://"):
        path = Path(urllib.parse.unquote(urllib.parse.urlparse(ref).path.lstrip("/") if os.name == "nt"
                                         else urllib.parse.urlparse(ref).path))
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SecretError(f"Can't read secret file {path}: {exc.strerror}") from None
    elif ref.startswith("vault://"):
        value = _vault().read(ref[8:])
    else:
        value = _aws(ref[9:])
    if ttl > 0:
        with _lock:
            _cache[ref] = (value, time.time() + ttl)
    return value


def clear_cache() -> None:
    with _lock:
        _cache.clear()
    global _client
    _client = None


def _split(spec: str) -> tuple:
    path, _, key = spec.partition("#")
    return path, (key or None)


def _aws(spec: str) -> str:
    try:
        import boto3
    except ImportError:
        raise SecretError("aws-sm:// references need boto3: pip install boto3") from None
    name, key = _split(spec)
    try:
        resp = boto3.client("secretsmanager").get_secret_value(SecretId=name)
    except Exception as exc:
        raise SecretError(f"AWS Secrets Manager: can't read {name!r} ({type(exc).__name__})") from None
    raw = resp.get("SecretString")
    if raw is None:
        raise SecretError(f"AWS secret {name!r} is binary; store it as a string")
    if key is None:
        return raw
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise SecretError(f"AWS secret {name!r} is not JSON, so it has no key {key!r}") from None
    if key not in data:
        raise SecretError(f"AWS secret {name!r} has no key {key!r}")
    return str(data[key])


# ── Vault ───────────────────────────────────────────────────────────────────
class VaultClient:
    """A minimal Vault HTTP client: token / AppRole / Kubernetes login, KV v1 + v2 read & write."""

    def __init__(self, addr: str | None = None, *, token: str | None = None, namespace: str | None = None,
                 cacert: str | None = None, timeout: float = 10.0):
        self.addr = (addr or os.environ.get("VAULT_ADDR") or "").rstrip("/")
        if not self.addr:
            raise SecretError("Set VAULT_ADDR (e.g. https://vault.example.com:8200)")
        self.namespace = namespace or os.environ.get("VAULT_NAMESPACE")
        self.timeout = timeout
        cafile = cacert or os.environ.get("VAULT_CACERT")
        self._ctx = ssl.create_default_context(cafile=cafile) if self.addr.startswith("https") else None
        self._token = token or os.environ.get("VAULT_TOKEN")
        self._expires = float("inf") if self._token else 0.0
        self._renewable = False
        self._lock = threading.Lock()

    def _call(self, method: str, path: str, body=None, *, auth: bool = True) -> dict:
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["X-Vault-Token"] = self.token()
        if self.namespace:
            headers["X-Vault-Namespace"] = self.namespace
        req = urllib.request.Request(f"{self.addr}/v1/{path.lstrip('/')}", method=method, headers=headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            try:
                errors = json.loads(exc.read()).get("errors") or []
            except Exception:
                errors = []
            msg = "; ".join(str(e) for e in errors)[:200] or exc.reason
            raise SecretError(f"Vault {method} {path.split('?')[0]}: HTTP {exc.code} {msg}") from None
        except urllib.error.URLError as exc:
            raise SecretError(f"Vault at {self.addr} is unreachable: {exc.reason}") from None

    # authentication
    def token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._expires - 30:
                return self._token
            self._login()
            return str(self._token)

    def _login(self):
        mount = os.environ.get("VAULT_AUTH_MOUNT")
        if os.environ.get("VAULT_ROLE_ID") and os.environ.get("VAULT_SECRET_ID"):
            res = self._call("POST", f"auth/{mount or 'approle'}/login",
                             {"role_id": os.environ["VAULT_ROLE_ID"], "secret_id": os.environ["VAULT_SECRET_ID"]},
                             auth=False)
        elif os.environ.get("VAULT_K8S_ROLE"):
            jwt_path = os.environ.get("VAULT_K8S_TOKEN_PATH", "/var/run/secrets/kubernetes.io/serviceaccount/token")
            try:
                jwt = Path(jwt_path).read_text().strip()
            except OSError:
                raise SecretError(f"Kubernetes service-account token not found at {jwt_path}") from None
            res = self._call("POST", f"auth/{mount or 'kubernetes'}/login",
                             {"role": os.environ["VAULT_K8S_ROLE"], "jwt": jwt}, auth=False)
        else:
            raise SecretError("No Vault credentials: set VAULT_TOKEN, VAULT_ROLE_ID + VAULT_SECRET_ID, or VAULT_K8S_ROLE")
        a = res.get("auth") or {}
        self._token = a.get("client_token")
        if not self._token:
            raise SecretError("Vault login returned no token")
        self._expires = time.time() + (a.get("lease_duration") or 3600)

    # KV
    def _kv(self, spec: str) -> tuple:
        path, key = _split(spec)
        if ":" in path.split("/")[0]:                    # "kv1:ml/hub" -> KV v1 mount "kv1"
            mount, rest = path.split(":", 1)
            return mount, rest.strip("/"), key, 1
        mount, _, rest = path.partition("/")
        if not rest:
            raise SecretError("Vault references look like vault://<mount>/<path>#<key>")
        return mount, rest.strip("/"), key, 2

    def read(self, spec: str) -> str:
        mount, path, key, version = self._kv(spec)
        res = self._call("GET", f"{mount}/data/{path}" if version == 2 else f"{mount}/{path}")
        data = (res.get("data") or {})
        data = data.get("data", {}) if version == 2 else data
        if key is None:
            if len(data) == 1:
                return str(next(iter(data.values())))
            raise SecretError(f"Vault secret {mount}/{path} has keys {sorted(data)}; add #<key>")
        if key not in data:
            raise SecretError(f"Vault secret {mount}/{path} has no key {key!r}")
        return str(data[key])

    def write(self, spec: str, values: dict) -> None:
        mount, path, _, version = self._kv(spec)
        self._call("POST", f"{mount}/data/{path}" if version == 2 else f"{mount}/{path}",
                   {"data": values} if version == 2 else values)


_client: VaultClient | None = None


def _vault() -> VaultClient:
    global _client
    if _client is None:
        _client = VaultClient()
    return _client
