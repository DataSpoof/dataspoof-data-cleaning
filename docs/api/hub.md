# Hub (platform server): API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Hub (platform server) guide](../guide/hub.md).

**On this page:**

- [`mlpilot.hub`](#mlpilothub)
- [`mlpilot.hub.app`](#mlpilothubapp): `WorkspaceIn`, `MemberIn`, `TokenIn`, `ServiceAccountIn`, `RunIn`, `RunUpdate`, `TransitionIn`, `DecisionIn`, `MonitorIn`, `create_app`, `bootstrap_admin`
- [`mlpilot.hub.client`](#mlpilothubclient): `HubError`, `HubClient`
- [`mlpilot.hub.db`](#mlpilothubdb): `utcnow`, `new_id`, `Base`, `Workspace`, `User`, `Membership`, `ApiToken`, `Run`, `ModelVersion`, `PromotionRequest`, `MonitorResult`, `AuditEntry`, `make_engine`, `init_db`
- [`mlpilot.hub.policy`](#mlpilothubpolicy): `PolicyUnavailable`, `OPAPolicy`
- [`mlpilot.hub.security`](#mlpilothubsecurity): `AuthError`, `hash_token`, `create_token`, `user_from_api_token`, `OIDCVerifier`, `user_from_claims`, `role_in`, `require_role`, `audit`, `verify_audit_chain`
- [`mlpilot.hub.storage`](#mlpilothubstorage): `tar_directory`, `untar_to`, `sha256`, `ArtifactStore`

---

## `mlpilot.hub`

<sub>Source: [`mlpilot/hub/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/hub/__init__.py)</sub>

MLPilot Hub: the multi-user platform (server + client).

* `create_app()`     FastAPI server: workspaces, RBAC (viewer / data_scientist / approver / admin),
  SSO via OIDC + API tokens, runs, model registry with artifact storage
  (disk / S3 / GCS / Azure), four-eyes production approvals with policy gates,
  monitoring results, a hash-chained audit log, and a web UI.
* `HubClient`        Python client (standard library only).
* CLI: `mlpilot hub serve | create-admin`.

The server needs `pip install "dataspoof-mlpilot[server]"`; the client needs nothing extra.

---

## `mlpilot.hub.app`

<sub>Source: [`mlpilot/hub/app.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/hub/app.py)</sub>

MLPilot Hub: the multi-user platform server (FastAPI).

Run: `mlpilot hub serve` (or `uvicorn mlpilot.hub.app:app`). Configuration
(environment variables):

* `AC_HUB_DATABASE_URL`   `postgresql+psycopg://…` in production (default: SQLite file `hub.db`)
* `AC_HUB_ARTIFACTS`      artifact root: a directory or `s3://` / `gs://` / `az://` URL
* `AC_OIDC_ISSUER` / `AC_OIDC_AUDIENCE` / `AC_OIDC_JWKS_URL`   single sign-on (OIDC)
* `AC_HUB_GROUP_ROLES`    JSON: identity-provider group -> ["workspace", "role"]
* `AC_HUB_MAX_ARTIFACT_MB` upload limit (default 500)
* `AC_MODEL_PUBLIC_KEY`   if set, uploaded models must carry a valid signature
* `AC_HUB_OPA_URL`        Open Policy Agent for access and promotion decisions (see hub.policy)

Any of these may be a secret reference (`vault://secret/hub#database_url`, `aws-sm://...`,
`file:///run/secrets/...`) instead of the value; see `mlpilot.secretstore`.

API: `/api/...` (OpenAPI docs at `/docs`); web UI at `/`.

### class `WorkspaceIn`

```python
class WorkspaceIn(BaseModel):
```

### class `MemberIn`

```python
class MemberIn(BaseModel):
```

### class `TokenIn`

```python
class TokenIn(BaseModel):
```

### class `ServiceAccountIn`

```python
class ServiceAccountIn(BaseModel):
```

### class `RunIn`

```python
class RunIn(BaseModel):
```

### class `RunUpdate`

```python
class RunUpdate(BaseModel):
```

### class `TransitionIn`

```python
class TransitionIn(BaseModel):
```

### class `DecisionIn`

```python
class DecisionIn(BaseModel):
```

### class `MonitorIn`

```python
class MonitorIn(BaseModel):
```

### `create_app()`

```python
def create_app(*, database_url: str | None=None, artifact_root: str | None=None, oidc=None, group_roles: dict | None=None, max_artifact_mb: int | None=None, opa=None) -> FastAPI
```

`opa`: an `OPAPolicy` (default: from `AC_HUB_OPA_URL`).

### `bootstrap_admin()`

```python
def bootstrap_admin(database_url: str | None=None, name: str='admin', email: str | None=None) -> str
```

Create a platform admin (if missing) and return a new API token for it (printed once).

---

## `mlpilot.hub.client`

<sub>Source: [`mlpilot/hub/client.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/hub/client.py)</sub>

Python client for the Hub (standard library only).

```python
hub = HubClient("https://hub.example.com", token=os.environ["AC_HUB_TOKEN"])
run = hub.start_run("churn", experiment="weekly", name="automl")
hub.log(run, params={"time_budget": 60}, metrics={"test_roc_auc": 0.91})
v = hub.register("churn", "churn-model", model, run_id=run["id"], metrics={"test_roc_auc": 0.91})
hub.promote("churn", "churn-model", v["version"], "production", reason="beats v3")  # -> pending approval
model = hub.load_model("churn", "churn-model", stage="production")
```

### class `HubError`

```python
class HubError(Exception):
```

**Constructor**

```python
HubError(status: int, detail)
```

### class `HubClient`

```python
class HubClient:
```

**Constructor**

```python
HubClient(url: str | None=None, token: str | None=None, *, timeout: float=60.0)
```

#### `HubClient.me()`

```python
me() -> dict
```

#### `HubClient.start_run()`

```python
start_run(workspace: str, *, experiment: str, name: str, params: dict | None=None, tags: dict | None=None) -> dict
```

#### `HubClient.log()`

```python
log(run, *, params=None, metrics=None, tags=None, datasets=None, status=None) -> dict
```

#### `HubClient.end_run()`

```python
end_run(run, status: str='finished') -> dict
```

#### `HubClient.register()`

```python
register(workspace: str, name: str, model, *, run_id: str | None=None, metrics: dict | None=None, description: str | None=None) -> dict
```

Upload a fitted model (or a saved model directory) as a new version (stage `dev`).

#### `HubClient.versions()`

```python
versions(workspace: str, name: str) -> list
```

#### `HubClient.promote()`

```python
promote(workspace: str, name: str, version: int, stage: str, *, reason: str | None=None, evidence: dict | None=None) -> dict
```

#### `HubClient.pending_approvals()`

```python
pending_approvals(workspace: str) -> list
```

#### `HubClient.approve()`

```python
approve(request_id: str, note: str | None=None) -> dict
```

#### `HubClient.reject()`

```python
reject(request_id: str, note: str | None=None) -> dict
```

#### `HubClient.download()`

```python
download(workspace: str, name: str, *, version: int | None=None, stage: str='production', dest=None) -> Path
```

Download and safely extract a model version; returns its directory.

#### `HubClient.load_model()`

```python
load_model(workspace: str, name: str, *, version: int | None=None, stage: str='production')
```

#### `HubClient.report_monitoring()`

```python
report_monitoring(workspace: str, name: str, report, version: int | None=None) -> dict
```

Send a `Monitor.check` result to the Hub (`Monitor.on_check(lambda r: hub.report_monitoring(...))`).

---

## `mlpilot.hub.db`

<sub>Source: [`mlpilot/hub/db.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/hub/db.py)</sub>

Hub metadata database (SQLAlchemy 2.0): PostgreSQL in production, SQLite for development.

### `utcnow()`

```python
def utcnow() -> _dt.datetime
```

### `new_id()`

```python
def new_id() -> str
```

### class `Base`

```python
class Base(DeclarativeBase):
```

### class `Workspace`

```python
class Workspace(Base):
```

### class `User`

```python
class User(Base):
```

### class `Membership`

```python
class Membership(Base):
```

### class `ApiToken`

```python
class ApiToken(Base):
```

### class `Run`

```python
class Run(Base):
```

### class `ModelVersion`

```python
class ModelVersion(Base):
```

### class `PromotionRequest`

```python
class PromotionRequest(Base):
```

### class `MonitorResult`

```python
class MonitorResult(Base):
```

### class `AuditEntry`

```python
class AuditEntry(Base):
```

### `make_engine()`

```python
def make_engine(url: str)
```

### `init_db()`

```python
def init_db(url: str)
```

Create tables (idempotent) and return a session factory.

---

## `mlpilot.hub.policy`

<sub>Source: [`mlpilot/hub/policy.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/hub/policy.py)</sub>

Policy as code for the Hub with Open Policy Agent (OPA).

The Hub's own roles (viewer < data_scientist < approver < admin) are always enforced. With OPA
configured, every authorised request is *also* sent to OPA, which can deny it (deny-overrides),
and every promotion is checked against OPA's promotion rules in addition to the workspace
policy. Policies live in Rego files you version and review like code; `deploy/opa` has a
ready-to-use set with tests (`opa test deploy/opa/policies`).

Configuration: `AC_HUB_OPA_URL` (e.g. `http://opa:8181`), optional `AC_HUB_OPA_TOKEN`
(bearer token if OPA runs with authentication), `AC_HUB_OPA_PACKAGE` (default
`mlpilot`) and `AC_HUB_OPA_FAIL_OPEN=1` to allow requests when OPA is unreachable
(default: fail closed - the request is refused with 503).

Queried documents:

* `data.<package>.authz.allow` (bool) and `data.<package>.authz.reasons` (list of strings).
  Input: `user` (id, email, is_admin, is_service), `workspace`, `role`,
  `required_role`, `request` (method, path).
* `data.<package>.promotion.deny` (set of messages). Input: `workspace` (name, policy,
  require_approval), `model` (name, version, stage, metrics, signed, task, description,
  stage_history), `to_stage`, `user`.

### class `PolicyUnavailable`

```python
class PolicyUnavailable(RuntimeError):
```

OPA could not be reached (or answered with an error) and the policy fails closed.

### class `OPAPolicy`

```python
class OPAPolicy:
```

**Constructor**

```python
OPAPolicy(url: str, *, package: str='mlpilot', token: str | None=None, timeout: float=2.0, fail_open: bool=False)
```

#### `OPAPolicy.from_env()` *(classmethod)*

```python
from_env() -> 'OPAPolicy | None'
```

#### `OPAPolicy.query()`

```python
query(document: str, input_: dict)
```

The value of `data.<package>.<document>` for `input_` (`None` if undefined).

#### `OPAPolicy.authorize()`

```python
authorize(input_: dict) -> tuple
```

`(allowed, reasons)`. When OPA is unreachable: `fail_open` decides.

#### `OPAPolicy.promotion_denials()`

```python
promotion_denials(input_: dict) -> list
```

---

## `mlpilot.hub.security`

<sub>Source: [`mlpilot/hub/security.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/hub/security.py)</sub>

Hub authentication (OIDC bearer tokens + API tokens), RBAC and the hash-chained audit log.

### class `AuthError`

```python
class AuthError(Exception):
```

**Constructor**

```python
AuthError(status: int, message: str)
```

### `hash_token()`

```python
def hash_token(token: str) -> str
```

### `create_token()`

```python
def create_token(session, user: User, name: str, days: int | None=90) -> str
```

Create an API token; returns the token (shown once — only its hash is stored).

### `user_from_api_token()`

```python
def user_from_api_token(session, token: str) -> User
```

### class `OIDCVerifier`

```python
class OIDCVerifier:
```

Validates OIDC access/ID tokens (JWT) against the issuer's published keys (JWKS).

Configure with `AC_OIDC_ISSUER` and `AC_OIDC_AUDIENCE` (and optionally `AC_OIDC_JWKS_URL`;
by default it is discovered from `<issuer>/.well-known/openid-configuration`).

**Constructor**

```python
OIDCVerifier(issuer: str, audience: str, jwks_url: str | None=None, *, leeway: int=30)
```

#### `OIDCVerifier.claims()`

```python
claims(token: str) -> dict
```

#### `OIDCVerifier.from_env()` *(classmethod)*

```python
from_env()
```

### `user_from_claims()`

```python
def user_from_claims(session, claims: dict, *, group_roles: dict | None=None, group_claim: str='groups') -> User
```

Find or create (just-in-time provisioning) the user for verified OIDC claims.

`group_roles` maps identity-provider groups to workspace roles, e.g.
`{"ml-admins": ("*", "admin"), "churn-team": ("churn", "data_scientist")}` — memberships
are synced on every login, so access follows the identity provider.

### `role_in()`

```python
def role_in(session, user: User, workspace: Workspace) -> str | None
```

### `require_role()`

```python
def require_role(session, user: User, workspace: Workspace, minimum: str) -> str
```

### `audit()`

```python
def audit(session, actor: str, action: str, *, workspace: str | None=None, target: str | None=None, details: dict | None=None, ip: str | None=None) -> AuditEntry
```

### `verify_audit_chain()`

```python
def verify_audit_chain(session) -> dict
```

Recompute every hash. Any edited, deleted or reordered entry breaks the chain.

---

## `mlpilot.hub.storage`

<sub>Source: [`mlpilot/hub/storage.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/hub/storage.py)</sub>

Artifact storage for the Hub: a local/NFS directory, or object storage via fsspec
(`s3://bucket/prefix` with s3fs, `gs://` with gcsfs, `az://` / `abfs://` with adlfs).

Artifacts are stored as immutable `.tar.gz` blobs keyed by workspace / model / version, each
with its SHA-256 recorded in the database and re-checked on download.

### `tar_directory()`

```python
def tar_directory(path) -> bytes
```

A model directory as a deterministic .tar.gz (sorted entries, no owner/time metadata).

### `untar_to()`

```python
def untar_to(blob: bytes, dest) -> Path
```

Extract an artifact safely: only regular files, no absolute paths, no '..' (zip-slip).

### `sha256()`

```python
def sha256(data: bytes) -> str
```

### class `ArtifactStore`

```python
class ArtifactStore:
```

`ArtifactStore("/var/lib/hub/artifacts")` or `ArtifactStore("s3://ml-artifacts/hub")`.

**Constructor**

```python
ArtifactStore(root: str)
```

#### `ArtifactStore.put()`

```python
put(blob: bytes, *parts) -> str
```

#### `ArtifactStore.get()`

```python
get(uri: str) -> bytes
```
