# Environments & CI/CD: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Environments & CI/CD guide](../guide/environments-cicd.md).

**On this page:**

- [`mlpilot.cicd`](#mlpilotcicd): `train`, `load_gates`, `gate`, `register`, `init_cicd`
- [`mlpilot.environments`](#mlpilotenvironments): `ReleaseError`, `Environment`, `Environments`

---

## `mlpilot.cicd`

<sub>Source: [`mlpilot/cicd.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/cicd.py)</sub>

Model CI/CD: the steps a pipeline runs, and ready-made pipelines that run them.

`mlpilot cicd init --provider github --name churn --data data/train.parquet --target churned`
writes a pipeline for *your* model repository:

    train  ->  gate  ->  register (dev)  ->  release to QA  ->  [approval]  ->  release to prod  ->  deploy

Each stage is one CLI command (so it runs the same on a laptop, in GitHub Actions, GitLab CI,
Jenkins or Azure DevOps):

* `mlpilot train`   split (leak-checked), poisoning check, AutoML fit, test metrics, model card
* `mlpilot gate`    fails the build unless the model meets `gates.yml`: metric floors /
  ceilings, fairness limits, robustness to noise and adversarial changes
* `mlpilot register` / `promote` / `release`   registry + environment promotion
  (`environments.yml`: isolated dev / qa / prod, approvals, signatures)
* `mlpilot deploy-sagemaker`   blue/green endpoint update (or use the Terraform modules)

`gates.yml`:
```
metrics:    {min: {roc_auc: 0.75}, max: {}}
fairness:   {sensitive: [gender], limits: {disparate_impact_ratio: 0.8}}
robustness: {max_noise_flip: 0.05, min_median_adversarial_distance: 0.1, rows: 200}
data:       {max_suspicious_share: 0.05}
quantization: {mode: int8, max_drop: 0.01, min_agreement: 0.99, required: false}
```

`quantization` checks that the model can ship quantized (`mlpilot.deploy.quantize`): the
`mode` variant must lose at most `max_drop` of the gate metric and agree with the original
model on `min_agreement` of rows. Models with nothing to quantize (tree ensembles, linear
models) pass with a note, unless `required: true`.

### `train()`

```python
def train(data, target: str, out, *, test=None, test_size: float=0.2, group: str | None=None, time: str | None=None, time_budget: float=300, include: list | None=None, seed: int=0, report=None, max_suspicious_share: float | None=None) -> dict
```

Split `data` (unless `test` is given), check the training data for poisoning / label
errors, fit AutoML, evaluate on the test split, and write `out` (the model), `out`/../
`test.parquet`, the model card and a JSON report.

### `load_gates()`

```python
def load_gates(path) -> dict
```

### `gate()`

```python
def gate(model, data, gates: dict, *, report=None) -> dict
```

Check a model against `gates` on held-out `data`; `result["passed"]` and the
`failures` list say why not.

### `register()`

```python
def register(model, registry, name: str, *, metrics=None, stage: str='dev', description: str | None=None, sign_key=None, tags: dict | None=None) -> dict
```

### `init_cicd()`

```python
def init_cicd(provider: str, *, name: str, data: str, target: str, out='.', metric: str='roc_auc', floor: float=0.7, budget: int=300, install: str | None=None, overwrite: bool=False) -> list
```

Write a pipeline (`github` or `gitlab`) plus `gates.yml` and `environments.yml`
into `out`. Existing files are kept unless `overwrite`. Returns the paths written.

---

## `mlpilot.environments`

<sub>Source: [`mlpilot/environments.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/environments.py)</sub>

Isolated dev -> qa -> prod environments with controlled promotion between them.

Each environment has its **own registry** (a separate folder, bucket mount or account), so a
mistake in dev cannot touch production. Models move forward only through `release`,
which copies the exact artifact and enforces the target environment's rules:

* only to the **next** environment in `order` (dev -> qa -> prod; no skipping);
* the source version must be in one of the source's `release_stages` (e.g. it passed QA);
* the artifact's checksum is verified, and its **signature** when the target requires it;
* the target's **metric gates** (`min_metrics` / `max_metrics`);
* `approvals` distinct approvers, none of them the person releasing (four-eyes);
* every release is recorded on both sides (`releases.jsonl`) with who, why, approvals and
  the source version, so production always knows where a model came from.

```python
envs = Environments.from_yaml("environments.yml")
envs.release("churn", 4, frm="qa", to="prod", released_by="ana", approved_by=["bob", "carol"],
             reason="Q3 refresh, QA sign-off #812")
envs.status("churn")          # which version is where, with provenance
```

`environments.yml`:
```
order: [dev, qa, prod]
environments:
  dev:  {registry: /mnt/ml/dev-registry}
  qa:   {registry: /mnt/ml/qa-registry, landing_stage: qa, min_metrics: {test_roc_auc: 0.70}}
  prod: {registry: /mnt/ml/prod-registry, landing_stage: staging, approvals: 2,
         require_signature: true, public_key: "vault://secret/ml/signing#public_key",
         min_metrics: {test_roc_auc: 0.75}, release_stages: [production]}
```

Values may be secret references (`vault://`, `aws-sm://`, `env://`, `file://`).

### class `ReleaseError`

```python
class ReleaseError(PromotionError):
```

A release between environments was refused; nothing was copied.

### class `Environment`

```python
class Environment:
```

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `registry` | `ModelRegistry` | required |
| `landing_stage` | `str` | `'staging'` |
| `release_stages` | `tuple` | `('qa', 'staging', 'production')` |
| `min_metrics` | `dict` | `dict()` |
| `max_metrics` | `dict` | `dict()` |
| `approvals` | `int` | `0` |
| `require_signature` | `bool` | `False` |
| `public_key` | `str \| None` | `None` |

#### `Environment.from_dict()` *(classmethod)*

```python
from_dict(name: str, d: dict) -> 'Environment'
```

### class `Environments`

```python
class Environments:
```

**Constructor**

```python
Environments(environments: dict, order: list | None=None)
```

#### `Environments.from_yaml()` *(classmethod)*

```python
from_yaml(path) -> 'Environments'
```

#### `Environments.next_of()`

```python
next_of(name: str) -> str | None
```

#### `Environments.check()`

```python
check(name: str, version: int, *, frm: str, to: str, released_by: str | None=None, approved_by=(), allow_skip: bool=False) -> list
```

Everything that would block `release` (an empty list means it would pass).

#### `Environments.release()`

```python
release(name: str, version: int, *, frm: str, to: str, released_by: str | None=None, approved_by=(), reason: str | None=None, allow_skip: bool=False)
```

Copy `name` v`version` from environment `frm` into `to` (see the module docs).
Returns the new `ModelVersion` in `to`.

#### `Environments.promote_path()`

```python
promote_path(name: str, version: int, *, released_by: str, approvals: dict | None=None, reason: str | None=None) -> list
```

Release a version step by step from its current environment to the last one, stopping at
the first refusal (raised). `approvals`: `{env: [approvers]}`. Returns each new version.

#### `Environments.releases()`

```python
releases(env: str | None=None) -> pl.DataFrame
```

#### `Environments.status()`

```python
status(name: str) -> pl.DataFrame
```

Every version of `name` in every environment, with its stage and where it came from.
