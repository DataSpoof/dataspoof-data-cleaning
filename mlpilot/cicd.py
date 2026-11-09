"""Model CI/CD: the steps a pipeline runs, and ready-made pipelines that run them.

``mlpilot cicd init --provider github --name churn --data data/train.parquet --target churned``
writes a pipeline for *your* model repository:

    train  ->  gate  ->  register (dev)  ->  release to QA  ->  [approval]  ->  release to prod  ->  deploy

Each stage is one CLI command (so it runs the same on a laptop, in GitHub Actions, GitLab CI,
Jenkins or Azure DevOps):

* ``mlpilot train``   split (leak-checked), poisoning check, AutoML fit, test metrics, model card
* ``mlpilot gate``    fails the build unless the model meets ``gates.yml``: metric floors /
                                ceilings, fairness limits, robustness to noise and adversarial changes
* ``mlpilot register`` / ``promote`` / ``release``   registry + environment promotion
                                (``environments.yml``: isolated dev / qa / prod, approvals, signatures)
* ``mlpilot deploy-sagemaker``   blue/green endpoint update (or use the Terraform modules)

``gates.yml``::

    metrics:    {min: {roc_auc: 0.75}, max: {}}
    fairness:   {sensitive: [gender], limits: {disparate_impact_ratio: 0.8}}
    robustness: {max_noise_flip: 0.05, min_median_adversarial_distance: 0.1, rows: 200}
    data:       {max_suspicious_share: 0.05}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from .ingestion import read as _read


def _frame(path) -> pl.DataFrame:
    df = _read(str(path))
    return df.collect() if isinstance(df, pl.LazyFrame) else df


def _write_json(path, obj) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ── steps ───────────────────────────────────────────────────────────────────
def train(data, target: str, out, *, test=None, test_size: float = 0.2, group: str | None = None,
          time: str | None = None, time_budget: float = 300, include: list | None = None, seed: int = 0,
          report=None, max_suspicious_share: float | None = None) -> dict:
    """Split ``data`` (unless ``test`` is given), check the training data for poisoning / label
    errors, fit AutoML, evaluate on the test split, and write ``out`` (the model), ``out``/../
    ``test.parquet``, the model card and a JSON report."""
    from .automl import AutoML
    from .evaluation import evaluate_model
    from .poisoning import poisoning_report
    from .splitting import split_data
    df = _frame(data)
    if test is None:
        split = split_data(df, target, test_size=test_size, group=group, time=time, seed=seed)
        train_df, test_df, issues = split.train, split.test, split.issues
        strategy = split.strategy
    else:
        train_df, test_df = df, _frame(test)
        from .splitting import check_split
        issues, strategy = check_split(train_df, test_df, group=group, time=time, target=target), "given"
    pois = poisoning_report(train_df, target, seed=seed)
    share = pois.suspicious.height / max(train_df.height, 1)
    if max_suspicious_share is not None and share > max_suspicious_share:
        raise SystemExit(f"Training data looks poisoned or mislabelled: {share:.1%} suspicious rows "
                         f"(limit {max_suspicious_share:.1%}).\n{pois.summary()}")
    kw: dict[str, Any] = {"time_budget": time_budget, "seed": seed, "verbose": False}
    if include:
        kw["include"] = include
    model = AutoML(target, **kw).fit(train_df)
    out = Path(out)
    model.save(out)
    test_path = out.parent / "test.parquet"
    test_df.write_parquet(test_path)
    metrics = {f"test_{k}": v for k, v in evaluate_model(model, test_df, bootstrap=0).metrics.items()
               if isinstance(v, (int, float)) and v == v}
    from .automl.automl import _metric_label, _shown
    metrics[f"cv_{_metric_label(model.metric_)}"] = _shown(model.metric_, model.best_["score"])
    try:
        model.model_card(test_df).save(out.parent / "model_card.md")
    except Exception as exc:                          # a card is documentation; never fail the build on it
        print(f"model card skipped: {exc}")
    result = {"model": str(out), "test_data": str(test_path), "rows": {"train": train_df.height, "test": test_df.height},
              "split": strategy, "split_issues": issues, "algorithm": model.best_["model"], "metrics": metrics,
              "suspicious_training_rows": pois.suspicious.height, "suspicious_share": round(share, 4)}
    _write_json(report, result)
    (out.parent / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return result


def load_gates(path) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def gate(model, data, gates: dict, *, report=None) -> dict:
    """Check a model against ``gates`` on held-out ``data``; ``result["passed"]`` and the
    ``failures`` list say why not."""
    from .automl import load_model
    from .evaluation import evaluate_model
    m = load_model(model) if isinstance(model, (str, Path)) else model
    df = _frame(data) if isinstance(data, (str, Path)) else data
    failures: list = []
    checks: dict[str, Any] = {}
    metrics = evaluate_model(m, df, bootstrap=0).metrics
    checks["metrics"] = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    for k, lo in ((gates.get("metrics") or {}).get("min") or {}).items():
        v = metrics.get(k)
        if v is None or v < lo:
            failures.append(f"{k} = {v} (needs >= {lo})")
    for k, hi in ((gates.get("metrics") or {}).get("max") or {}).items():
        v = metrics.get(k)
        if v is None or v > hi:
            failures.append(f"{k} = {v} (needs <= {hi})")
    fair = gates.get("fairness") or {}
    if fair.get("sensitive"):
        from .evaluation import fairness_report
        fr = fairness_report(m, df, fair["sensitive"], limits=fair.get("limits"))
        checks["fairness"] = {"passed": fr.passed, "flags": [str(f) for f in fr.flags]}
        failures += [f"fairness: {f}" for f in fr.flags]
    rob = gates.get("robustness") or {}
    if rob:
        from .robustness import robustness_report
        rr = robustness_report(m, df, n=int(rob.get("rows", 200)))
        checks["robustness"] = {"noise_flip_5pct": rr.noise.get(0.05), "median_adversarial_distance":
                                rr.attack.get("median_distance"), "extreme": rr.extreme}
        if rob.get("max_noise_flip") is not None and rr.noise.get(0.05, 0) > rob["max_noise_flip"]:
            failures.append(f"robustness: 5% noise changes {rr.noise[0.05]:.1%} of predictions "
                            f"(limit {rob['max_noise_flip']:.1%})")
        med = rr.attack.get("median_distance")
        if rob.get("min_median_adversarial_distance") is not None and med is not None \
                and med < rob["min_median_adversarial_distance"]:
            failures.append(f"robustness: median adversarial change {med:.3g} sd "
                            f"(needs >= {rob['min_median_adversarial_distance']})")
        if rr.extreme["crashed"] or rr.extreme["non_finite"]:
            failures.append(f"robustness: extreme inputs break the model ({rr.extreme})")
    result = {"passed": not failures, "failures": failures, "checks": checks}
    _write_json(report, result)
    return result


def register(model, registry, name: str, *, metrics=None, stage: str = "dev", description: str | None = None,
             sign_key=None, tags: dict | None = None) -> dict:
    from .registry import ModelRegistry
    m = json.loads(Path(metrics).read_text()) if isinstance(metrics, (str, Path)) else (metrics or {})
    mv = ModelRegistry(registry).register(str(model), name, stage=stage, metrics=m, description=description,
                                          tags=tags, sign_with=sign_key)
    return {"name": name, "version": mv.version, "stage": mv.stage, "model_sha256": mv.model_sha256}


# ── pipeline templates ──────────────────────────────────────────────────────
GATES_TEMPLATE = """# Quality gates checked by `mlpilot gate` before a model may be registered.
metrics:
  min: {{{metric}: {floor}}}
  max: {{}}
# fairness:
#   sensitive: [gender, age]
#   limits: {{disparate_impact_ratio: 0.8}}
robustness:
  max_noise_flip: 0.05                    # share of predictions changed by 5%-of-sd noise
  min_median_adversarial_distance: 0.05   # in standard deviations
  rows: 200
"""

ENVIRONMENTS_TEMPLATE = """# Isolated environments; models move dev -> qa -> prod only through `mlpilot release`.
order: [dev, qa, prod]
environments:
  dev:  {{registry: registry/dev}}
  qa:   {{registry: registry/qa, landing_stage: qa}}
  prod:
    registry: registry/prod
    landing_stage: staging
    approvals: 0          # the CI environment's required reviewers approve; raise for CLI-only flows
    min_metrics: {{test_{metric}: {floor}}}
"""

GITHUB_TEMPLATE = """# Model pipeline generated by `mlpilot cicd init` - edit freely.
# Required reviewers: configure GitHub environments "qa" and "production" (Settings -> Environments).
# Registries live in S3 (synced around each job) so every job and run sees the same history.
name: model-{name}

on:
  push:
    branches: [main]
    paths: ["{data_glob}", "gates.yml", "environments.yml", ".github/workflows/model-{name}.yml"]
  workflow_dispatch:

permissions:
  contents: read
  id-token: write          # OIDC to AWS - no long-lived keys

concurrency:
  group: model-{name}
  cancel-in-progress: false

env:
  MODEL_NAME: {name}
  REGISTRY_S3: ${{{{ vars.REGISTRY_S3 }}}}          # e.g. s3://ml-registries/{name}
  AWS_REGION: ${{{{ vars.AWS_REGION }}}}
  INSTALL: "{install}"

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install "$INSTALL"
      - name: Train (leak-checked split, poisoning check, AutoML, model card)
        run: mlpilot train --data {data} --target {target} --out build/model --time-budget {budget} --report build/train.json
      - name: Quality gates
        run: mlpilot gate --model build/model --data build/test.parquet --gates gates.yml --report build/gate.json
      - uses: actions/upload-artifact@v4
        with:
          name: model
          path: build/

  register:
    needs: train
    runs-on: ubuntu-latest
    environment: dev
    outputs:
      version: ${{{{ steps.reg.outputs.version }}}}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "$INSTALL"
      - uses: actions/download-artifact@v4
        with:
          name: model
          path: build/
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{{{ vars.AWS_ROLE_ARN }}}}
          aws-region: ${{{{ vars.AWS_REGION }}}}
      - run: aws s3 sync "$REGISTRY_S3/dev" registry/dev
      - id: reg
        run: |
          mlpilot register --model build/model --registry registry/dev --name "$MODEL_NAME" --metrics build/metrics.json --stage staging --report build/register.json
          echo "version=$(python -c 'import json; print(json.load(open("build/register.json"))["version"])')" >> "$GITHUB_OUTPUT"
      - run: aws s3 sync registry/dev "$REGISTRY_S3/dev"

  release-qa:
    needs: register
    runs-on: ubuntu-latest
    environment: qa
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "$INSTALL"
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{{{ vars.AWS_ROLE_ARN }}}}
          aws-region: ${{{{ vars.AWS_REGION }}}}
      - run: for e in dev qa; do aws s3 sync "$REGISTRY_S3/$e" "registry/$e"; done
      - run: mlpilot release --config environments.yml --name "$MODEL_NAME" --version "${{{{ needs.register.outputs.version }}}}" --from dev --to qa --released-by "$GITHUB_ACTOR" --reason "run $GITHUB_RUN_ID"
      - run: aws s3 sync registry/qa "$REGISTRY_S3/qa"

  deploy-production:
    needs: release-qa
    runs-on: ubuntu-latest
    environment: production         # required reviewers approve here
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "$INSTALL"
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{{{ vars.AWS_ROLE_ARN }}}}
          aws-region: ${{{{ vars.AWS_REGION }}}}
      - run: for e in qa prod; do aws s3 sync "$REGISTRY_S3/$e" "registry/$e"; done
      - name: Release QA -> prod
        run: |
          QA_VERSION=$(mlpilot --registry registry/qa latest --name "$MODEL_NAME" --stage qa)
          mlpilot release --config environments.yml --name "$MODEL_NAME" --version "$QA_VERSION" --from qa --to prod --released-by "$GITHUB_ACTOR" --approved-by "${{{{ github.triggering_actor }}}}" --reason "run $GITHUB_RUN_ID" --report build/release.json
      - run: aws s3 sync registry/prod "$REGISTRY_S3/prod"
      - name: Build and push the serving image
        run: |
          mlpilot --registry registry/prod docker --model "$MODEL_NAME:staging" --out build/image --name "$MODEL_NAME" --target sagemaker
          docker build -t "$MODEL_NAME:$GITHUB_RUN_ID" build/image
      - name: Deploy (blue/green)
        run: mlpilot deploy-sagemaker --image "$MODEL_NAME:$GITHUB_RUN_ID" --repository "ml/$MODEL_NAME" --endpoint "$MODEL_NAME-prod" --role-arn "${{{{ vars.SAGEMAKER_ROLE_ARN }}}}"
"""

GITLAB_TEMPLATE = """# Model pipeline generated by `mlpilot cicd init` - edit freely.
# Protected environments "qa" / "production" give the approvals; registries sync with S3.
stages: [train, gate, register, release-qa, deploy]

variables:
  MODEL_NAME: {name}
  INSTALL: "{install}"
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip"

default:
  image: python:3.12-slim
  cache:
    paths: [.pip]
  before_script:
    - pip install "$INSTALL" awscli

train:
  stage: train
  script:
    - mlpilot train --data {data} --target {target} --out build/model --time-budget {budget} --report build/train.json
  artifacts:
    paths: [build/]

gate:
  stage: gate
  needs: [train]
  script:
    - mlpilot gate --model build/model --data build/test.parquet --gates gates.yml --report build/gate.json
  artifacts:
    paths: [build/]

register:
  stage: register
  needs: [gate]
  environment: {{name: dev}}
  rules: [{{if: '$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH'}}]
  script:
    - aws s3 sync "$REGISTRY_S3/dev" registry/dev
    - mlpilot register --model build/model --registry registry/dev --name "$MODEL_NAME" --metrics build/metrics.json --stage staging --report build/register.json
    - aws s3 sync registry/dev "$REGISTRY_S3/dev"
    - echo "MODEL_VERSION=$(python -c 'import json; print(json.load(open("build/register.json"))["version"])')" > build/version.env
  artifacts:
    reports:
      dotenv: build/version.env

release-qa:
  stage: release-qa
  needs: [register]
  environment: {{name: qa}}
  rules: [{{if: '$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH'}}]
  script:
    - for e in dev qa; do aws s3 sync "$REGISTRY_S3/$e" "registry/$e"; done
    - mlpilot release --config environments.yml --name "$MODEL_NAME" --version "$MODEL_VERSION" --from dev --to qa --released-by "$GITLAB_USER_LOGIN" --reason "pipeline $CI_PIPELINE_ID"
    - aws s3 sync registry/qa "$REGISTRY_S3/qa"

deploy-production:
  stage: deploy
  needs: [release-qa]
  environment: {{name: production}}
  rules: [{{if: '$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH', when: manual}}]
  script:
    - for e in qa prod; do aws s3 sync "$REGISTRY_S3/$e" "registry/$e"; done
    - QA_VERSION=$(mlpilot --registry registry/qa latest --name "$MODEL_NAME" --stage qa)
    - mlpilot release --config environments.yml --name "$MODEL_NAME" --version "$QA_VERSION" --from qa --to prod --released-by "$GITLAB_USER_LOGIN" --reason "pipeline $CI_PIPELINE_ID"
    - aws s3 sync registry/prod "$REGISTRY_S3/prod"
"""


def init_cicd(provider: str, *, name: str, data: str, target: str, out=".", metric: str = "roc_auc",
              floor: float = 0.7, budget: int = 300, install: str | None = None, overwrite: bool = False) -> list:
    """Write a pipeline (``github`` or ``gitlab``) plus ``gates.yml`` and ``environments.yml``
    into ``out``. Existing files are kept unless ``overwrite``. Returns the paths written."""
    from importlib.metadata import PackageNotFoundError, version
    if provider not in ("github", "gitlab"):
        raise ValueError("provider must be 'github' or 'gitlab'")
    try:
        install = install or f"dataspoof-mlpilot[aws]=={version('dataspoof-mlpilot')}"
    except PackageNotFoundError:
        install = install or "dataspoof-mlpilot[aws]"
    root = Path(out)
    files = {
        (".github/workflows/model-" + name + ".yml") if provider == "github" else ".gitlab-ci.yml":
            (GITHUB_TEMPLATE if provider == "github" else GITLAB_TEMPLATE).format(
                name=name, data=data, data_glob=str(Path(data).parent.as_posix()) + "/**", target=target,
                budget=budget, install=install),
        "gates.yml": GATES_TEMPLATE.format(metric=metric, floor=floor),
        "environments.yml": ENVIRONMENTS_TEMPLATE.format(metric=metric, floor=floor),
    }
    written = []
    for rel, text in files.items():
        p = root / rel
        if p.exists() and not overwrite:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        written.append(p)
    return written
