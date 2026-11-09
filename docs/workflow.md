# Workflow

![AutomatedCleaning workflow: data in, build and validate, release, operate](assets/workflow.svg)

Read lane 1 left to right, lane 2 right to left, and so on: the flow snakes down the page.

- **Gates** (amber) can stop a run: the quality gate, the poisoning check, the CI gate, and the two
  releases between environments.
- **Dashed lines** are side paths. The held-out test split skips feature engineering and AutoML and
  goes straight to evaluation. Served inputs reach the monitor at once, so drift and estimated
  accuracy are known before any labels exist.
- **The orange loop** is retraining. A challenger re-enters at AutoML, so it passes the same gates
  and releases as the model it may replace.
- **Across every stage:** secrets come from Vault or AWS Secrets Manager, OPA policies and the Hub's
  audit log govern changes, and every run reports lineage and experiment tracking.

| Lane | Stages |
|---|---|
| 1 · Data in | sources → ingest → version → clean → quality gate → split |
| 2 · Build & validate | features → poisoning check → AutoML → evaluate → CI gate → register |
| 3 · Release | dev → QA → prod registries, approvals, OPA, signatures → deploy |
| 4 · Operate | serve → feedback → monitor → alerts → retrain |

Each stage is a library call or a CLI command; the [Environments & CI/CD](guide/environments-cicd.md) guide shows the commands a pipeline runs.
