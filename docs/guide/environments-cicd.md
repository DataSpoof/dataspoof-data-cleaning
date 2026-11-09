# Environments & CI/CD


```bash
automatedcleaning cicd init --provider github --name churn --data data/train.parquet --target churned
#  -> .github/workflows/model-churn.yml, gates.yml, environments.yml
automatedcleaning train --data data/train.parquet --target churned --out build/model --group customer_id
automatedcleaning gate  --model build/model --data build/test.parquet --gates gates.yml     # exit 1 = blocked
automatedcleaning register --model build/model --registry registry/dev --name churn --metrics build/metrics.json
automatedcleaning release --config environments.yml --name churn --version 7 --from qa --to prod \
                          --released-by ana --approved-by bob,carol
```

**Environments:**
- dev, QA and prod each have their **own registry**, and two environments may not share one.
- Models move forward only through `release`, one environment at a time, so nothing skips QA.
- The target environment's rules are enforced: metric floors, the number of approvers (never
  counting the person releasing), and a signature check against its public key.
- Every release, and every refusal, is recorded on both sides, so production always knows where
  a model came from.
- The registry gains a `qa` stage: `dev → qa → staging → production`.

**The pipeline** has five stages: train, gate, register, release to QA, then a release to
production approved through GitHub environments and a blue/green deploy.
- It is generated for GitHub Actions or GitLab CI.
- Every stage is one CLI command, so it runs the same way locally and in CI.
- **Gates** cover metric floors, fairness limits and robustness. A model that fails them never
  gets registered.
