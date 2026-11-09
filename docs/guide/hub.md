# Hub: the platform server


A multi-user server for teams, installed with `pip install "dataspoof-mlpilot[server]"`:

```bash
mlpilot hub create-admin --name admin      # prints an API token once
mlpilot hub serve --host 0.0.0.0 --port 8080
# or: docker compose -f deploy/hub/docker-compose.yml up -d   (Hub + PostgreSQL)
```

```python
from mlpilot.hub import HubClient
hub = HubClient("https://hub.example.com", token=os.environ["AC_HUB_TOKEN"])
run = hub.start_run("churn", experiment="weekly", name="automl")
v = hub.register("churn", "churn-model", model, run_id=run["id"], metrics={"test_roc_auc": 0.91})
hub.promote("churn", "churn-model", v["version"], "production", reason="beats v3")   # -> pending approval
model = hub.load_model("churn", "churn-model", stage="production")
```

**Identity**
- **Single sign-on (OIDC):** works with Okta, Entra ID / Azure AD, Google, Keycloak and Auth0.
- **Group mapping:** identity-provider groups map to workspace roles, synced on every login.
- **API tokens** for service accounts. Tokens are stored only as hashes, can expire and can be
  revoked.

**Roles, per workspace**

| Role | Can |
|---|---|
| viewer | read |
| data_scientist | log runs, register models, request promotion |
| approver | approve or reject promotions |
| admin | manage members |

People outside a workspace can't tell that it exists.

**Governance**
- **Two-person rule:** promoting a model to production needs approval from someone other than
  the requester, with the evidence attached.
- **Policy gates:** each workspace can require minimum metrics or signed models.
- **Tamper-evident audit log:**
  - It records every action, including refused attempts.
  - Each entry is hash-chained to the one before it; `GET /api/audit/verify` checks the chain.
  - `GET /api/audit/export` exports JSONL for your security tools (SIEM).

**Storage:** metadata goes to PostgreSQL. Artifacts are immutable and checksummed, stored on
disk, S3, GCS or Azure. The server inspects uploads without ever unpickling them, and refuses
path-traversal ("zip-slip") archives.

**Operations**
- **Web UI** for models, approvals, runs, monitoring, members and the audit log.
- **OpenAPI docs** at `/docs`.
- **Deployment:**
  - a Helm chart (`deploy/helm`): non-root, read-only filesystem, health probes, pod
    disruption budget;
  - a Terraform module for AWS (`deploy/terraform/aws`): S3, RDS PostgreSQL, Secrets Manager,
    an IAM role scoped to the Hub's service account, and the Helm release.
