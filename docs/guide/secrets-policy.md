# Secrets & policy


```bash
export AC_API_KEY="vault://secret/ml/churn#api_key"          # resolved at start-up, never stored
export AC_HUB_DATABASE_URL="aws-sm://prod/hub#database_url"
export AC_HUB_OPA_URL="http://opa:8181"                        # policy as code for the Hub
```

**Secret references** work anywhere the library reads a credential: the model API key, the Hub's
database / OIDC / role settings, SQL and Snowflake connections, and PagerDuty and Opsgenie keys.
- **Vault** (KV v1 and v2) signs in with a token, AppRole or Kubernetes auth, and no extra
  package is needed.
- Also supported: AWS Secrets Manager, environment variables and mounted files.
- Values are cached, and error messages never contain them.

**Open Policy Agent** adds rules on top of the Hub's own roles. OPA can only take access away,
never grant it.
- The shipped Rego policies (`deploy/opa`, with 10 unit tests) enforce that:
  - nothing skips QA;
  - production needs a description and the organisation's metric floors;
  - the four-fifths fairness rule applies;
  - models must be signed;
  - service accounts can promote at most to QA and can never approve;
  - frozen workspaces are read-only;
  - restricted workspaces only accept identities from the listed e-mail domains.
- Refusals go into the tamper-evident audit log.
- If OPA is unreachable, requests are refused: the Hub fails closed.

**Tested against the real servers in Docker:**
- **Vault:** AppRole login with a read-only policy, refused paths, and a model server, SQL source
  and Hub configured purely by `vault://` references.
- **OPA:** the Hub enforced every rule, including a change freeze pushed to OPA while it was
  running.
