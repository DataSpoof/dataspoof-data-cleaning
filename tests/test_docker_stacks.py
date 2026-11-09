"""End-to-end tests against real infrastructure in Docker (opt-in: AC_DOCKER_TESTS=1):

* Prometheus + Grafana (deploy/monitoring): a live model server is scraped, the alert rules load
  and fire, Grafana provisions the dashboard and every panel query runs.
* HashiCorp Vault: secrets resolved through KV v2 with token and AppRole auth.
* Open Policy Agent: the shipped Rego policies decide Hub access and promotion gates.
* Marquez: OpenLineage events are accepted and the lineage graph is queryable.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import automatedcleaning as ac

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.environ.get("AC_DOCKER_TESTS") != "1" or not shutil.which("docker"),
                                reason="set AC_DOCKER_TESTS=1 (needs Docker)")
FAST = dict(time_budget=30, cv=3, verbose=False, max_trials=1, include=["logistic_regression"])


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method, url, body=None, headers=None, auth=None, timeout=10):
    data = json.dumps(body).encode() if body is not None and not isinstance(body, bytes) else body
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json",
                                                                          **(headers or {})})
    if auth:
        req.add_header("Authorization", "Basic " + base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode())
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode())


def wait_until(fn, timeout=120, every=1.0, what="condition"):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            last = fn()
            if last:
                return last
        except Exception as exc:              # service still starting
            last = exc
        time.sleep(every)
    raise TimeoutError(f"{what} not reached: {last!r}")


def docker(*args, check=True, env=None):
    p = subprocess.run(["docker", *args], capture_output=True, text=True, env={**os.environ, **(env or {}),
                                                                                "MSYS_NO_PATHCONV": "1"})
    if check and p.returncode:
        raise RuntimeError(f"docker {' '.join(args)}: {p.stderr[-2000:]}")
    return p.stdout


# ═══════════════════════════════ Prometheus + Grafana ═══════════════════════════
@pytest.fixture(scope="module")
def model_server():
    from sklearn.datasets import make_classification
    from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server
    from socketserver import ThreadingMixIn
    X, y = make_classification(n_samples=1200, n_features=4, n_informative=3, n_redundant=0, random_state=0)
    df = pl.DataFrame(X, schema=list("abcd")).with_columns(pl.Series("y", y))
    model = ac.AutoML("y", **FAST).fit(df.head(800))
    mon = ac.Monitor(model, df.tail(400), name="churn")
    app = ac.make_app(model, name="churn", version="7", monitor=mon, drift_window=400, drift_every=200)

    class Srv(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    class Quiet(WSGIRequestHandler):
        def log_message(self, *a):
            pass
    port = free_port()
    httpd = make_server("0.0.0.0", port, app, server_class=Srv, handler_class=Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield {"port": port, "app": app, "df": df}
    httpd.shutdown()
    app.close()


@pytest.fixture(scope="module")
def monitoring_stack(model_server):
    project = f"acmon{secrets.token_hex(3)}"
    pw = secrets.token_urlsafe(12)
    ports = {"PROMETHEUS_PORT": str(free_port()), "GRAFANA_PORT": str(free_port())}
    env = {"MODEL_TARGET": f"host.docker.internal:{model_server['port']}", "GRAFANA_ADMIN_PASSWORD": pw, **ports}
    compose = str(ROOT / "deploy" / "monitoring" / "docker-compose.yml")
    docker("compose", "-p", project, "-f", compose, "up", "-d", env=env)
    try:
        prom = f"http://127.0.0.1:{ports['PROMETHEUS_PORT']}"
        graf = f"http://127.0.0.1:{ports['GRAFANA_PORT']}"
        wait_until(lambda: http("GET", f"{prom}/-/ready")[0] == 200, what="prometheus ready")
        wait_until(lambda: http("GET", f"{graf}/api/health")[0] == 200, what="grafana ready")
        yield {"prom": prom, "graf": graf, "auth": ("admin", pw)}
    finally:
        docker("compose", "-p", project, "-f", compose, "down", "-v", check=False, env=env)


def test_prometheus_scrapes_live_model_and_alerts(model_server, monitoring_stack):
    base = f"http://127.0.0.1:{model_server['port']}"
    df = model_server["df"]
    rng = np.random.default_rng(0)
    ids = []
    for i in range(6):                                                   # drifted traffic: column a shifted
        recs = [{**r, "a": r["a"] + 3 + rng.normal(0, 0.1)} for r in df.drop("y").slice(i * 100, 100).to_dicts()]
        status, res = http("POST", f"{base}/predict", {"records": recs})
        assert status == 200
        ids += list(zip(res["prediction_ids"], res["predictions"]))
    fb = [{"prediction_id": pid, "label": int(p) if j % 4 else 1 - int(p)} for j, (pid, p) in enumerate(ids[:200])]
    assert http("POST", f"{base}/feedback", {"feedback": fb})[1]["matched"] == 200
    model_server["app"].drift_pool.submit(lambda: None).result()        # let the drift check finish
    prom = monitoring_stack["prom"]

    def query(q):
        status, res = http("GET", f"{prom}/api/v1/query?" + urllib.parse.urlencode({"query": q}))
        assert status == 200 and res["status"] == "success", (q, res)
        return res["data"]["result"]
    psi = wait_until(lambda: query('ac_feature_psi{feature="a"}'), timeout=120, what="ac_feature_psi scraped")
    assert float(psi[0]["value"][1]) > 0.25 and psi[0]["metric"]["model"] == "churn"
    assert float(query("ac_live_accuracy")[0]["value"][1]) == pytest.approx(0.75)
    assert query('ac_performance{kind="estimated"}')
    up = query('up{job="automatedcleaning"}')
    assert up and up[0]["value"][1] == "1"
    rules = http("GET", f"{prom}/api/v1/rules")[1]["data"]["groups"]
    names = {r["name"] for g in rules for r in g["rules"]}
    unhealthy = {r["name"]: r.get("lastError") for g in rules for r in g["rules"] if r["health"] == "err"}
    assert len(names) == 14 and not unhealthy, unhealthy

    def drift_alert():
        alerts = http("GET", f"{prom}/api/v1/alerts")[1]["data"]["alerts"]
        return [a for a in alerts if a["labels"]["alertname"] == "ACFeatureDrift" and a["labels"].get("feature") == "a"]
    alert = wait_until(drift_alert, timeout=120, what="ACFeatureDrift pending")
    assert alert[0]["state"] in ("pending", "firing")


def test_grafana_dashboard_is_provisioned_and_every_query_runs(model_server, monitoring_stack):
    graf, auth, prom = monitoring_stack["graf"], monitoring_stack["auth"], monitoring_stack["prom"]
    status, dash = wait_until(lambda: (lambda r: r if r[0] == 200 else None)(
        http("GET", f"{graf}/api/dashboards/uid/ac-model", auth=auth)), what="dashboard provisioned")
    panels = [p for p in dash["dashboard"]["panels"] if p["type"] != "row"]
    assert dash["meta"]["folderTitle"] == "AutomatedCleaning" and len(panels) == 20
    exprs = [t["expr"] for p in panels for t in p.get("targets", [])]
    for e in exprs:
        q = (e.replace("$model", ".*").replace("$job", ".*").replace("$__rate_interval", "1m"))
        status, res = http("GET", f"{prom}/api/v1/query?" + urllib.parse.urlencode({"query": q}))
        assert status == 200 and res["status"] == "success", (e, res)
    # and through Grafana's own data-source proxy, as the dashboard does
    body = {"queries": [{"refId": "A", "datasource": {"uid": "prometheus", "type": "prometheus"},
                         "expr": 'max by (feature) (ac_feature_psi{model=~".*"})', "instant": True}],
            "from": "now-5m", "to": "now"}
    status, res = http("POST", f"{graf}/api/ds/query", body, auth=auth)
    assert status == 200
    frames = res["results"]["A"]["frames"]
    assert {f["schema"]["fields"][1]["labels"]["feature"] for f in frames} >= {"a", "b"}


# ═══════════════════════════════════ Vault ══════════════════════════════════════
@pytest.fixture(scope="module")
def vault():
    port = free_port()
    root = secrets.token_hex(8)
    cid = docker("run", "-d", "--rm", "--cap-add=IPC_LOCK", "-e", f"VAULT_DEV_ROOT_TOKEN_ID={root}",
                 "-p", f"{port}:8200", "hashicorp/vault:1.18").strip()
    addr = f"http://127.0.0.1:{port}"
    try:
        wait_until(lambda: http("GET", f"{addr}/v1/sys/health")[0] == 200, what="vault ready")
        yield addr, root
    finally:
        docker("rm", "-f", cid, check=False)


def test_vault_token_approle_kv1_kv2_and_no_leaks(vault, monkeypatch, tmp_path):
    import io as _io

    from automatedcleaning import secretstore as ss
    addr, root = vault
    admin = ss.VaultClient(addr, token=root)
    admin.write("secret/ml/hub", {"database_url": f"sqlite:///{(tmp_path / 'vault-hub.db').as_posix()}",
                                  "api_key": "k-123", "warehouse": f"sqlite:///{(tmp_path / 'wh.db').as_posix()}"})
    admin.write("secret/other/payroll", {"password": "p-999"})
    H = {"X-Vault-Token": root}
    assert http("POST", f"{addr}/v1/sys/mounts/kv1", {"type": "kv", "options": {"version": "1"}}, headers=H)[0] == 204
    admin.write("kv1:ml/legacy", {"password": "legacy-pw"})
    # an AppRole that may only read secret/ml/* and kv1/ml/*
    assert http("POST", f"{addr}/v1/sys/auth/approle", {"type": "approle"}, headers=H)[0] == 204
    policy = 'path "secret/data/ml/*" { capabilities = ["read"] }\npath "kv1/ml/*" { capabilities = ["read"] }'
    assert http("PUT", f"{addr}/v1/sys/policies/acl/ml-read", {"policy": policy}, headers=H)[0] == 204
    assert http("POST", f"{addr}/v1/auth/approle/role/model-server",
                {"token_policies": ["ml-read"], "token_ttl": "1h"}, headers=H)[0] == 204
    role_id = http("GET", f"{addr}/v1/auth/approle/role/model-server/role-id", headers=H)[1]["data"]["role_id"]
    secret_id = http("POST", f"{addr}/v1/auth/approle/role/model-server/secret-id", {}, headers=H)[1]["data"]["secret_id"]

    monkeypatch.setenv("VAULT_ADDR", addr)
    monkeypatch.delenv("VAULT_TOKEN", raising=False)
    monkeypatch.setenv("VAULT_ROLE_ID", role_id)
    monkeypatch.setenv("VAULT_SECRET_ID", secret_id)
    ss.clear_cache()
    assert ss.get_secret("vault://secret/ml/hub#api_key") == "k-123"           # AppRole login + KV v2
    assert ss.get_secret("vault://kv1:ml/legacy#password") == "legacy-pw"      # KV v1
    assert ss.resolve("plain-value") == "plain-value" and ss.resolve(None, default="d") == "d"
    with pytest.raises(ss.SecretError) as denied:
        ss.get_secret("vault://secret/other/payroll#password")                 # outside the role's policy
    assert "403" in str(denied.value) and "p-999" not in str(denied.value)
    with pytest.raises(ss.SecretError, match="no key"):
        ss.get_secret("vault://secret/ml/hub#nope")

    # the library resolves references wherever it reads credentials
    from sklearn.datasets import make_classification
    X, y = make_classification(200, 4, n_informative=3, n_redundant=0, random_state=0)
    model = ac.AutoML("y", **FAST).fit(pl.DataFrame(X, schema=list("abcd")).with_columns(pl.Series("y", y)))
    app = ac.make_app(model, api_key="vault://secret/ml/hub#api_key")
    status = {}
    for key, label in (("k-123", "ok"), ("wrong", "bad")):
        env = {"REQUEST_METHOD": "GET", "PATH_INFO": "/metadata", "CONTENT_LENGTH": "0",
               "wsgi.input": _io.BytesIO(b""), "HTTP_X_API_KEY": key}
        app(env, lambda s, h, label=label: status.update({label: s}))
    assert status["ok"].startswith("200") and status["bad"].startswith("401")
    from automatedcleaning.ingestion import SQLSource
    db = SQLSource("vault://secret/ml/hub#warehouse")
    db.execute("CREATE TABLE t (x INTEGER)")
    db.insert("t", [{"x": 1}, {"x": 2}])
    assert db.query("SELECT SUM(x) AS s FROM t")["s"][0] == 3
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from automatedcleaning.hub.app import create_app
    hub = TestClient(create_app(database_url="vault://secret/ml/hub#database_url", artifact_root=str(tmp_path / "a")))
    assert hub.get("/health").json() == {"status": "ok", "database": "ok"}
    assert (tmp_path / "vault-hub.db").exists()
    ss.clear_cache()


# ════════════════════════════════════ OPA ═══════════════════════════════════════
@pytest.fixture(scope="module")
def opa_server():
    port = free_port()
    policies = str(ROOT / "deploy" / "opa" / "policies")
    cid = docker("run", "-d", "--rm", "-p", f"{port}:8181", "-v", f"{policies}:/policies:ro",
                 "openpolicyagent/opa:1.4.2", "run", "--server", "--addr=0.0.0.0:8181", "/policies").strip()
    url = f"http://127.0.0.1:{port}"
    try:
        wait_until(lambda: http("GET", f"{url}/health")[0] == 200, what="opa ready")
        yield url
    finally:
        docker("rm", "-f", cid, check=False)


def test_hub_with_real_opa_policies(opa_server, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sklearn.datasets import make_classification

    from automatedcleaning.hub.app import bootstrap_admin, create_app
    from automatedcleaning.hub.policy import OPAPolicy
    from automatedcleaning.hub.storage import tar_directory
    url = opa_server
    db_url = f"sqlite:///{(tmp_path / 'hub.db').as_posix()}"
    root_tok = bootstrap_admin(db_url, name="root", email="root@corp.com")
    client = TestClient(create_app(database_url=db_url, artifact_root=str(tmp_path / "art"), opa=OPAPolicy(url)))
    H = lambda t: {"Authorization": f"Bearer {t}"}          # noqa: E731
    assert client.post("/api/workspaces", json={"name": "churn", "require_approval": False},
                       headers=H(root_tok)).status_code == 201
    sa = client.post("/api/service-accounts", json={"name": "ci"}, headers=H(root_tok)).json()
    assert client.put("/api/workspaces/churn/members", json={"user": sa["id"], "role": "approver"},
                      headers=H(root_tok)).status_code == 200
    ci = sa["token"]
    X, y = make_classification(200, 4, n_informative=3, n_redundant=0, random_state=0)
    model = ac.AutoML("y", **FAST).fit(pl.DataFrame(X, schema=list("abcd")).with_columns(pl.Series("y", y)))
    blob = tar_directory(model.save(tmp_path / "m"))
    r = client.post("/api/workspaces/churn/models/churn/versions", headers=H(ci),
                    files={"artifact": ("m.tar.gz", blob, "application/gzip")},
                    data={"metrics": json.dumps({"test_roc_auc": 0.9}), "description": "weekly churn model"})
    assert r.status_code == 201, r.text

    def tr(tok, stage):
        return client.post("/api/workspaces/churn/models/churn/versions/1/transitions", json={"stage": stage},
                           headers=H(tok))
    # the service account promotes only up to QA; nobody skips QA
    blocked = tr(ci, "staging")
    assert blocked.status_code == 422 and any("service accounts" in f for f in blocked.json()["detail"]["failures"])
    skip = tr(root_tok, "production")
    assert skip.status_code == 422 and any("must pass QA" in f for f in skip.json()["detail"]["failures"])
    assert tr(ci, "qa").status_code == 200
    assert tr(root_tok, "staging").status_code == 200
    assert tr(root_tok, "production").json()["version"]["stage"] == "production"
    # a change freeze pushed to OPA at runtime: writes refused, reads fine
    assert http("PUT", f"{url}/v1/data/automatedcleaning/config/frozen_workspaces", ["churn"])[0] == 204
    frozen = client.post("/api/workspaces/churn/runs", json={"experiment": "e", "name": "r"}, headers=H(ci))
    assert frozen.status_code == 403 and "change freeze" in frozen.json()["detail"]
    assert client.get("/api/workspaces/churn/models", headers=H(ci)).status_code == 200
    assert http("PUT", f"{url}/v1/data/automatedcleaning/config/frozen_workspaces", [])[0] == 204
    # refusals stay in the tamper-evident audit trail
    audit = client.get("/api/audit", headers=H(root_tok)).json()
    entries = audit["entries"] if isinstance(audit, dict) else audit
    actions = [e["action"] for e in entries]
    assert "access.denied_by_policy" in actions and actions.count("model.promotion_blocked") == 2
    assert client.get("/api/audit/verify", headers=H(root_tok)).json()["valid"]
    # OPA unreachable: fail closed
    down = TestClient(create_app(database_url=db_url, artifact_root=str(tmp_path / "art"),
                                 opa=OPAPolicy(f"http://127.0.0.1:{free_port()}", timeout=0.5)))
    assert down.get("/api/workspaces/churn/models", headers=H(ci)).status_code == 503


# ═══════════════════════════════════ Marquez ════════════════════════════════════
@pytest.fixture(scope="module")
def marquez():
    net = f"acmq{secrets.token_hex(3)}"
    port = free_port()
    docker("network", "create", net)
    pg = mq = None
    try:
        pg = docker("run", "-d", "--rm", "--network", net, "--network-alias", "mqdb", "-e", "POSTGRES_USER=marquez",
                    "-e", "POSTGRES_PASSWORD=marquez", "-e", "POSTGRES_DB=marquez", "postgres:16-alpine").strip()
        wait_until(lambda: "accepting connections" in docker("exec", pg, "pg_isready", "-U", "marquez", check=False),
                   what="postgres ready")
        mq = docker("run", "-d", "--rm", "--network", net, "-p", f"{port}:5000", "-e", "POSTGRES_HOST=mqdb",
                    "-e", "SEARCH_ENABLED=false", "marquezproject/marquez:0.51.1").strip()
        url = f"http://127.0.0.1:{port}"
        wait_until(lambda: http("GET", f"{url}/api/v1/namespaces")[0] == 200, timeout=180, what="marquez ready")
        yield url
    finally:
        for c in (mq, pg):
            if c:
                docker("rm", "-f", c, check=False)
        docker("network", "rm", net, check=False)


def test_openlineage_events_in_marquez(marquez):
    from automatedcleaning.lineage import HttpTransport, Lineage, dataset
    lin = Lineage(HttpTransport(marquez), namespace="acme", strict=True)
    raw = pl.DataFrame({"id": [1, 2, 3], "spend": [10.0, None, 30.0]})
    with lin.run("clean_customers", inputs=[dataset("s3://raw/customers.csv", raw)],
                 outputs=[dataset("warehouse.customers_clean", raw.drop_nulls())], description="Dedup + impute"):
        pass
    with lin.run("train_churn", inputs=[dataset("warehouse.customers_clean")], outputs=[dataset("models/churn/v4")]):
        pass
    with pytest.raises(RuntimeError):
        with lin.run("broken", inputs=[dataset("warehouse.customers_clean")]):
            raise RuntimeError("bad data")

    def job(name):
        status, j = http("GET", f"{marquez}/api/v1/namespaces/acme/jobs/{name}")
        return j if status == 200 and j.get("latestRun") else None
    j = wait_until(lambda: job("train_churn"), what="job in marquez")
    assert j["latestRun"]["state"] == "COMPLETED"
    assert wait_until(lambda: job("broken"))["latestRun"]["state"] == "FAILED"
    assert job("clean_customers")["description"] == "Dedup + impute"
    status, ds = http("GET", f"{marquez}/api/v1/namespaces/automatedcleaning/datasets/warehouse.customers_clean")
    assert status == 200 and [f["name"] for f in ds["fields"]] == ["id", "spend"]
    node = urllib.parse.quote("dataset:automatedcleaning:models/churn/v4", safe="")
    status, graph = http("GET", f"{marquez}/api/v1/lineage?nodeId={node}&depth=10")
    ids = {n["id"] for n in graph["graph"]}
    assert {"job:acme:train_churn", "job:acme:clean_customers", "dataset:s3://raw:customers.csv",
            "dataset:automatedcleaning:warehouse.customers_clean"} <= ids


# ═══════════════════════════════════ actionlint ═════════════════════════════════
def test_workflows_pass_actionlint(tmp_path):
    from automatedcleaning.cicd import init_cicd
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    init_cicd("github", name="churn", data="data/train.parquet", target="churned", out=tmp_path)
    shutil.copytree(ROOT / ".github" / "workflows", tmp_path / ".github" / "workflows", dirs_exist_ok=True)
    out = docker("run", "--rm", "-v", f"{tmp_path}:/repo", "-w", "/repo", "rhysd/actionlint:1.7.7", "-color", check=False)
    assert out.strip() == "", out
    p = subprocess.run(["docker", "run", "--rm", "-v", f"{tmp_path}:/repo", "-w", "/repo", "rhysd/actionlint:1.7.7"],
                       capture_output=True, text=True, env={**os.environ, "MSYS_NO_PATHCONV": "1"})
    assert p.returncode == 0, p.stdout + p.stderr


# ═══════════════════════ shipped policy + alert-rule unit tests ═════════════════
def test_opa_policy_unit_tests_and_format():
    policies = str(ROOT / "deploy" / "opa" / "policies")
    out = docker("run", "--rm", "-v", f"{policies}:/p:ro", "openpolicyagent/opa:1.4.2", "test", "/p", "-v")
    assert re.search(r"PASS: (\d+)/\1", out), out
    assert docker("run", "--rm", "-v", f"{policies}:/p:ro", "openpolicyagent/opa:1.4.2", "fmt", "-l", "/p").strip() == ""
    docker("run", "--rm", "-v", f"{policies}:/p:ro", "openpolicyagent/opa:1.4.2", "check", "--strict", "/p")


def test_prometheus_alert_rules_unit_tests():
    rules = str(ROOT / "deploy" / "monitoring" / "prometheus")
    base = ["run", "--rm", "-v", f"{rules}:/r:ro", "-w", "/r", "--entrypoint", "promtool", "prom/prometheus:v3.4.1"]
    assert "SUCCESS: 14 rules found" in docker(*base, "check", "rules", "alerts.yml")
    assert "SUCCESS" in docker(*base, "test", "rules", "alerts_test.yml")
