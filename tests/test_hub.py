"""AutomatedCleaning Hub: RBAC, SSO (OIDC), tokens, runs, model registry, approvals, policy gates,
audit log integrity, monitoring, the UI and the Python client against a live server."""

import base64
import datetime as dt
import io
import json
import socket
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import polars as pl
import pytest
from sklearn.datasets import make_classification

pytest.importorskip("fastapi")
pytest.importorskip("jwt")
from fastapi.testclient import TestClient  # noqa: E402

import automatedcleaning as ac  # noqa: E402
from automatedcleaning.hub.app import bootstrap_admin, create_app  # noqa: E402
from automatedcleaning.hub.storage import tar_directory  # noqa: E402


@pytest.fixture(scope="module")
def model_and_data():
    X, y = make_classification(300, 4, n_informative=3, n_redundant=0, random_state=0)
    df = pl.DataFrame(X, schema=list("abcd")).with_columns(pl.Series("y", y))
    return ac.AutoML("y", time_budget=20, cv=3, verbose=False, max_trials=1, include=["logistic_regression"]).fit(df), df


@pytest.fixture()
def hub(tmp_path):
    url = f"sqlite:///{tmp_path / 'hub.db'}"
    admin_token = bootstrap_admin(url, name="root", email="root@corp.com")
    app = create_app(database_url=url, artifact_root=str(tmp_path / "artifacts"))
    client = TestClient(app)
    H = lambda tok: {"Authorization": f"Bearer {tok}"}          # noqa: E731
    r = client.post("/api/workspaces", json={"name": "churn", "policy": {"min_metrics": {"test_roc_auc": 0.6}}},
                    headers=H(admin_token))
    assert r.status_code == 201
    tokens = {"root": admin_token}
    for name, role in [("alice", "data_scientist"), ("bob", "approver"), ("carol", "viewer"), ("mallory", None)]:
        sa = client.post("/api/service-accounts", json={"name": name}, headers=H(admin_token)).json()
        tokens[name] = sa["token"]
        if role:
            assert client.put("/api/workspaces/churn/members", json={"user": sa["id"], "role": role},
                              headers=H(admin_token)).status_code == 200
    return client, tokens, H, url


def upload(client, H, tok, model, tmp_path, name="churn-model", metrics=None, blob=None):
    blob = blob if blob is not None else tar_directory(model.save(tmp_path / f"m{time.time_ns()}"))
    return client.post(f"/api/workspaces/churn/models/{name}/versions",
                       files={"artifact": ("model.tar.gz", blob, "application/gzip")},
                       data={"metrics": json.dumps(metrics or {"test_roc_auc": 0.9}), "description": "test"},
                       headers=H(tok))


def test_auth_rbac_and_tokens(hub):
    client, t, H, _ = hub
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/me", headers=H("ach_wrong")).status_code == 401
    me = client.get("/api/me", headers=H(t["root"])).json()
    assert me["is_admin"] and me["workspaces"] == {"churn": "admin"}
    assert client.get("/api/me", headers=H(t["alice"])).json()["workspaces"] == {"churn": "data_scientist"}
    run = {"experiment": "weekly", "name": "r1"}
    assert client.post("/api/workspaces/churn/runs", json=run, headers=H(t["carol"])).status_code == 403
    assert client.post("/api/workspaces/churn/runs", json=run, headers=H(t["mallory"])).status_code == 404
    assert client.get("/api/workspaces/churn/models", headers=H(t["mallory"])).status_code == 404
    assert client.post("/api/workspaces", json={"name": "x"}, headers=H(t["alice"])).status_code == 403
    assert client.post("/api/workspaces", json={"name": "../bad"}, headers=H(t["root"])).status_code == 422
    new = client.post("/api/tokens", json={"name": "laptop", "days": 1}, headers=H(t["alice"])).json()["token"]
    assert client.get("/api/me", headers=H(new)).status_code == 200
    tid = next(x["id"] for x in client.get("/api/tokens", headers=H(t["alice"])).json() if x["name"] == "laptop")
    assert "token_hash" not in client.get("/api/tokens", headers=H(t["alice"])).json()[0]
    client.delete(f"/api/tokens/{tid}", headers=H(t["alice"]))
    assert client.get("/api/me", headers=H(new)).status_code == 401
    r = client.get("/health")
    assert r.status_code == 200 and r.headers["X-Frame-Options"] == "DENY" and "script-src 'self'" in r.headers["Content-Security-Policy"]
    assert "AutomatedCleaning Hub" in client.get("/").text and "esc(" in client.get("/ui.js").text


def test_runs(hub):
    client, t, H, _ = hub
    run = client.post("/api/workspaces/churn/runs", json={"experiment": "weekly", "name": "r1", "params": {"a": 1}},
                      headers=H(t["alice"])).json()
    r = client.patch(f"/api/runs/{run['id']}", json={"metrics": {"auc": 0.9}, "status": "finished"}, headers=H(t["alice"]))
    assert r.json()["status"] == "finished" and r.json()["metrics"] == {"auc": 0.9}
    assert client.patch(f"/api/runs/{run['id']}", json={"metrics": {"x": 1}}, headers=H(t["alice"])).status_code == 409
    assert client.patch(f"/api/runs/{run['id']}", json={"metrics": {"x": 1}}, headers=H(t["carol"])).status_code == 403
    rows = client.get("/api/workspaces/churn/runs?experiment=weekly", headers=H(t["carol"])).json()
    assert len(rows) == 1 and rows[0]["params"] == {"a": 1}


def test_registry_approvals_policy_and_audit(hub, model_and_data, tmp_path):
    client, t, H, url = hub
    model, df = model_and_data
    assert upload(client, H, t["carol"], model, tmp_path).status_code == 403
    v1 = upload(client, H, t["alice"], model, tmp_path)
    assert v1.status_code == 201 and v1.json()["version"] == 1 and v1.json()["stage"] == "dev"
    # tampered model file: checksum mismatch
    d = model.save(tmp_path / "tampered")
    (d / "model.joblib").write_bytes(b"evil")
    assert upload(client, H, t["alice"], model, tmp_path, blob=tar_directory(d)).status_code == 422
    # path traversal inside the tarball is refused
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("../../escape.txt")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"hi"))
    assert upload(client, H, t["alice"], model, tmp_path, blob=buf.getvalue()).status_code == 422
    base = "/api/workspaces/churn/models/churn-model/versions"
    assert client.post(f"{base}/1/transitions", json={"stage": "staging"}, headers=H(t["alice"])).json()["status"] == "applied"
    r = client.post(f"{base}/1/transitions", json={"stage": "production", "reason": "first"}, headers=H(t["alice"]))
    assert r.status_code == 202
    req = r.json()["request"]["id"]
    assert client.post(f"{base}/1/transitions", json={"stage": "production"}, headers=H(t["alice"])).status_code == 409
    assert client.post(f"/api/promotions/{req}/approve", json={}, headers=H(t["alice"])).status_code == 403   # not approver
    pending = client.get("/api/workspaces/churn/promotions", headers=H(t["bob"])).json()
    assert len(pending) == 1 and pending[0]["evidence"]["metrics"]["test_roc_auc"] == 0.9
    r = client.post(f"/api/promotions/{req}/approve", json={"note": "lgtm"}, headers=H(t["bob"]))
    assert r.json()["status"] == "approved" and r.json()["version"]["stage"] == "production"
    # four-eyes: an approver can't approve their own request
    v2 = upload(client, H, t["alice"], model, tmp_path, metrics={"test_roc_auc": 0.95}).json()
    own = client.post(f"{base}/2/transitions", json={"stage": "production"}, headers=H(t["bob"])).json()["request"]["id"]
    assert client.post(f"/api/promotions/{own}/approve", json={}, headers=H(t["bob"])).status_code == 403
    assert client.post(f"/api/promotions/{own}/approve", json={}, headers=H(t["root"])).status_code == 200
    versions = {v["version"]: v["stage"] for v in client.get(base, headers=H(t["carol"])).json()}
    assert versions == {1: "archived", 2: "production"} and v2["version"] == 2
    # policy gate: below the workspace's min metric
    upload(client, H, t["alice"], model, tmp_path, metrics={"test_roc_auc": 0.5})
    r = client.post(f"{base}/3/transitions", json={"stage": "production"}, headers=H(t["alice"]))
    assert r.status_code == 422 and "policy" in json.dumps(r.json())
    resolved = client.get("/api/workspaces/churn/models/churn-model/resolve?stage=production", headers=H(t["carol"])).json()
    assert resolved["version"] == 2
    art = client.get(f"{base}/2/artifact", headers=H(t["carol"]))
    assert art.status_code == 200 and art.headers["X-Artifact-SHA256"] == resolved["artifact_sha256"]
    # audit: every step recorded, chain intact; editing history is detected
    log = client.get("/api/audit?workspace_name=churn&limit=100", headers=H(t["root"])).json()
    actions = [e["action"] for e in log]
    for a in ("model.registered", "promotion.requested", "promotion.approved", "model.stage_changed", "model.archived",
              "model.promotion_blocked", "model.downloaded", "member.role_set"):
        assert a in actions, a
    assert client.get("/api/audit", headers=H(t["alice"])).status_code == 403
    assert client.get("/api/audit/verify", headers=H(t["root"])).json()["valid"]
    from sqlalchemy import create_engine, text
    with create_engine(url).begin() as conn:
        conn.execute(text("UPDATE audit_log SET actor = 'someone-else' WHERE action = 'promotion.approved'"))
    assert client.get("/api/audit/verify", headers=H(t["root"])).json()["valid"] is False
    export = client.get("/api/audit/export", headers=H(t["root"])).text.strip().splitlines()
    assert len(export) >= len(log) and json.loads(export[0])["prev_hash"] == "0" * 64


def test_monitoring_results(hub):
    client, t, H, _ = hub
    body = {"version": 1, "rows": 500, "drift_share": 0.4, "metric": "roc_auc", "metric_value": 0.7,
            "alerts": [{"kind": "data_drift"}], "retrain_recommended": True}
    assert client.post("/api/workspaces/churn/models/m/monitoring", json=body, headers=H(t["alice"])).status_code == 201
    rows = client.get("/api/workspaces/churn/models/m/monitoring", headers=H(t["carol"])).json()
    assert rows[0]["retrain_recommended"] and rows[0]["drift_share"] == 0.4


# ── OIDC single sign-on ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def idp():
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    nums = key.public_key().public_numbers()
    b64 = lambda n: base64.urlsafe_b64encode(n.to_bytes((n.bit_length() + 7) // 8, "big")).rstrip(b"=").decode()  # noqa: E731
    jwks = json.dumps({"keys": [{"kty": "RSA", "kid": "k1", "use": "sig", "alg": "RS256", "n": b64(nums.n), "e": b64(nums.e)}]}).encode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(jwks)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield key, f"http://127.0.0.1:{httpd.server_port}/jwks"
    httpd.shutdown()


def test_oidc_login_provisions_users_and_maps_groups(tmp_path, idp):
    import jwt

    from automatedcleaning.hub.security import OIDCVerifier
    key, jwks_url = idp
    url = f"sqlite:///{tmp_path / 'hub.db'}"
    admin = bootstrap_admin(url)
    app = create_app(database_url=url, artifact_root=str(tmp_path / "a"),
                     oidc=OIDCVerifier("https://idp.example.com", "hub", jwks_url),
                     group_roles={"ml-churn": ("churn", "data_scientist"), "ml-admins": ("*", "admin")})
    client = TestClient(app)
    client.post("/api/workspaces", json={"name": "churn"}, headers={"Authorization": f"Bearer {admin}"})

    def token(**over):
        now = int(time.time())
        claims = {"iss": "https://idp.example.com", "aud": "hub", "sub": "u-123", "email": "dana@corp.com",
                  "name": "Dana", "iat": now, "exp": now + 300, "groups": ["ml-churn"], **over}
        return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k1"})
    me = client.get("/api/me", headers={"Authorization": f"Bearer {token()}"}).json()
    assert me["email"] == "dana@corp.com" and me["workspaces"] == {"churn": "data_scientist"} and not me["is_admin"]
    adm = client.get("/api/me", headers={"Authorization": f"Bearer {token(sub='u-9', groups=['ml-admins'])}"}).json()
    assert adm["is_admin"]
    for bad in (token(aud="other"), token(iss="https://evil.example.com"), token(exp=int(time.time()) - 600)):
        assert client.get("/api/me", headers={"Authorization": f"Bearer {bad}"}).status_code == 401
    from cryptography.hazmat.primitives.asymmetric import rsa
    forged = jwt.encode({"iss": "https://idp.example.com", "aud": "hub", "sub": "x", "iat": int(time.time()),
                         "exp": int(time.time()) + 60}, rsa.generate_private_key(public_exponent=65537, key_size=2048),
                        algorithm="RS256", headers={"kid": "k1"})
    assert client.get("/api/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


# ── the Python client against a real server ─────────────────────────────────

def test_hub_client_end_to_end(tmp_path, model_and_data):
    import uvicorn

    from automatedcleaning.hub import HubClient, HubError
    model, df = model_and_data
    url = f"sqlite:///{tmp_path / 'hub.db'}"
    admin = bootstrap_admin(url)
    app = create_app(database_url=url, artifact_root=str(tmp_path / "art"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        hub = HubClient(f"http://127.0.0.1:{port}", admin)
        hub._request("POST", "/api/workspaces", body={"name": "churn", "require_approval": False})
        run = hub.start_run("churn", experiment="weekly", name="automl", params={"budget": 20})
        hub.log(run, metrics={"test_roc_auc": 0.91})
        hub.end_run(run)
        v = hub.register("churn", "churn-model", model, run_id=run["id"], metrics={"test_roc_auc": 0.91})
        assert hub.promote("churn", "churn-model", v["version"], "production")["status"] == "applied"
        loaded = hub.load_model("churn", "churn-model", stage="production")
        assert loaded.predict(df).to_list() == model.predict(df).to_list()
        mon = ac.Monitor(model, df.tail(100))
        mon.on_check(lambda r: hub.report_monitoring("churn", "churn-model", r, version=v["version"]))
        mon.check(df.head(100))
        rows = hub._request("GET", "/api/workspaces/churn/models/churn-model/monitoring")
        assert len(rows) == 1 and rows[0]["rows"] == 100
        with pytest.raises(HubError) as err:
            HubClient(f"http://127.0.0.1:{port}", "ach_nope").me()
        assert err.value.status == 401
        with pytest.raises(ValueError, match="https"):
            HubClient("http://hub.example.com", admin)
    finally:
        server.should_exit = True
        th.join(timeout=10)
