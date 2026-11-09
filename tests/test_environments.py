"""Isolated dev -> qa -> prod environments and controlled releases between them."""

import polars as pl
import pytest
from sklearn.datasets import make_classification

import automatedcleaning as ac
from automatedcleaning.environments import Environments, ReleaseError

pytest.importorskip("cryptography")


@pytest.fixture(scope="module")
def model():
    X, y = make_classification(300, 4, n_informative=3, n_redundant=0, random_state=0)
    df = pl.DataFrame(X, schema=list("abcd")).with_columns(pl.Series("y", y))
    return ac.AutoML("y", time_budget=30, cv=3, verbose=False, max_trials=1, include=["logistic_regression"]).fit(df)


@pytest.fixture()
def setup(tmp_path, model, monkeypatch):
    from automatedcleaning.security import generate_keypair
    priv, pub = generate_keypair(tmp_path / "sign.key", tmp_path / "sign.pub")
    monkeypatch.setenv("PROD_SIGNING_PUB", str(pub))
    cfg = tmp_path / "environments.yml"
    cfg.write_text(f"""
order: [dev, qa, prod]
environments:
  dev:  {{registry: "{(tmp_path / 'dev').as_posix()}"}}
  qa:   {{registry: "{(tmp_path / 'qa').as_posix()}", landing_stage: qa, min_metrics: {{test_auc: 0.70}}}}
  prod:
    registry: "{(tmp_path / 'prod').as_posix()}"
    landing_stage: staging
    approvals: 2
    require_signature: true
    public_key: "env://PROD_SIGNING_PUB"
    min_metrics: {{test_auc: 0.75}}
""", encoding="utf-8")
    envs = Environments.from_yaml(cfg)
    dev = envs["dev"].registry
    return envs, dev, priv


def test_release_path_with_gates_approvals_and_signatures(setup, model):
    envs, dev, priv = setup
    v = dev.register(model, "churn", metrics={"test_auc": 0.81}, description="weekly churn", sign_with=priv)
    with pytest.raises(ReleaseError, match="must be in one of"):
        envs.release("churn", v.version, frm="dev", to="qa", released_by="ana")          # still in 'dev'
    dev.promote("churn", v.version, "staging")
    with pytest.raises(ReleaseError, match="can't be skipped"):
        envs.release("churn", v.version, frm="dev", to="prod", released_by="ana")
    qa_v = envs.release("churn", v.version, frm="dev", to="qa", released_by="ana", reason="sprint 14")
    assert qa_v.stage == "qa" and qa_v.tags["source_env"] == "dev" and qa_v.model_sha256 == v.model_sha256
    assert envs["qa"].registry.load("churn", version=qa_v.version).predict  # loads (checksum verified)

    problems = envs.check("churn", qa_v.version, frm="qa", to="prod", released_by="ana", approved_by=["ana", "bob"])
    assert any("2 approval(s)" in p for p in problems)                     # self-approval doesn't count
    prod_v = envs.release("churn", qa_v.version, frm="qa", to="prod", released_by="ana",
                          approved_by=["bob", "carol"], reason="QA sign-off #812")
    assert prod_v.stage == "staging" and prod_v.tags["approved_by"] == "bob,carol"
    status = envs.status("churn")
    assert status["environment"].to_list() == ["dev", "qa", "prod"]
    assert status.filter(pl.col("environment") == "prod")["source"][0] == f"qa v{qa_v.version}"
    rel = envs.releases()
    assert rel.filter(pl.col("event") == "released").height == 4                # recorded on both sides of 2 releases
    assert rel.filter(pl.col("event") == "release_refused").height == 2


def test_metric_gate_and_unsigned_model_are_refused(setup, model):
    envs, dev, priv = setup
    weak = dev.register(model, "churn", metrics={"test_auc": 0.72}, stage="staging", sign_with=priv)
    qa_v = envs.release("churn", weak.version, frm="dev", to="qa", released_by="ana")
    with pytest.raises(ReleaseError, match=r"test_auc = 0.72 \(prod needs >= 0.75\)"):
        envs.release("churn", qa_v.version, frm="qa", to="prod", released_by="ana", approved_by=["bob", "carol"])
    unsigned = dev.register(model, "churn", metrics={"test_auc": 0.9}, stage="staging")
    qa_u = envs.release("churn", unsigned.version, frm="dev", to="qa", released_by="ana")
    with pytest.raises(ReleaseError, match="signature check failed"):
        envs.release("churn", qa_u.version, frm="qa", to="prod", released_by="ana", approved_by=["bob", "carol"])
    assert envs["prod"].registry._versions("churn") == []                     # nothing reached prod


def test_environments_must_be_isolated(tmp_path):
    with pytest.raises(ValueError, match="own registry"):
        Environments({"dev": {"registry": str(tmp_path / "r")}, "prod": {"registry": str(tmp_path / "r")}})
    with pytest.raises(ValueError, match="unknown environments"):
        Environments({"dev": {"registry": str(tmp_path / "a")}}, order=["dev", "prod"])


def test_qa_stage_in_registry(tmp_path, model):
    reg = ac.ModelRegistry(tmp_path / "r")
    reg.register(model, "m")
    reg.promote("m", 1, "qa")
    assert reg.get("m", stage="qa").version == 1 and reg.models()["qa"][0] == 1
