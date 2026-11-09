"""Secret references: env://, file://, aws-sm:// (moto), caching and safe errors.
Vault is tested against a real server in test_docker_stacks.py."""

import importlib.util
import json

import pytest

from mlpilot import secretstore as ss


@pytest.fixture(autouse=True)
def fresh():
    ss.clear_cache()
    yield
    ss.clear_cache()


def test_env_and_file_references(tmp_path, monkeypatch):
    monkeypatch.setenv("REAL_DB_URL", "postgresql://u:pw@db/x")
    assert ss.get_secret("env://REAL_DB_URL") == "postgresql://u:pw@db/x"
    f = tmp_path / "api_key"
    f.write_text("k-42\n")
    assert ss.get_secret(f.as_uri()) == "k-42"                          # trailing newline stripped
    assert ss.resolve("not-a-ref") == "not-a-ref" and ss.is_reference("vault://a/b#c")
    with pytest.raises(ss.SecretError, match="not set"):
        ss.get_secret("env://MISSING_VAR_XYZ")
    with pytest.raises(ss.SecretError, match="Not a secret reference"):
        ss.get_secret("http://example.com")


def test_cache_respects_ttl(monkeypatch):
    monkeypatch.setenv("ROTATING", "v1")
    assert ss.get_secret("env://ROTATING") == "v1"
    monkeypatch.setenv("ROTATING", "v2")
    assert ss.get_secret("env://ROTATING") == "v1"                      # cached
    assert ss.get_secret("env://ROTATING", ttl=0) == "v2"               # bypass
    ss.clear_cache()
    assert ss.get_secret("env://ROTATING") == "v2"


def test_vault_errors_without_credentials(monkeypatch):
    monkeypatch.delenv("VAULT_ADDR", raising=False)
    with pytest.raises(ss.SecretError, match="VAULT_ADDR"):
        ss.get_secret("vault://secret/x#y")
    monkeypatch.setenv("VAULT_ADDR", "http://127.0.0.1:9")
    for v in ("VAULT_TOKEN", "VAULT_ROLE_ID", "VAULT_SECRET_ID", "VAULT_K8S_ROLE"):
        monkeypatch.delenv(v, raising=False)
    ss.clear_cache()
    with pytest.raises(ss.SecretError, match="No Vault credentials"):
        ss.get_secret("vault://secret/x#y")


@pytest.mark.skipif(importlib.util.find_spec("moto") is None, reason="needs moto")
def test_aws_secrets_manager(monkeypatch):
    import boto3
    from moto import mock_aws
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        sm = boto3.client("secretsmanager")
        sm.create_secret(Name="prod/hub", SecretString=json.dumps({"database_url": "postgresql://h", "port": 5432}))
        sm.create_secret(Name="prod/token", SecretString="plain-token")
        assert ss.get_secret("aws-sm://prod/hub#database_url") == "postgresql://h"
        assert ss.get_secret("aws-sm://prod/hub#port") == "5432"
        assert ss.get_secret("aws-sm://prod/token") == "plain-token"
        with pytest.raises(ss.SecretError, match="no key"):
            ss.get_secret("aws-sm://prod/hub#nope")
        with pytest.raises(ss.SecretError, match="can't read") as e:
            ss.get_secret("aws-sm://missing/one#x")
        assert "postgresql" not in str(e.value)
