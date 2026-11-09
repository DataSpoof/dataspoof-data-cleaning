"""SageMaker deployment and ECR push (boto3 against moto; docker calls recorded, not run)."""

import importlib.util
import subprocess

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("moto") is None, reason="needs moto + boto3")


@pytest.fixture()
def aws(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    from moto import mock_aws
    with mock_aws():
        yield


ROLE = "arn:aws:iam::123456789012:role/sagemaker-churn"
IMAGE = "123456789012.dkr.ecr.us-east-1.amazonaws.com/ml/churn:4"


def test_create_then_blue_green_update_and_prune(aws, monkeypatch):
    import boto3

    from mlpilot.deploy import aws as acaws
    stamps = iter(["20261109010101", "20261109020202", "20261109030303"])
    monkeypatch.setattr(acaws, "_stamp", lambda: next(stamps))
    sm = boto3.client("sagemaker")
    first = acaws.deploy_sagemaker(IMAGE, "churn_prod", role_arn=ROLE, instance_count=2, autoscale=(2, 6),
                                   environment={"AC_API_KEY": "aws-sm://prod/churn#api_key"}, model_version=4,
                                   data_capture_s3="s3://capture/churn/", sagemaker=sm)
    assert first["endpoint"] == "churn-prod" and first["status"] == "InService"
    model = sm.describe_model(ModelName=first["model"])
    assert model["PrimaryContainer"]["Environment"] == {"AC_MODEL_NAME": "churn-prod", "AC_MODEL_VERSION": "4",
                                                        "AC_API_KEY": "aws-sm://prod/churn#api_key"}
    cfg = sm.describe_endpoint_config(EndpointConfigName=first["config"])
    assert cfg["ProductionVariants"][0]["InitialInstanceCount"] == 2
    assert cfg["DataCaptureConfig"]["DestinationS3Uri"] == "s3://capture/churn/"
    aas = boto3.client("application-autoscaling")
    target = aas.describe_scalable_targets(ServiceNamespace="sagemaker")["ScalableTargets"][0]
    assert (target["MinCapacity"], target["MaxCapacity"]) == (2, 6)
    updates = []

    class BlueGreen:
        """moto has no update_endpoint: validate the request against the real API schema, then switch."""
        def __init__(self, client):
            self._c = client

        def __getattr__(self, name):
            return getattr(self._c, name)

        def update_endpoint(self, **params):
            from botocore.validate import validate_parameters
            validate_parameters(params, self._c.meta.service_model.operation_model("UpdateEndpoint").input_shape)
            updates.append(params)
            self._c.delete_endpoint(EndpointName=params["EndpointName"])
            self._c.create_endpoint(EndpointName=params["EndpointName"], EndpointConfigName=params["EndpointConfigName"])
    second = acaws.deploy_sagemaker(IMAGE.replace(":4", ":5"), "churn_prod", role_arn=ROLE, sagemaker=BlueGreen(sm),
                                    rollback_alarms=["churn-prod-5xx"],
                                    keep_configs=1)
    assert sm.describe_endpoint(EndpointName="churn-prod")["EndpointConfigName"] == second["config"]
    policy = updates[0]["DeploymentConfig"]
    assert policy["AutoRollbackConfiguration"] == {"Alarms": [{"AlarmName": "churn-prod-5xx"}]}
    assert policy["BlueGreenUpdatePolicy"]["TrafficRoutingConfiguration"]["Type"] == "ALL_AT_ONCE"
    names = [c["EndpointConfigName"] for c in sm.list_endpoint_configs()["EndpointConfigs"]]
    assert names == [second["config"]]                                        # older config pruned
    serverless = acaws.deploy_sagemaker(IMAGE, "churn-sl", role_arn=ROLE, serverless={"memory_mb": 3072}, sagemaker=sm)
    v = sm.describe_endpoint_config(EndpointConfigName=serverless["config"])["ProductionVariants"][0]
    assert v["ServerlessConfig"]["MemorySizeInMB"] == 3072 and "InstanceType" not in v


def test_push_image_creates_repo_and_runs_docker(aws, monkeypatch):
    import boto3

    from mlpilot.deploy import aws as acaws
    calls = []

    def fake_run(cmd, input=None, capture_output=True, text=True):
        calls.append((cmd, input))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(acaws.subprocess, "run", fake_run)
    uri = acaws.push_image("churn:4", "ml/churn", ecr=boto3.client("ecr"))
    assert uri.endswith("/ml/churn:4")
    repo = boto3.client("ecr").describe_repositories(repositoryNames=["ml/churn"])["repositories"][0]
    assert repo["imageScanningConfiguration"]["scanOnPush"] and repo["imageTagMutability"] == "IMMUTABLE"
    (login, pw), (tag, _), (push, _) = calls
    assert login[:2] == ["docker", "login"] and "--password-stdin" in login and pw    # password never on argv
    assert tag == ["docker", "tag", "churn:4", uri] and push == ["docker", "push", uri]

    def failing(cmd, input=None, capture_output=True, text=True):
        return subprocess.CompletedProcess(cmd, 1, "", "denied: not authorized")
    monkeypatch.setattr(acaws.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="denied"):
        acaws.push_image("churn:4", "ml/churn", ecr=boto3.client("ecr"))
