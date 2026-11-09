"""Deploy a model to AWS: push the container to ECR and run it as a SageMaker endpoint.

>>> from automatedcleaning.deploy import aws
>>> ac.build_docker(model, "build/churn", target="sagemaker")          # then: docker build -t churn:4 build/churn
>>> uri = aws.push_image("churn:4", "ml/churn")                         # ECR repo created if missing
>>> aws.deploy_sagemaker(uri, "churn-prod", role_arn="arn:aws:iam::…:role/sagemaker-churn",
...                      instance_type="ml.m5.large", instance_count=2, autoscale=(2, 8),
...                      environment={"AC_API_KEY": "aws-sm://prod/churn#api_key"},
...                      data_capture_s3="s3://ml-capture/churn/")
>>> aws.invoke_sagemaker("churn-prod", [{"tenure": 3, "plan": "basic"}])

* **Blue/green updates**: every deploy creates a new model + endpoint configuration and updates the
  endpoint in place; SageMaker shifts traffic only once the new fleet is healthy, and rolls back if
  it isn't. Old configurations are pruned (``keep_configs``).
* **Serverless** (``serverless={"memory_mb": 2048, "max_concurrency": 10}``) or real-time instances,
  with target-tracking autoscaling on invocations per instance.
* **Data capture** of requests and responses to S3 for monitoring.
* Secrets: environment values may be ``aws-sm://`` references; the container resolves them at
  start-up (give the role ``secretsmanager:GetSecretValue``).

The Terraform modules in ``deploy/terraform/aws-sagemaker`` and ``deploy/terraform/aws-ecs`` create
the same resources declaratively (plus IAM, alarms and, for ECS, a load balancer).
"""

from __future__ import annotations

import base64
import datetime as _dt
import json
import re
import subprocess
import time
from typing import Any

try:
    import boto3
except ImportError as exc:                                   # pragma: no cover
    raise ImportError('AWS deployment needs boto3: pip install "AutomatedCleaning[aws]"') from exc


def _name(s: str) -> str:
    """SageMaker resource names: letters, digits and hyphens, at most 63 characters."""
    return re.sub(r"[^A-Za-z0-9-]", "-", s)[:63].strip("-")


def _stamp() -> str:
    """UTC timestamp suffix for each release's model / config names (sortable)."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d%H%M%S")


def push_image(local_image: str, repository: str, *, tag: str | None = None, ecr=None, docker: str = "docker") -> str:
    """Tag and push a local image to ECR (creating the repository, scan-on-push, if needed).
    Returns the pushed image URI."""
    ecr = ecr or boto3.client("ecr")
    try:
        repo = ecr.describe_repositories(repositoryNames=[repository])["repositories"][0]
    except ecr.exceptions.RepositoryNotFoundException:
        repo = ecr.create_repository(repositoryName=repository, imageScanningConfiguration={"scanOnPush": True},
                                     imageTagMutability="IMMUTABLE")["repository"]
    tag = tag or (local_image.split(":", 1)[1] if ":" in local_image else "latest")
    uri = f"{repo['repositoryUri']}:{tag}"
    auth = ecr.get_authorization_token()["authorizationData"][0]
    user, password = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
    registry = auth["proxyEndpoint"].replace("https://", "")
    for cmd, inp in (([docker, "login", "--username", user, "--password-stdin", registry], password),
                     ([docker, "tag", local_image, uri], None), ([docker, "push", uri], None)):
        p = subprocess.run(cmd, input=inp, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(f"{' '.join(cmd[:2])} failed: {(p.stderr or p.stdout).strip()[-500:]}")
    return uri


def deploy_sagemaker(image_uri: str, endpoint_name: str, *, role_arn: str, instance_type: str = "ml.m5.large",
                     instance_count: int = 1, serverless: dict | None = None, autoscale: tuple | None = None,
                     target_invocations: int = 300, environment: dict | None = None, data_capture_s3: str | None = None,
                     capture_percentage: int = 100, model_name: str | None = None, model_version: str | None = None,
                     tags: dict | None = None, wait: bool = True, timeout: float = 1800, keep_configs: int = 3,
                     vpc: dict | None = None, rollback_alarms: list | None = None, bake_seconds: int = 300,
                     sagemaker=None, autoscaling=None) -> dict:
    """Create or blue/green-update ``endpoint_name`` to serve ``image_uri``. Returns the endpoint
    description (``EndpointStatus`` ``InService`` when ``wait``). ``rollback_alarms``: CloudWatch alarm
    names that roll a blue/green update back if they fire during ``bake_seconds``."""
    sm = sagemaker or boto3.client("sagemaker")
    stamp = _stamp()
    base = _name(endpoint_name)
    sm_model = _name(f"{base}-{stamp}")
    env = {"AC_MODEL_NAME": model_name or base, **({"AC_MODEL_VERSION": str(model_version)} if model_version else {}),
           **{str(k): str(v) for k, v in (environment or {}).items()}}
    tag_list = [{"Key": k, "Value": str(v)} for k, v in {"managed-by": "automatedcleaning", **(tags or {})}.items()]
    model_kw: dict[str, Any] = {"ModelName": sm_model, "ExecutionRoleArn": role_arn, "Tags": tag_list,
                "PrimaryContainer": {"Image": image_uri, "Environment": env}}
    if vpc:
        model_kw["VpcConfig"] = {"Subnets": list(vpc["subnets"]), "SecurityGroupIds": list(vpc["security_groups"])}
    sm.create_model(**model_kw)
    variant: dict[str, Any] = {"VariantName": "primary", "ModelName": sm_model, "InitialVariantWeight": 1.0}
    if serverless:
        variant["ServerlessConfig"] = {"MemorySizeInMB": int(serverless.get("memory_mb", 2048)),
                                       "MaxConcurrency": int(serverless.get("max_concurrency", 10))}
    else:
        variant.update(InstanceType=instance_type, InitialInstanceCount=int(instance_count))
    cfg_name = _name(f"{base}-{stamp}")
    cfg_kw: dict[str, Any] = {"EndpointConfigName": cfg_name, "ProductionVariants": [variant], "Tags": tag_list}
    if data_capture_s3:
        cfg_kw["DataCaptureConfig"] = {"EnableCapture": True, "InitialSamplingPercentage": int(capture_percentage),
                                       "DestinationS3Uri": data_capture_s3,
                                       "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
                                       "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json"]}}
    sm.create_endpoint_config(**cfg_kw)
    try:
        sm.describe_endpoint(EndpointName=base)
        exists = True
    except sm.exceptions.ClientError:
        exists = False
    if exists:
        update_kw: dict = {"EndpointName": base, "EndpointConfigName": cfg_name}
        if not serverless:                            # blue/green (serverless endpoints update in place)
            deploy_cfg: dict = {"BlueGreenUpdatePolicy": {
                "TrafficRoutingConfiguration": {"Type": "ALL_AT_ONCE", "WaitIntervalInSeconds": int(bake_seconds)},
                "TerminationWaitInSeconds": 300}}
            if rollback_alarms:                       # roll back automatically if any alarm fires while baking
                deploy_cfg["AutoRollbackConfiguration"] = {"Alarms": [{"AlarmName": a} for a in rollback_alarms]}
            update_kw["DeploymentConfig"] = deploy_cfg
        sm.update_endpoint(**update_kw)
    else:
        sm.create_endpoint(EndpointName=base, EndpointConfigName=cfg_name, Tags=tag_list)
    desc = _wait(sm, base, timeout) if wait else sm.describe_endpoint(EndpointName=base)
    if autoscale and not serverless:
        lo, hi = autoscale
        aas = autoscaling or boto3.client("application-autoscaling")
        resource = f"endpoint/{base}/variant/primary"
        aas.register_scalable_target(ServiceNamespace="sagemaker", ResourceId=resource,
                                     ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                                     MinCapacity=int(lo), MaxCapacity=int(hi))
        aas.put_scaling_policy(PolicyName=f"{base}-invocations", ServiceNamespace="sagemaker", ResourceId=resource,
                               ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                               PolicyType="TargetTrackingScaling",
                               TargetTrackingScalingPolicyConfiguration={
                                   "TargetValue": float(target_invocations),
                                   "PredefinedMetricSpecification": {
                                       "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"},
                                   "ScaleInCooldown": 300, "ScaleOutCooldown": 60})
    if keep_configs:
        _prune(sm, base, keep_configs, current=cfg_name)
    return {"endpoint": base, "status": desc.get("EndpointStatus"), "config": cfg_name, "model": sm_model,
            "arn": desc.get("EndpointArn"), "image": image_uri}


def _wait(sm, name: str, timeout: float) -> dict:
    t0 = time.time()
    while True:
        d = sm.describe_endpoint(EndpointName=name)
        status = d.get("EndpointStatus")
        if status == "InService":
            return d
        if status in ("Failed", "OutOfService", "RollingBack"):
            raise RuntimeError(f"SageMaker endpoint {name} is {status}: {d.get('FailureReason', '')}".strip())
        if time.time() - t0 > timeout:
            raise RuntimeError(f"SageMaker endpoint {name} still {status} after {timeout:.0f}s")
        time.sleep(15)


def _prune(sm, base: str, keep: int, *, current: str) -> None:
    """Delete this endpoint's older configs and models beyond the newest ``keep``."""
    cfgs = sorted((c for c in sm.list_endpoint_configs(NameContains=base, MaxResults=100)["EndpointConfigs"]
                   if re.fullmatch(rf"{re.escape(base)}-\d{{14}}", c["EndpointConfigName"])),
                  key=lambda c: c["EndpointConfigName"], reverse=True)
    for c in cfgs[keep:]:
        if c["EndpointConfigName"] == current:
            continue
        sm.delete_endpoint_config(EndpointConfigName=c["EndpointConfigName"])
        try:
            sm.delete_model(ModelName=c["EndpointConfigName"])
        except Exception:
            pass


def invoke_sagemaker(endpoint_name: str, records, *, runtime=None) -> dict:
    """Score records through the endpoint (the same JSON contract as the REST API)."""
    rt = runtime or boto3.client("sagemaker-runtime")
    body = json.dumps({"records": records if isinstance(records, list) else [records]}, default=str)
    res = rt.invoke_endpoint(EndpointName=_name(endpoint_name), ContentType="application/json",
                             Accept="application/json", Body=body)
    raw = res["Body"].read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw.decode(errors="replace")}
