# Deploy on AWS


```python
from automatedcleaning.deploy import aws
uri = aws.push_image("churn:7", "ml/churn")                    # ECR, scan-on-push, immutable tags
aws.deploy_sagemaker(uri, "churn-prod", role_arn=ROLE, instance_count=2, autoscale=(2, 8),
                     environment={"AC_API_KEY": "aws-sm://prod/churn#api_key"},
                     data_capture_s3="s3://ml-capture/churn/", rollback_alarms=["churn-prod-5xx"])
```

```bash
terraform -chdir=deploy/terraform/aws-sagemaker apply -var name=churn -var image_uri=$URI
terraform -chdir=deploy/terraform/aws-ecs apply -var name=churn -var image_uri=$URI -var vpc_id=...
```

**SageMaker:**
- Blue/green updates, rolled back automatically if an alarm fires while the new fleet is baking.
- Serverless or instance-based variants, with autoscaling and request/response data capture.
- IAM with least privilege.

**ECS Fargate:** a load balancer with HTTPS, tasks in private subnets, the API key from Secrets
Manager, a deployment circuit breaker with rollback, and CPU plus request-count autoscaling.

**Tested:**
- Both Terraform modules pass `terraform validate` and `fmt` against AWS provider 6.x.
- The Python deploy path is tested against moto, with requests validated against the real AWS
  API schemas.

> Nothing here has been run against a real AWS account: that needs your account, credentials and
> budget. The modules and the boto3 code are ready to point at one.
