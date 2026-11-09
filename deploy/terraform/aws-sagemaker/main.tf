# A model served by an MLPilot container (`mlpilot docker --target sagemaker`)
# as an Amazon SageMaker endpoint: IAM role, model, endpoint config, endpoint with blue/green
# deployment + automatic rollback on alarms, autoscaling, data capture and CloudWatch alarms.
#
#   terraform init && terraform apply -var name=churn -var image_uri=<ecr uri>

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws    = { source = "hashicorp/aws", version = ">= 6.0, < 7.0" }
    random = { source = "hashicorp/random", version = ">= 3.5" }
  }
}

locals {
  serverless = var.serverless_memory_mb != null
  tags       = merge({ "managed-by" = "mlpilot", "model" = var.name }, var.tags)
  environment = merge(
    { AC_MODEL_NAME = var.name, AC_MODEL_VERSION = var.model_version },
    var.api_key_secret_id == null ? {} : { AC_API_KEY = "aws-sm://${var.api_key_secret_id}" },
    var.environment,
  )
}

data "aws_partition" "current" {}
data "aws_region" "current" {}

# ── IAM ────────────────────────────────────────────────────────────────────
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "endpoint" {
  name_prefix        = substr("${var.name}-sm-", 0, 32)
  assume_role_policy = data.aws_iam_policy_document.assume.json
  tags               = local.tags
}

data "aws_iam_policy_document" "endpoint" {
  statement {
    sid       = "PullImage"
    actions   = ["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = ["*"]
  }
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "cloudwatch:PutMetricData"]
    resources = ["*"]
  }
  dynamic "statement" {
    for_each = var.data_capture_s3_uri == null ? [] : [1]
    content {
      sid       = "DataCapture"
      actions   = ["s3:PutObject", "s3:AbortMultipartUpload"]
      resources = ["arn:${data.aws_partition.current.partition}:s3:::${split("/", replace(var.data_capture_s3_uri, "s3://", ""))[0]}/*"]
    }
  }
  dynamic "statement" {
    for_each = var.api_key_secret_id == null ? [] : [1]
    content {
      sid       = "ReadApiKey"
      actions   = ["secretsmanager:GetSecretValue"]
      resources = [var.api_key_secret_arn != null ? var.api_key_secret_arn : "*"]
    }
  }
  dynamic "statement" {
    for_each = length(var.subnet_ids) == 0 ? [] : [1]
    content {
      sid = "Vpc"
      actions = ["ec2:CreateNetworkInterface", "ec2:CreateNetworkInterfacePermission", "ec2:DeleteNetworkInterface",
        "ec2:DeleteNetworkInterfacePermission", "ec2:DescribeNetworkInterfaces", "ec2:DescribeVpcs",
      "ec2:DescribeDhcpOptions", "ec2:DescribeSubnets", "ec2:DescribeSecurityGroups"]
      resources = ["*"]
    }
  }
}

resource "aws_iam_role_policy" "endpoint" {
  role   = aws_iam_role.endpoint.id
  policy = data.aws_iam_policy_document.endpoint.json
}

# ── model, config, endpoint ────────────────────────────────────────────────
# A new suffix whenever the image or settings change: a new model + config are created first,
# then the endpoint switches to them (blue/green), then the old ones are destroyed.
resource "random_id" "release" {
  byte_length = 4
  keepers = {
    image       = var.image_uri
    environment = jsonencode(local.environment)
    variant     = jsonencode([var.instance_type, var.instance_count, var.serverless_memory_mb, var.serverless_max_concurrency])
    capture     = jsonencode([var.data_capture_s3_uri, var.data_capture_percentage])
    network     = jsonencode([var.subnet_ids, var.security_group_ids])
  }
}

resource "aws_sagemaker_model" "this" {
  name               = "${substr(var.name, 0, 50)}-${random_id.release.hex}"
  execution_role_arn = aws_iam_role.endpoint.arn
  primary_container {
    image       = var.image_uri
    environment = local.environment
  }
  dynamic "vpc_config" {
    for_each = length(var.subnet_ids) == 0 ? [] : [1]
    content {
      subnets            = var.subnet_ids
      security_group_ids = var.security_group_ids
    }
  }
  tags       = local.tags
  depends_on = [aws_iam_role_policy.endpoint]
  lifecycle { create_before_destroy = true }
}

resource "aws_sagemaker_endpoint_configuration" "this" {
  name        = "${substr(var.name, 0, 50)}-${random_id.release.hex}"
  kms_key_arn = var.kms_key_arn
  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.this.name
    initial_variant_weight = 1
    instance_type          = local.serverless ? null : var.instance_type
    initial_instance_count = local.serverless ? null : var.instance_count
    dynamic "serverless_config" {
      for_each = local.serverless ? [1] : []
      content {
        memory_size_in_mb = var.serverless_memory_mb
        max_concurrency   = var.serverless_max_concurrency
      }
    }
  }
  dynamic "data_capture_config" {
    for_each = var.data_capture_s3_uri == null ? [] : [1]
    content {
      enable_capture              = true
      initial_sampling_percentage = var.data_capture_percentage
      destination_s3_uri          = var.data_capture_s3_uri
      capture_options { capture_mode = "Input" }
      capture_options { capture_mode = "Output" }
      capture_content_type_header { json_content_types = ["application/json"] }
    }
  }
  tags = local.tags
  lifecycle { create_before_destroy = true }
}

resource "aws_sagemaker_endpoint" "this" {
  name                 = var.name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.this.name
  dynamic "deployment_config" {
    for_each = local.serverless ? [] : [1]
    content {
      blue_green_update_policy {
        traffic_routing_configuration {
          type                     = "ALL_AT_ONCE"
          wait_interval_in_seconds = var.bake_seconds
        }
        termination_wait_in_seconds = 300
      }
      auto_rollback_configuration {
        alarms { alarm_name = aws_cloudwatch_metric_alarm.errors.alarm_name }
        alarms { alarm_name = aws_cloudwatch_metric_alarm.latency.alarm_name }
      }
    }
  }
  tags = local.tags
}

# ── autoscaling ────────────────────────────────────────────────────────────
resource "aws_appautoscaling_target" "this" {
  count              = local.serverless ? 0 : 1
  service_namespace  = "sagemaker"
  resource_id        = "endpoint/${aws_sagemaker_endpoint.this.name}/variant/primary"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  min_capacity       = var.min_instances
  max_capacity       = var.max_instances
}

resource "aws_appautoscaling_policy" "invocations" {
  count              = local.serverless ? 0 : 1
  name               = "${var.name}-invocations"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.this[0].service_namespace
  resource_id        = aws_appautoscaling_target.this[0].resource_id
  scalable_dimension = aws_appautoscaling_target.this[0].scalable_dimension
  target_tracking_scaling_policy_configuration {
    target_value       = var.target_invocations_per_instance
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
    predefined_metric_specification {
      predefined_metric_type = "SageMakerVariantInvocationsPerInstance"
    }
  }
}

# ── alarms (also drive automatic rollback of a bad deployment) ─────────────
resource "aws_cloudwatch_metric_alarm" "errors" {
  alarm_name          = "${var.name}-5xx"
  namespace           = "AWS/SageMaker"
  metric_name         = "Invocation5XXErrors"
  dimensions          = { EndpointName = var.name, VariantName = "primary" }
  statistic           = "Sum"
  period              = 60
  evaluation_periods  = 3
  threshold           = var.alarm_5xx_per_minute
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_topic_arn == null ? [] : [var.alarm_topic_arn]
  tags                = local.tags
}

resource "aws_cloudwatch_metric_alarm" "latency" {
  alarm_name          = "${var.name}-latency-p95"
  namespace           = "AWS/SageMaker"
  metric_name         = "ModelLatency"
  dimensions          = { EndpointName = var.name, VariantName = "primary" }
  extended_statistic  = "p95"
  period              = 60
  evaluation_periods  = 5
  threshold           = var.alarm_p95_latency_ms * 1000 # ModelLatency is in microseconds
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_topic_arn == null ? [] : [var.alarm_topic_arn]
  tags                = local.tags
}
