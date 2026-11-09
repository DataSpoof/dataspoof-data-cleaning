variable "name" {
  description = "Endpoint name (letters, digits, hyphens)."
  type        = string
  validation {
    condition     = can(regex("^[A-Za-z0-9][A-Za-z0-9-]{0,62}$", var.name))
    error_message = "name: letters, digits and hyphens, at most 63 characters."
  }
}

variable "image_uri" {
  description = "ECR image built with `mlpilot docker --target sagemaker` (aws.push_image)."
  type        = string
}

variable "model_version" {
  description = "Shown in /metadata, responses and metrics."
  type        = string
  default     = ""
}

variable "instance_type" {
  type    = string
  default = "ml.m5.large"
}

variable "instance_count" {
  type    = number
  default = 2
}

variable "min_instances" {
  type    = number
  default = 2
}

variable "max_instances" {
  type    = number
  default = 8
}

variable "target_invocations_per_instance" {
  description = "Autoscaling target: invocations per instance per minute."
  type        = number
  default     = 300
}

variable "serverless_memory_mb" {
  description = "Set (1024-6144) for a serverless endpoint instead of instances."
  type        = number
  default     = null
}

variable "serverless_max_concurrency" {
  type    = number
  default = 10
}

variable "data_capture_s3_uri" {
  description = "s3://bucket/prefix to capture requests and responses for monitoring (null = off)."
  type        = string
  default     = null
}

variable "data_capture_percentage" {
  type    = number
  default = 100
}

variable "api_key_secret_id" {
  description = "Secrets Manager id (optionally with #key) of the API key; the container resolves it as aws-sm://<id>."
  type        = string
  default     = null
}

variable "api_key_secret_arn" {
  description = "ARN of that secret, to scope the IAM permission (default: any secret)."
  type        = string
  default     = null
}

variable "environment" {
  description = "Extra container environment (AC_* settings; values may be aws-sm:// references)."
  type        = map(string)
  default     = {}
}

variable "subnet_ids" {
  description = "Private subnets to run the model in (empty = SageMaker-managed network)."
  type        = list(string)
  default     = []
}

variable "security_group_ids" {
  type    = list(string)
  default = []
}

variable "kms_key_arn" {
  type    = string
  default = null
}

variable "bake_seconds" {
  description = "Blue/green: how long the new fleet must stay healthy before the old one is removed."
  type        = number
  default     = 600
}

variable "alarm_5xx_per_minute" {
  type    = number
  default = 5
}

variable "alarm_p95_latency_ms" {
  type    = number
  default = 500
}

variable "alarm_topic_arn" {
  description = "SNS topic for alarm notifications (null = alarms only drive rollback)."
  type        = string
  default     = null
}

variable "tags" {
  type    = map(string)
  default = {}
}
