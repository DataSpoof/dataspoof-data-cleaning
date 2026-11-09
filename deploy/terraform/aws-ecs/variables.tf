variable "name" {
  type = string
  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{0,30}$", var.name))
    error_message = "name: lowercase letters, digits and hyphens, at most 31 characters."
  }
}

variable "image_uri" {
  description = "ECR image built with `mlpilot docker` (aws.push_image)."
  type        = string
}

variable "model_version" {
  type    = string
  default = ""
}

variable "vpc_id" {
  type = string
}

variable "public_subnet_ids" {
  description = "Subnets for the load balancer (private ones if internal = true)."
  type        = list(string)
}

variable "private_subnet_ids" {
  description = "Subnets for the tasks (no public IPs)."
  type        = list(string)
}

variable "internal" {
  description = "An internal load balancer (reachable only inside the VPC)."
  type        = bool
  default     = false
}

variable "allowed_cidr" {
  description = "Who may reach the load balancer."
  type        = string
  default     = "0.0.0.0/0"
}

variable "certificate_arn" {
  description = "ACM certificate: enables HTTPS and redirects HTTP to it."
  type        = string
  default     = null
}

variable "container_port" {
  type    = number
  default = 8000
}

variable "cpu" {
  type    = number
  default = 1024
}

variable "memory" {
  type    = number
  default = 2048
}

variable "cpu_architecture" {
  type    = string
  default = "X86_64"
}

variable "desired_count" {
  type    = number
  default = 2
}

variable "min_tasks" {
  type    = number
  default = 2
}

variable "max_tasks" {
  type    = number
  default = 10
}

variable "cpu_target_percent" {
  type    = number
  default = 60
}

variable "requests_per_task_target" {
  type    = number
  default = 500
}

variable "api_key_secret_arn" {
  description = "Secrets Manager secret holding the API key (injected as AC_API_KEY)."
  type        = string
  default     = null
}

variable "environment" {
  description = "Extra AC_* settings (values may be aws-sm:// / vault:// references)."
  type        = map(string)
  default     = {}
}

variable "log_retention_days" {
  type    = number
  default = 30
}

variable "kms_key_arn" {
  type    = string
  default = null
}

variable "alarm_topic_arn" {
  type    = string
  default = null
}

variable "tags" {
  type    = map(string)
  default = {}
}
