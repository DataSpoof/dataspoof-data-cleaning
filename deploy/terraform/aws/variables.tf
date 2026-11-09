variable "name" {
  description = "Name prefix for every resource (also the Kubernetes release / service account name)."
  type        = string
  default     = "ac-hub"
}

variable "region" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "private_subnet_ids" {
  description = "Private subnets for the database (at least two AZs)."
  type        = list(string)
}

variable "eks_cluster_name" {
  type = string
}

variable "eks_oidc_provider_arn" {
  description = "ARN of the EKS cluster's IAM OIDC provider (for IRSA)."
  type        = string
}

variable "namespace" {
  type    = string
  default = "automatedcleaning"
}

variable "artifact_bucket_name" {
  type = string
}

variable "image_repository" {
  type    = string
  default = "ghcr.io/dataspoof/automatedcleaning-hub"
}

variable "image_tag" {
  type    = string
  default = "2.0.0"
}

variable "replicas" {
  type    = number
  default = 2
}

variable "postgres_version" {
  type    = string
  default = "16"
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.medium"
}

variable "db_storage_gb" {
  type    = number
  default = 50
}

variable "db_multi_az" {
  type    = bool
  default = true
}

variable "oidc_issuer" {
  description = "SSO issuer URL (Okta, Entra ID, Google, Keycloak). Empty = API tokens only."
  type        = string
  default     = ""
}

variable "oidc_audience" {
  type    = string
  default = "automatedcleaning-hub"
}

variable "oidc_group_roles" {
  description = "Identity-provider group -> [workspace, role], e.g. {\"ml-admins\" = [\"*\", \"admin\"]}."
  type        = map(list(string))
  default     = {}
}

variable "hostname" {
  description = "Public hostname for the ingress; empty = no ingress."
  type        = string
  default     = ""
}

variable "ingress_class" {
  type    = string
  default = "alb"
}

variable "ingress_annotations" {
  type    = map(string)
  default = {}
}

variable "tags" {
  type    = map(string)
  default = {}
}
