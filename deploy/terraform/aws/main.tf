# AutomatedCleaning Hub on AWS: S3 artifacts + RDS PostgreSQL + IRSA, deployed to an existing EKS cluster.
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.0" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
    helm   = { source = "hashicorp/helm", version = "~> 2.13" }
  }
}

provider "aws" {
  region = var.region
}

data "aws_eks_cluster" "this" {
  name = var.eks_cluster_name
}

data "aws_eks_cluster_auth" "this" {
  name = var.eks_cluster_name
}

provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.this.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.this.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.this.token
  }
}

locals {
  tags = merge({ "app" = "automatedcleaning-hub", "managed-by" = "terraform" }, var.tags)
}

# ── artifact bucket: private, encrypted, versioned ────────────────────────────
resource "aws_s3_bucket" "artifacts" {
  bucket        = var.artifact_bucket_name
  force_destroy = false
  tags          = local.tags
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "tls_only" {
  bucket = aws_s3_bucket.artifacts.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

# ── PostgreSQL: private, encrypted, password in Secrets Manager ───────────────
resource "random_password" "db" {
  length  = 32
  special = false
}

resource "aws_db_subnet_group" "hub" {
  name       = "${var.name}-db"
  subnet_ids = var.private_subnet_ids
  tags       = local.tags
}

resource "aws_security_group" "db" {
  name        = "${var.name}-db"
  description = "PostgreSQL for AutomatedCleaning Hub (from the EKS cluster only)"
  vpc_id      = var.vpc_id
  ingress {
    description     = "PostgreSQL from EKS nodes"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [data.aws_eks_cluster.this.vpc_config[0].cluster_security_group_id]
  }
  tags = local.tags
}

resource "aws_db_instance" "hub" {
  identifier                   = "${var.name}-db"
  engine                       = "postgres"
  engine_version               = var.postgres_version
  instance_class               = var.db_instance_class
  allocated_storage            = var.db_storage_gb
  max_allocated_storage        = var.db_storage_gb * 5
  db_name                      = "hub"
  username                     = "hub"
  password                     = random_password.db.result
  db_subnet_group_name         = aws_db_subnet_group.hub.name
  vpc_security_group_ids       = [aws_security_group.db.id]
  storage_encrypted            = true
  publicly_accessible          = false
  multi_az                     = var.db_multi_az
  backup_retention_period      = 14
  deletion_protection          = true
  skip_final_snapshot          = false
  final_snapshot_identifier    = "${var.name}-db-final"
  performance_insights_enabled = true
  auto_minor_version_upgrade   = true
  tags                         = local.tags
}

resource "aws_secretsmanager_secret" "db_url" {
  name = "${var.name}/database-url"
  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "db_url" {
  secret_id     = aws_secretsmanager_secret.db_url.id
  secret_string = "postgresql+psycopg://hub:${random_password.db.result}@${aws_db_instance.hub.address}:5432/hub?sslmode=require"
}

# ── IRSA: the Hub's service account may read/write only its own artifact prefix ──
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [var.eks_oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${replace(var.eks_oidc_provider_arn, "/^.*oidc-provider\\//", "")}:sub"
      values   = ["system:serviceaccount:${var.namespace}:${var.name}"]
    }
  }
}

resource "aws_iam_role" "hub" {
  name               = "${var.name}-hub"
  assume_role_policy = data.aws_iam_policy_document.assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy" "artifacts" {
  name = "artifacts"
  role = aws_iam_role.hub.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:ListBucket"], Resource = aws_s3_bucket.artifacts.arn,
      Condition = { StringLike = { "s3:prefix" = ["hub/*"] } } },
      { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = "${aws_s3_bucket.artifacts.arn}/hub/*" }
    ]
  })
}

# ── the Hub itself ────────────────────────────────────────────────────────────
resource "helm_release" "hub" {
  name             = var.name
  namespace        = var.namespace
  create_namespace = true
  chart            = "${path.module}/../../helm/automatedcleaning-hub"
  values = [yamlencode({
    image          = { repository = var.image_repository, tag = var.image_tag }
    replicas       = var.replicas
    database       = { url = aws_secretsmanager_secret_version.db_url.secret_string }
    artifacts      = { root = "s3://${aws_s3_bucket.artifacts.bucket}/hub" }
    serviceAccount = { create = true, name = var.name, annotations = { "eks.amazonaws.com/role-arn" = aws_iam_role.hub.arn } }
    oidc           = { issuer = var.oidc_issuer, audience = var.oidc_audience, groupRoles = var.oidc_group_roles }
    ingress = {
      enabled     = var.hostname != ""
      className   = var.ingress_class
      host        = var.hostname
      annotations = var.ingress_annotations
      tls         = { enabled = true, secretName = "${var.name}-tls" }
    }
  })]
}
