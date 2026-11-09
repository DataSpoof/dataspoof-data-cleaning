output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "database_endpoint" {
  value = aws_db_instance.hub.address
}

output "database_url_secret_arn" {
  description = "Secrets Manager secret holding the database URL."
  value       = aws_secretsmanager_secret.db_url.arn
}

output "hub_role_arn" {
  value = aws_iam_role.hub.arn
}

output "next_step" {
  value = "kubectl -n ${var.namespace} exec deploy/${var.name} -- automatedcleaning hub create-admin --name admin"
}
