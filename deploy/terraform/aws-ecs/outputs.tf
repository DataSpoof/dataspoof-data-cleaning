output "url" {
  value = "${local.https ? "https" : "http"}://${aws_lb.this.dns_name}"
}

output "cluster" {
  value = aws_ecs_cluster.this.name
}

output "service" {
  value = aws_ecs_service.this.name
}

output "log_group" {
  value = aws_cloudwatch_log_group.this.name
}
