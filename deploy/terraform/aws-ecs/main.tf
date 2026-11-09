# A model API (the AutomatedCleaning container, port 8000) on ECS Fargate behind an
# Application Load Balancer: private tasks, the API key injected from Secrets Manager,
# rolling deploys with a circuit breaker that rolls back automatically, CPU + request-count
# autoscaling, logs and alarms.
#
#   terraform init && terraform apply -var name=churn -var image_uri=<ecr uri> \
#     -var vpc_id=vpc-… -var 'public_subnet_ids=[…]' -var 'private_subnet_ids=[…]'

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = ">= 6.0, < 7.0" }
  }
}

locals {
  tags  = merge({ "managed-by" = "automatedcleaning", "model" = var.name }, var.tags)
  https = var.certificate_arn != null
  environment = merge({ AC_MODEL_NAME = var.name, AC_MODEL_VERSION = var.model_version, AC_LOG_LEVEL = "INFO" },
  var.environment)
}

data "aws_region" "current" {}

resource "aws_cloudwatch_log_group" "this" {
  name              = "/automatedcleaning/${var.name}"
  retention_in_days = var.log_retention_days
  kms_key_id        = var.kms_key_arn
  tags              = local.tags
}

resource "aws_ecs_cluster" "this" {
  name = var.name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = local.tags
}

# ── IAM ────────────────────────────────────────────────────────────────────
data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name_prefix        = substr("${var.name}-exec-", 0, 32)
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "read_api_key" {
  count = var.api_key_secret_arn == null ? 0 : 1
  role  = aws_iam_role.execution.id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = [var.api_key_secret_arn] }]
  })
}

resource "aws_iam_role" "task" {
  name_prefix        = substr("${var.name}-task-", 0, 32)
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
  tags               = local.tags
}

# ── networking ─────────────────────────────────────────────────────────────
resource "aws_security_group" "alb" {
  name_prefix = "${var.name}-alb-"
  vpc_id      = var.vpc_id
  tags        = local.tags
  lifecycle { create_before_destroy = true }
}

resource "aws_vpc_security_group_ingress_rule" "alb" {
  for_each          = toset([for p in(local.https ? [80, 443] : [80]) : tostring(p)])
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = var.allowed_cidr
  from_port         = tonumber(each.value)
  to_port           = tonumber(each.value)
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "alb" {
  security_group_id            = aws_security_group.alb.id
  referenced_security_group_id = aws_security_group.task.id
  from_port                    = var.container_port
  to_port                      = var.container_port
  ip_protocol                  = "tcp"
}

resource "aws_security_group" "task" {
  name_prefix = "${var.name}-task-"
  vpc_id      = var.vpc_id
  tags        = local.tags
  lifecycle { create_before_destroy = true }
}

resource "aws_vpc_security_group_ingress_rule" "task" {
  security_group_id            = aws_security_group.task.id
  referenced_security_group_id = aws_security_group.alb.id
  from_port                    = var.container_port
  to_port                      = var.container_port
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "task" {
  security_group_id = aws_security_group.task.id
  cidr_ipv4         = "0.0.0.0/0" # ECR, Secrets Manager, CloudWatch (use VPC endpoints to keep it private)
  ip_protocol       = "-1"
}

resource "aws_lb" "this" {
  name_prefix                = substr(var.name, 0, 6)
  load_balancer_type         = "application"
  internal                   = var.internal
  subnets                    = var.public_subnet_ids
  security_groups            = [aws_security_group.alb.id]
  drop_invalid_header_fields = true
  tags                       = local.tags
}

resource "aws_lb_target_group" "this" {
  name_prefix          = substr(var.name, 0, 6)
  port                 = var.container_port
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = var.vpc_id
  deregistration_delay = 30
  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 15
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
  tags = local.tags
  lifecycle { create_before_destroy = true }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = local.https ? "redirect" : "forward"
    target_group_arn = local.https ? null : aws_lb_target_group.this.arn
    dynamic "redirect" {
      for_each = local.https ? [1] : []
      content {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  }
}

resource "aws_lb_listener" "https" {
  count             = local.https ? 1 : 0
  load_balancer_arn = aws_lb.this.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.this.arn
  }
}

# ── service ────────────────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "this" {
  family                   = var.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = var.cpu_architecture
  }
  container_definitions = jsonencode([{
    name                   = "model"
    image                  = var.image_uri
    essential              = true
    readonlyRootFilesystem = false
    portMappings           = [{ containerPort = var.container_port, protocol = "tcp" }]
    environment            = [for k, v in local.environment : { name = k, value = v }]
    secrets                = var.api_key_secret_arn == null ? [] : [{ name = "AC_API_KEY", valueFrom = var.api_key_secret_arn }]
    logConfiguration = {
      logDriver = "awslogs"
      options = { "awslogs-group" = aws_cloudwatch_log_group.this.name, "awslogs-region" = data.aws_region.current.region,
      "awslogs-stream-prefix" = "model" }
    }
  }])
  tags = local.tags
}

resource "aws_ecs_service" "this" {
  name                               = var.name
  cluster                            = aws_ecs_cluster.this.id
  task_definition                    = aws_ecs_task_definition.this.arn
  desired_count                      = var.desired_count
  launch_type                        = "FARGATE"
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  health_check_grace_period_seconds  = 60
  enable_execute_command             = false
  propagate_tags                     = "SERVICE"
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.task.id]
    assign_public_ip = false
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.this.arn
    container_name   = "model"
    container_port   = var.container_port
  }
  lifecycle { ignore_changes = [desired_count] } # owned by autoscaling
  depends_on = [aws_lb_listener.http]
  tags       = local.tags
}

resource "aws_appautoscaling_target" "this" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.this.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.min_tasks
  max_capacity       = var.max_tasks
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "${var.name}-cpu"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.this.service_namespace
  resource_id        = aws_appautoscaling_target.this.resource_id
  scalable_dimension = aws_appautoscaling_target.this.scalable_dimension
  target_tracking_scaling_policy_configuration {
    target_value = var.cpu_target_percent
    predefined_metric_specification { predefined_metric_type = "ECSServiceAverageCPUUtilization" }
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}

resource "aws_appautoscaling_policy" "requests" {
  name               = "${var.name}-requests"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.this.service_namespace
  resource_id        = aws_appautoscaling_target.this.resource_id
  scalable_dimension = aws_appautoscaling_target.this.scalable_dimension
  target_tracking_scaling_policy_configuration {
    target_value = var.requests_per_task_target
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${aws_lb.this.arn_suffix}/${aws_lb_target_group.this.arn_suffix}"
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "target_5xx" {
  alarm_name          = "${var.name}-5xx"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  dimensions          = { LoadBalancer = aws_lb.this.arn_suffix, TargetGroup = aws_lb_target_group.this.arn_suffix }
  statistic           = "Sum"
  period              = 60
  evaluation_periods  = 3
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = var.alarm_topic_arn == null ? [] : [var.alarm_topic_arn]
  tags                = local.tags
}
