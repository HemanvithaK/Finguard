# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---- VPC (Virtual Private Cloud) ----
# A private network in AWS where your containers run.
# Think of it as your own isolated section of AWS's data center.
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "finguard-vpc" }
}

# Public subnets in two availability zones (AWS requires at least 2 for load balancers)
resource "aws_subnet" "public_a" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = { Name = "finguard-public-a" }
}

resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true

  tags = { Name = "finguard-public-b" }
}

# Internet gateway — lets your VPC talk to the internet
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "finguard-igw" }
}

# Route table — tells traffic how to reach the internet
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "finguard-public-rt" }
}

resource "aws_route_table_association" "public_a" {
  subnet_id      = aws_subnet.public_a.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

# ---- Security Group ----
# Firewall rules: allow HTTP traffic in, allow all traffic out
resource "aws_security_group" "ecs" {
  name   = "finguard-ecs-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "finguard-ecs-sg" }
}

# ---- ECR (Elastic Container Registry) ----
# AWS's Docker Hub equivalent — stores your container image
resource "aws_ecr_repository" "finguard" {
  name                 = "finguard"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  tags = { Name = "finguard-ecr" }
}

# ---- ECS Cluster ----
# A logical grouping of your containers
resource "aws_ecs_cluster" "main" {
  name = "finguard-cluster"
  tags = { Name = "finguard-cluster" }
}

# ---- IAM Role for ECS Tasks ----
# Permissions your container needs to run (pull images, write logs)
resource "aws_iam_role" "ecs_task_execution" {
  name = "finguard-ecs-task-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ---- CloudWatch Log Group ----
# Where your container's stdout/stderr goes — viewable in AWS Console
resource "aws_cloudwatch_log_group" "finguard" {
  name              = "/ecs/finguard"
  retention_in_days = 7

  tags = { Name = "finguard-logs" }
}

# ---- ECS Task Definition ----
# Describes your container: what image to run, how much CPU/memory,
# environment variables, and where to send logs
resource "aws_ecs_task_definition" "finguard" {
  family                   = "finguard"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"    # 0.5 vCPU (free tier eligible)
  memory                   = "1024"   # 1 GB RAM
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn

  container_definitions = jsonencode([{
    name  = "finguard"
    image = "${aws_ecr_repository.finguard.repository_url}:latest"

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    environment = [
      { name = "ANTHROPIC_API_KEY", value = var.anthropic_api_key }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.finguard.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

# ---- Application Load Balancer ----
# Distributes incoming traffic to your containers
resource "aws_lb" "main" {
  name               = "finguard-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.ecs.id]
  subnets            = [aws_subnet.public_a.id, aws_subnet.public_b.id]

  tags = { Name = "finguard-alb" }
}

resource "aws_lb_target_group" "finguard" {
  name        = "finguard-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.finguard.arn
  }
}

# ---- ECS Service ----
# Keeps your container running and connects it to the load balancer
resource "aws_ecs_service" "finguard" {
  name            = "finguard-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.finguard.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [aws_subnet.public_a.id, aws_subnet.public_b.id]
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.finguard.arn
    container_name   = "finguard"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.http]
}