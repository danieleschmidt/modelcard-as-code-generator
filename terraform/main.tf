# Terraform Configuration for Model Card Generator Infrastructure
# Implements enterprise-grade cloud infrastructure with high availability,
# security, monitoring, and disaster recovery capabilities

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    datadog = {
      source  = "DataDog/datadog"
      version = "~> 3.30"
    }
  }
  
  backend "s3" {
    bucket         = "modelcard-generator-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Variables
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_monitoring" {
  description = "Enable comprehensive monitoring"
  type        = bool
  default     = true
}

variable "enable_backup" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "cluster_version" {
  description = "EKS cluster version"
  type        = string
  default     = "1.27"
}

# Locals
locals {
  name_prefix = "modelcard-generator-${var.environment}"
  common_tags = {
    Environment   = var.environment
    Project      = "modelcard-generator"
    ManagedBy    = "terraform"
    Owner        = "terragon-labs"
    CostCenter   = "engineering"
    Compliance   = "required"
  }
}

# Data Sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# VPC and Networking
module "vpc" {
  source = "./modules/vpc"
  
  name               = local.name_prefix
  cidr               = var.vpc_cidr
  availability_zones = var.availability_zones
  
  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # Flow logs for security monitoring
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  
  tags = local.common_tags
}

# Security Groups
resource "aws_security_group" "application" {
  name_prefix = "${local.name_prefix}-app-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for Model Card Generator application"

  # HTTP/HTTPS ingress
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP"
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS"
  }

  # Application port
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Application port"
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-app-sg"
  })
}

# EKS Cluster
module "eks" {
  source = "./modules/eks"
  
  cluster_name    = local.name_prefix
  cluster_version = var.cluster_version
  
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets
  
  # Node groups
  node_groups = {
    general = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 2
      
      instance_types = ["t3.medium", "t3.large"]
      capacity_type  = "ON_DEMAND"
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "general"
      }
      
      additional_tags = local.common_tags
    }
    
    compute = {
      desired_capacity = 2
      max_capacity     = 20
      min_capacity     = 0
      
      instance_types = ["c5.large", "c5.xlarge"]
      capacity_type  = "SPOT"
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "compute"
      }
      
      taints = [
        {
          key    = "compute"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      additional_tags = local.common_tags
    }
  }
  
  # OIDC for service accounts
  enable_irsa = true
  
  # Cluster logging
  cluster_enabled_log_types = [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler"
  ]
  
  tags = local.common_tags
}

# Application Load Balancer
resource "aws_lb" "application" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.application.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = var.environment == "prod"
  
  # Access logs
  access_logs {
    bucket  = aws_s3_bucket.logs.bucket
    prefix  = "alb-access-logs"
    enabled = true
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-alb"
  })
}

# S3 Buckets
resource "aws_s3_bucket" "application_data" {
  bucket = "${local.name_prefix}-app-data"
  tags   = local.common_tags
}

resource "aws_s3_bucket" "logs" {
  bucket = "${local.name_prefix}-logs"
  tags   = local.common_tags
}

resource "aws_s3_bucket" "backups" {
  bucket = "${local.name_prefix}-backups"
  tags   = local.common_tags
}

# S3 Bucket Configurations
resource "aws_s3_bucket_versioning" "application_data" {
  bucket = aws_s3_bucket.application_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "application_data" {
  bucket = aws_s3_bucket.application_data.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "application_data" {
  bucket = aws_s3_bucket.application_data.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# RDS Database
module "database" {
  source = "./modules/rds"
  count  = var.environment == "prod" ? 1 : 0
  
  identifier = local.name_prefix
  
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "db.r6g.large"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_encrypted     = true
  
  # Multi-AZ for production
  multi_az = true
  
  db_name  = "modelcard_generator"
  username = "admin"
  
  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  # Backup configuration
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Monitoring
  monitoring_interval = 60
  
  tags = local.common_tags
}

# Database Security Group
resource "aws_security_group" "database" {
  name_prefix = "${local.name_prefix}-db-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for RDS database"

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.application.id]
    description     = "PostgreSQL from application"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db-sg"
  })
}

resource "aws_db_subnet_group" "main" {
  name       = local.name_prefix
  subnet_ids = module.vpc.private_subnets

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db-subnet-group"
  })
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = local.name_prefix
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id         = local.name_prefix
  description                  = "Redis cluster for Model Card Generator"
  
  node_type                    = "cache.t3.micro"
  port                         = 6379
  parameter_group_name         = "default.redis7"
  
  num_cache_clusters           = 2
  automatic_failover_enabled   = true
  multi_az_enabled            = true
  
  subnet_group_name           = aws_elasticache_subnet_group.main.name
  security_group_ids          = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  
  tags = local.common_tags
}

resource "aws_security_group" "redis" {
  name_prefix = "${local.name_prefix}-redis-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for Redis cluster"

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.application.id]
    description     = "Redis from application"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-redis-sg"
  })
}

# CloudWatch Monitoring
resource "aws_cloudwatch_log_group" "application" {
  name              = "/aws/eks/${local.name_prefix}/application"
  retention_in_days = 30
  
  tags = local.common_tags
}

# SNS Topic for Alerts
resource "aws_sns_topic" "alerts" {
  name = "${local.name_prefix}-alerts"
  
  tags = local.common_tags
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "${local.name_prefix}-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EKS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors eks cpu utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    ClusterName = module.eks.cluster_id
  }
  
  tags = local.common_tags
}

# IAM Roles for Kubernetes Service Accounts
module "iam_assumable_role_admin" {
  source = "./modules/iam-assumable-role-with-oidc"
  
  create_role                   = true
  role_name                    = "${local.name_prefix}-admin"
  provider_url                 = replace(module.eks.cluster_oidc_issuer_url, "https://", "")
  role_policy_arns            = ["arn:aws:iam::aws:policy/AdministratorAccess"]
  oidc_fully_qualified_subjects = ["system:serviceaccount:kube-system:admin"]
  
  tags = local.common_tags
}

# Backup and Disaster Recovery
resource "aws_backup_vault" "main" {
  name        = local.name_prefix
  kms_key_arn = aws_kms_key.backup.arn
  
  tags = local.common_tags
}

resource "aws_kms_key" "backup" {
  description             = "KMS key for backup encryption"
  deletion_window_in_days = 7
  
  tags = local.common_tags
}

resource "aws_backup_plan" "main" {
  name = local.name_prefix
  
  rule {
    rule_name         = "daily_backup"
    target_vault_name = aws_backup_vault.main.name
    schedule          = "cron(0 5 ? * * *)"
    
    lifecycle {
      cold_storage_after = 30
      delete_after       = 120
    }
    
    recovery_point_tags = local.common_tags
  }
  
  tags = local.common_tags
}

# Route53 DNS
data "aws_route53_zone" "main" {
  name         = "terragonlabs.com"
  private_zone = false
}

resource "aws_route53_record" "app" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "modelcard-generator.terragonlabs.com"
  type    = "A"
  
  alias {
    name                   = aws_lb.application.dns_name
    zone_id               = aws_lb.application.zone_id
    evaluate_target_health = true
  }
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = module.eks.cluster_iam_role_name
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "application_url" {
  description = "Application URL"
  value       = "https://modelcard-generator.terragonlabs.com"
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = var.environment == "prod" ? module.database[0].endpoint : null
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
}