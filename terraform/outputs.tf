# terraform/outputs.tf
output "api_url" {
  description = "URL of the FinGuard API (use this to call /predict, /investigate, /drift, /ab-results)"
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "ECR repository URL (push your Docker image here)"
  value       = aws_ecr_repository.finguard.repository_url
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for container logs"
  value       = aws_cloudwatch_log_group.finguard.name
}