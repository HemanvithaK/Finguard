# terraform/variables.tf
variable "aws_region" {
  description = "AWS region to deploy to"
  type        = string
  default     = "us-west-2"
}

variable "anthropic_api_key" {
  description = "Anthropic API key for the investigation agent"
  type        = string
  sensitive   = true
}