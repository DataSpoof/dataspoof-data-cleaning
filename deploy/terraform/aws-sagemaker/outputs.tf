output "endpoint_name" {
  value = aws_sagemaker_endpoint.this.name
}

output "endpoint_arn" {
  value = aws_sagemaker_endpoint.this.arn
}

output "role_arn" {
  value = aws_iam_role.endpoint.arn
}

output "invoke_example" {
  description = "Call the endpoint with the AWS CLI."
  value       = "aws sagemaker-runtime invoke-endpoint --region ${data.aws_region.current.region} --endpoint-name ${aws_sagemaker_endpoint.this.name} --content-type application/json --body '{\"records\": [{}]}' --cli-binary-format raw-in-base64-out out.json"
}
