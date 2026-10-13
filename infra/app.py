"""CDK app entrypoint. Usage: `cdk deploy -c env=dev --profile dev`."""

import os

import aws_cdk as cdk

from infra.build_lambda import build_lambda_bundle
from infra.stacks.api_stack import ApiStack

REGION = "us-east-1"

app = cdk.App()
app_env = str(app.node.try_get_context("env") or "dev")
# Change the model in cdk.json, or per deploy: cdk deploy -c bedrockModelId=<id>
bedrock_model_id = str(app.node.get_context("bedrockModelId"))

ApiStack(
    app,
    f"AgentsApi-{app_env}",
    app_env=app_env,
    bundle_dir=build_lambda_bundle(),
    bedrock_model_id=bedrock_model_id,
    env=cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region=REGION),
)

cdk.Tags.of(app).add("Project", "langgraph-agents-aws")
cdk.Tags.of(app).add("Environment", app_env)

app.synth()
