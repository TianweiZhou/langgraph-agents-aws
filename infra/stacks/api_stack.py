"""HTTP API (IAM auth) -> Lambda running the LangGraph app."""

from pathlib import Path

from aws_cdk import CfnOutput, Duration, Environment, RemovalPolicy, Stack
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_authorizers as authorizers
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct


class ApiStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        app_env: str,
        bundle_dir: Path,
        env: Environment,
    ) -> None:
        super().__init__(scope, construct_id, env=env)

        log_group = logs.LogGroup(
            self,
            "AgentsFnLogs",
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )

        fn = lambda_.Function(
            self,
            "AgentsFn",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.ARM_64,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(str(bundle_dir)),
            memory_size=512,
            timeout=Duration.seconds(29),  # HTTP API max integration timeout is 30s
            tracing=lambda_.Tracing.ACTIVE,
            log_group=log_group,
            environment={"APP_ENV": app_env, "LOG_LEVEL": "INFO"},
        )

        api = apigw.HttpApi(
            self,
            "AgentsApi",
            api_name=f"agents-{app_env}",
            default_authorizer=authorizers.HttpIamAuthorizer(),
        )
        api.add_routes(
            path="/invoke",
            methods=[apigw.HttpMethod.POST],
            # Function implements IFunction; jsii's generated types hide that from pyright.
            integration=integrations.HttpLambdaIntegration(
                "AgentsFnIntegration",
                fn,  # pyright: ignore[reportArgumentType]
            ),
        )

        CfnOutput(self, "InvokeUrl", value=f"{api.api_endpoint}/invoke")
        CfnOutput(self, "FunctionName", value=fn.function_name)
