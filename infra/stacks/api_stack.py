"""HTTP API (IAM auth) -> Lambda running the LangGraph app."""

# CDK's generated (jsii) Python types hide that classes like Role and Function implement
# their interfaces (IRole, IFunction, ...), so pyright flags valid arguments here.
# pyright: reportArgumentType=false

from pathlib import Path

from aws_cdk import CfnOutput, Duration, Environment, RemovalPolicy, Stack
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_authorizers as authorizers
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct

from infra.bedrock_access import bedrock_resource_arns


class ApiStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        app_env: str,
        bundle_dir: Path,
        bedrock_model_id: str,
        env: Environment,
    ) -> None:
        super().__init__(scope, construct_id, env=env)

        # Explicit names so resources are easy to find in the console.
        prefix = f"agents-{app_env}"
        function_name = f"{prefix}-orchestrator"

        log_group = logs.LogGroup(
            self,
            "AgentsFnLogs",
            log_group_name=f"/aws/lambda/{function_name}",
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Conversation memory: LangGraph checkpoints per thread (schema set by DynamoDBSaver).
        conversations = dynamodb.TableV2(
            self,
            "ConversationsTable",
            table_name=f"{prefix}-conversations",
            partition_key=dynamodb.Attribute(name="PK", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="SK", type=dynamodb.AttributeType.STRING),
            billing=dynamodb.Billing.on_demand(),
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY,  # dev only: chats are disposable
        )

        # Long-term memory about people (LangGraph store; schema set by DynamoDBStore).
        memory = dynamodb.TableV2(
            self,
            "MemoryTable",
            table_name=f"{prefix}-memory",
            partition_key=dynamodb.Attribute(name="PK", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="SK", type=dynamodb.AttributeType.STRING),
            billing=dynamodb.Billing.on_demand(),
            time_to_live_attribute="expires_at",
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True  # memories are worth keeping
            ),
            removal_policy=RemovalPolicy.RETAIN,  # survives a stack delete
        )

        role = iam.Role(
            self,
            "AgentsFnRole",
            role_name=f"{function_name}-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        fn = lambda_.Function(
            self,
            "AgentsFn",
            function_name=function_name,
            role=role,
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.ARM_64,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(str(bundle_dir)),
            memory_size=512,
            timeout=Duration.seconds(29),  # HTTP API max integration timeout is 30s
            tracing=lambda_.Tracing.ACTIVE,
            log_group=log_group,
            environment={
                "APP_ENV": app_env,
                "LOG_LEVEL": "INFO",
                "BEDROCK_MODEL_ID": bedrock_model_id,
                "CHECKPOINT_TABLE": conversations.table_name,
                "SUPERVISOR_ENABLED": "true",
                "MEMORY_TABLE": memory.table_name,
            },
        )
        conversations.grant_read_write_data(fn)
        memory.grant_read_write_data(fn)
        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=bedrock_resource_arns(bedrock_model_id, self.region, self.account),
            )
        )

        api = apigw.HttpApi(
            self,
            "AgentsApi",
            api_name=f"{prefix}-api",
            default_authorizer=authorizers.HttpIamAuthorizer(),
        )
        api.add_routes(
            path="/invoke",
            methods=[apigw.HttpMethod.POST],
            integration=integrations.HttpLambdaIntegration(
                "AgentsFnIntegration",
                fn,
            ),
        )

        CfnOutput(self, "InvokeUrl", value=f"{api.api_endpoint}/invoke")
        CfnOutput(self, "FunctionName", value=fn.function_name)
        CfnOutput(self, "BedrockModelId", value=bedrock_model_id)
