from infra.bedrock_access import bedrock_resource_arns, is_inference_profile


def test_inference_profile_needs_profile_and_base_model_arns() -> None:
    arns = bedrock_resource_arns("us.amazon.nova-2-lite-v1:0", "us-east-1", "123456789012")
    assert arns == [
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-2-lite-v1:0",
        "arn:aws:bedrock:*::foundation-model/amazon.nova-2-lite-v1:0",
    ]


def test_plain_model_id_needs_only_foundation_model_arn() -> None:
    arns = bedrock_resource_arns("amazon.nova-micro-v1:0", "us-east-1", "123456789012")
    assert arns == ["arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0"]


def test_profile_detection() -> None:
    assert is_inference_profile("global.amazon.nova-2-lite-v1:0")
    assert not is_inference_profile("anthropic.claude-model-v1:0")
