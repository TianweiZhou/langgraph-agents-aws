"""IAM resources the Lambda needs to call the configured Bedrock model."""

# Cross-region inference profile ids start with one of these, e.g. "us.amazon.nova-2-lite-v1:0".
_PROFILE_PREFIXES = ("us.", "eu.", "apac.", "global.", "us-gov.", "jp.", "au.", "ca.")


def is_inference_profile(model_id: str) -> bool:
    return model_id.startswith(_PROFILE_PREFIXES)


def bedrock_resource_arns(model_id: str, region: str, account: str) -> list[str]:
    """ARNs to allow `bedrock:InvokeModel` on for `model_id`.

    An inference profile routes to the base model in several regions, so it
    needs the profile ARN plus the base model's ARN in any region.
    """
    if not is_inference_profile(model_id):
        return [f"arn:aws:bedrock:{region}::foundation-model/{model_id}"]
    base_model_id = model_id.split(".", 1)[1]
    return [
        f"arn:aws:bedrock:{region}:{account}:inference-profile/{model_id}",
        f"arn:aws:bedrock:*::foundation-model/{base_model_id}",
    ]
