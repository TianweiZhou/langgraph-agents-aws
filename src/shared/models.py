"""Bedrock chat model factory. Agents get models here and never name a model id."""

from functools import cache

from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel

from shared.config import get_settings
from shared.guardrails import guardrail_config

DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1024


class ModelNotConfiguredError(RuntimeError):
    """BEDROCK_MODEL_ID is missing."""


@cache
def get_chat_model() -> BaseChatModel:
    """The configured Bedrock chat model, built once per process."""
    settings = get_settings()
    if settings.bedrock_model_id is None:
        raise ModelNotConfiguredError("Set BEDROCK_MODEL_ID to a Bedrock model id")
    return ChatBedrockConverse(
        model=settings.bedrock_model_id,
        region_name=settings.aws_region,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        guardrails=guardrail_config(settings),
    )
