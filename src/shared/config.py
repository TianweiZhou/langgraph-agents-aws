"""Runtime settings, read from environment variables (never hardcoded)."""

import os
from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class Settings:
    app_env: str  # e.g. "local", "dev", "prod"
    aws_region: str
    log_level: str
    bedrock_model_id: str | None  # model or inference profile id, set by CDK
    guardrail_id: str | None  # Bedrock Guardrail; off when unset
    guardrail_version: str


@cache
def get_settings() -> Settings:
    return Settings(
        app_env=os.environ.get("APP_ENV", "local"),
        aws_region=os.environ.get("AWS_REGION", "us-east-1"),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        bedrock_model_id=os.environ.get("BEDROCK_MODEL_ID") or None,
        guardrail_id=os.environ.get("GUARDRAIL_ID") or None,
        guardrail_version=os.environ.get("GUARDRAIL_VERSION", "DRAFT"),
    )
