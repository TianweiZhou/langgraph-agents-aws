"""Bedrock Guardrails hook. Off until a guardrail id is configured."""

from shared.config import Settings


def guardrail_config(settings: Settings) -> dict[str, str] | None:
    """Converse API `guardrailConfig`, or None when no guardrail is set."""
    if settings.guardrail_id is None:
        return None
    return {
        "guardrailIdentifier": settings.guardrail_id,
        "guardrailVersion": settings.guardrail_version,
    }
