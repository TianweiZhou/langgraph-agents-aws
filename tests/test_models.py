import pytest
from langchain_aws import ChatBedrockConverse

from shared import models
from shared.config import Settings, get_settings
from shared.guardrails import guardrail_config


def _settings(**overrides: str | None) -> Settings:
    base: dict[str, str | bool | None] = {
        "app_env": "test",
        "aws_region": "us-east-1",
        "log_level": "INFO",
        "bedrock_model_id": "us.amazon.nova-2-lite-v1:0",
        "guardrail_id": None,
        "guardrail_version": "DRAFT",
        "checkpoint_table": None,
        "supervisor_enabled": False,
    }
    base.update(overrides)
    return Settings(**base)  # pyright: ignore[reportArgumentType]


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    get_settings.cache_clear()
    models.get_chat_model.cache_clear()


def test_model_id_comes_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BEDROCK_MODEL_ID", "us.amazon.nova-micro-v1:0")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    model = models.get_chat_model()
    assert isinstance(model, ChatBedrockConverse)
    assert model.model_id == "us.amazon.nova-micro-v1:0"
    assert model.guardrail_config is None


def test_missing_model_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    with pytest.raises(models.ModelNotConfiguredError):
        models.get_chat_model()


def test_guardrail_off_by_default() -> None:
    assert guardrail_config(_settings()) is None


def test_guardrail_config_when_set() -> None:
    config = guardrail_config(_settings(guardrail_id="gr-123", guardrail_version="2"))
    assert config == {"guardrailIdentifier": "gr-123", "guardrailVersion": "2"}
