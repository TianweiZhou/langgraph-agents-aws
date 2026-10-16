import pytest

from shared.prompts import load_prompt


def test_loads_local_prompt() -> None:
    assert "{text}" in load_prompt("tone")


def test_missing_agent_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does-not-exist")


def test_non_local_version_not_supported_yet() -> None:
    with pytest.raises(NotImplementedError):
        load_prompt("tone", version="3")


def test_loads_orchestrator_prompt() -> None:
    assert "{agents}" in load_prompt("supervisor", kind="orchestrator")
