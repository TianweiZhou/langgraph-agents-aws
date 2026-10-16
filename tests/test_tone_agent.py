from collections.abc import Sequence

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage

from agents.base import InvalidAgentInput
from agents.tone import agent as tone_module
from agents.tone.agent import EmptyModelResponseError, agent, matches, run
from agents.tone.schemas import ToneOutput
from helpers import make_state


class _Recorder:
    """Fake model that records the prompt it was sent."""

    def __init__(self) -> None:
        self.prompt = ""

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self.prompt = messages[-1].text
        return AIMessage(content="ok")


def _use_model(monkeypatch: pytest.MonkeyPatch, model: object) -> None:
    monkeypatch.setattr(tone_module, "get_chat_model", lambda: model)


def _reply(text: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


def test_returns_model_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_model(monkeypatch, _reply("  Could you please send the report?  "))
    update = run(make_state(request="send me the report", inputs={"tone": "polite"}))
    expected = {
        "rewritten_text": "Could you please send the report?",
        "tone": "polite",
        "note": None,
    }
    assert update.get("result") == expected
    assert update.get("reply") == "Could you please send the report?"
    ToneOutput.model_validate(update.get("result"))


def test_prompt_carries_text_tone_and_audience(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    _use_model(monkeypatch, recorder)
    run(make_state(request="ship it {now}", inputs={"tone": "formal", "audience": "the board"}))
    sent = recorder.prompt
    assert "Target tone: formal" in sent
    assert "Audience: the board" in sent
    assert "<text>\nship it {now}\n</text>" in sent


def test_default_audience_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    _use_model(monkeypatch, recorder)
    run(make_state(request="hi", inputs={"tone": "warm"}))
    assert "Audience: a general audience" in recorder.prompt


def test_missing_tone_is_invalid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_model(monkeypatch, _reply("unused"))
    with pytest.raises(InvalidAgentInput, match="tone"):
        run(make_state(request="rewrite this"))


def test_empty_model_reply_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_model(monkeypatch, _reply("   "))
    with pytest.raises(EmptyModelResponseError):
        run(make_state(request="x", inputs={"tone": "formal"}))


@pytest.mark.parametrize(
    ("request_text", "inputs", "expected"),
    [
        ("hey", {"tone": "formal"}, True),
        ("Please rewrite this", None, True),
        ("change the TONE here", None, True),
        ("hello there", None, False),
        ("stone wall", None, False),
    ],
)
def test_intent_matching(request_text: str, inputs: dict[str, str] | None, expected: bool) -> None:
    assert matches(make_state(request=request_text, inputs=inputs)) is expected


def test_contract() -> None:
    assert agent.name == "tone"
    assert agent.matches is matches
