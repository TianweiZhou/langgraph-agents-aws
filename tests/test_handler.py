import base64
import json
from typing import Any

import pytest

import handler


def _event(body: dict[str, Any] | str, *, b64: bool = False) -> dict[str, Any]:
    raw = body if isinstance(body, str) else json.dumps(body)
    if b64:
        return {"body": base64.b64encode(raw.encode()).decode(), "isBase64Encoded": True}
    return {"body": raw, "isBase64Encoded": False}


def _call(event: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    resp = handler.lambda_handler(event, None)
    return resp["statusCode"], json.loads(resp["body"])


def test_hello_request_returns_200() -> None:
    status, body = _call(_event({"user_id": "u1", "thread_id": "t1", "request": "ping"}))
    assert status == 200
    assert body == {
        "thread_id": "t1",
        "agent": "hello",
        "result": {"message": "Hello from the hello agent! You said: ping"},
    }


def test_thread_id_is_generated_when_missing() -> None:
    status, body = _call(_event({"user_id": "u1", "request": "ping"}))
    assert status == 200
    assert body["thread_id"]


def test_base64_body_is_decoded() -> None:
    status, _ = _call(_event({"user_id": "u1", "request": "ping"}, b64=True))
    assert status == 200


@pytest.mark.parametrize(
    "body",
    ["not json", {"request": "ping"}, {"user_id": "u1"}, {"user_id": "u1", "request": ""}],
)
def test_bad_body_returns_400(body: dict[str, Any] | str) -> None:
    status, resp = _call(_event(body))
    assert status == 400
    assert resp == {"error": "Invalid request body"}


def test_missing_body_returns_400() -> None:
    status, _ = _call({})
    assert status == 400


def test_unknown_agent_returns_400() -> None:
    status, body = _call(_event({"user_id": "u1", "request": "ping", "agent": "nope"}))
    assert status == 400
    assert "nope" in body["error"]


def test_unexpected_error_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(req: handler.InvokeRequest) -> handler.InvokeResponse:
        raise RuntimeError("secret detail")

    monkeypatch.setattr(handler, "_run", boom)
    status, body = _call(_event({"user_id": "u1", "request": "ping"}))
    assert status == 500
    assert body == {"error": "Internal error"}  # no internal details leak


def _fake_tone_model(monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from agents.tone import agent as tone_module

    model = GenericFakeChatModel(messages=iter([AIMessage(content=reply)]))
    monkeypatch.setattr(tone_module, "get_chat_model", lambda: model)


def test_tone_request_returns_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tone_model(monkeypatch, "Could you please send the report?")
    status, body = _call(
        _event({"user_id": "u1", "request": "send the report", "inputs": {"tone": "polite"}})
    )
    assert status == 200
    assert body["agent"] == "tone"
    expected = {"rewritten_text": "Could you please send the report?", "tone": "polite"}
    assert body["result"] == expected


def test_tone_without_target_tone_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tone_model(monkeypatch, "unused")
    status, body = _call(_event({"user_id": "u1", "request": "x", "agent": "tone"}))
    assert status == 400
    assert "tone" in body["error"]
