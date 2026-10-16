import base64
import json
import uuid
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_checkpoint_aws import DynamoDBSaver

import handler
from agents.tone import agent as tone_module
from helpers import Script, decision
from orchestrator import supervisor as supervisor_module
from shared.checkpointer import checkpoint_thread_id, make_checkpointer
from shared.config import Settings


def _event(body: dict[str, Any] | str, *, b64: bool = False) -> dict[str, Any]:
    raw = body if isinstance(body, str) else json.dumps(body)
    if b64:
        return {"body": base64.b64encode(raw.encode()).decode(), "isBase64Encoded": True}
    return {"body": raw, "isBase64Encoded": False}


def _call(event: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    resp = handler.lambda_handler(event, None)
    return resp["statusCode"], json.loads(resp["body"])


def _script(monkeypatch: pytest.MonkeyPatch, *decisions: Any) -> Script:
    script = Script(decisions)
    monkeypatch.setattr(supervisor_module, "decide_with_model", script)
    return script


class _ToneModel:
    """Fake tone model: records prompts, replies with numbered rewrites."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self.prompts.append(messages[-1].text)
        return AIMessage(content=f"rewrite {len(self.prompts)}")


@pytest.fixture
def tone_model(monkeypatch: pytest.MonkeyPatch) -> _ToneModel:
    model = _ToneModel()
    monkeypatch.setattr(tone_module, "get_chat_model", lambda: model)
    return model


def _body(request: str, thread: str | None = None, user: str = "u1") -> dict[str, Any]:
    return {"user_id": user, "thread_id": thread or f"t-{uuid.uuid4().hex}", "request": request}


def test_supervisor_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, decision(action="respond", reply="Hi! How can I help?"))
    status, body = _call(_event(_body("hi", "t1")))
    assert status == 200
    assert body["reply"] == "Hi! How can I help?"
    assert (body["thread_id"], body["route"], body["note"]) == ("t1", [], None)
    assert body["request_id"]


def test_routed_rewrite(monkeypatch: pytest.MonkeyPatch, tone_model: _ToneModel) -> None:
    _script(monkeypatch, decision(action="route", agent="tone", tone="polite", text="send it"))
    status, body = _call(_event(_body("make this polite: send it")))
    assert (status, body["reply"], body["route"]) == (200, "rewrite 1", ["tone"])
    assert "<text>\nsend it\n</text>" in tone_model.prompts[0]


def test_follow_up_reworks_previous_rewrite(
    monkeypatch: pytest.MonkeyPatch, tone_model: _ToneModel
) -> None:
    thread = f"t-{uuid.uuid4().hex}"
    _script(
        monkeypatch,
        decision(action="route", agent="tone", tone="formal", text="send me the report"),
        decision(action="route", agent="tone", follow_up=True),
    )
    _call(_event(_body("make this formal: send me the report", thread)))
    status, body = _call(_event(_body("shorter please", thread)))
    assert (status, body["route"]) == (200, ["tone"])
    second = tone_model.prompts[1]
    assert "<text>\nrewrite 1\n</text>" in second  # works on its previous rewrite
    assert "apply this change: shorter please" in second


def test_threads_are_isolated_per_user(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script(
        monkeypatch,
        decision(action="respond", reply="one"),
        decision(action="respond", reply="two"),
    )
    thread = f"t-{uuid.uuid4().hex}"
    _call(_event(_body("secret plan", thread, user="u1")))
    _call(_event(_body("what did I say?", thread, user="u2")))
    assert "secret plan" not in script.prompts[1]


def test_missing_tone_asks_the_user(
    monkeypatch: pytest.MonkeyPatch, tone_model: _ToneModel
) -> None:
    _script(monkeypatch, decision(action="route", agent="tone"))
    status, body = _call(_event(_body("rewrite this")))
    assert status == 200
    assert body["reply"].startswith("Which tone should I use?")


def test_thread_id_is_generated_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, decision(action="respond", reply="ok"))
    status, body = _call(_event({"user_id": "u1", "request": "ping"}))
    assert status == 200 and body["thread_id"]


def test_base64_body_is_decoded(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, decision(action="respond", reply="ok"))
    status, _ = _call(_event({"user_id": "u1", "request": "ping"}, b64=True))
    assert status == 200


@pytest.mark.parametrize(
    "body",
    ["not json", {"request": "ping"}, {"user_id": "u1"}, {"user_id": "u1", "request": ""}],
)
def test_bad_body_returns_400(body: dict[str, Any] | str) -> None:
    status, resp = _call(_event(body))
    assert (status, resp["error"]) == (400, "Invalid request body")


def test_missing_body_returns_400() -> None:
    status, _ = _call({})
    assert status == 400


def test_unexpected_error_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(req: handler.InvokeRequest, request_id: str) -> handler.InvokeResponse:
        raise RuntimeError("secret detail")

    monkeypatch.setattr(handler, "_run", boom)
    status, body = _call(_event(_body("ping")))
    assert status == 500
    assert body["error"] == "Internal error"  # no internal details leak
    assert "secret" not in json.dumps(body)


def _settings(table: str | None) -> Settings:
    return Settings(
        app_env="test",
        aws_region="us-east-1",
        log_level="INFO",
        bedrock_model_id=None,
        guardrail_id=None,
        guardrail_version="DRAFT",
        checkpoint_table=table,
        trace_content=True,
        memory_table=None,
    )


def test_checkpointer_is_in_memory_without_table() -> None:
    assert isinstance(make_checkpointer(_settings(None)), InMemorySaver)


def test_checkpointer_uses_dynamodb_with_table() -> None:
    saver = make_checkpointer(_settings("agents-test-conversations"))
    assert isinstance(saver, DynamoDBSaver)
    assert saver.table_name == "agents-test-conversations"


def test_thread_key_includes_user() -> None:
    assert checkpoint_thread_id("u1", "t1") == "u1:t1"
