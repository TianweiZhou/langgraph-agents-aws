"""Conversation memory: follow-ups in a thread see earlier turns; threads are per user."""

import json
import uuid
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_checkpoint_aws import DynamoDBSaver

import handler
from agents.tone import agent as tone_module
from helpers import make_state
from orchestrator.router import last_agent
from shared.checkpointer import checkpoint_thread_id, make_checkpointer
from shared.config import Settings
from shared.history import format_history


class _Recorder:
    """Fake model: records each prompt, replies with numbered rewrites."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self.prompts.append(messages[-1].text)
        return AIMessage(content=f"rewrite {len(self.prompts)}")


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(tone_module, "get_chat_model", lambda: recorder)
    return recorder


def _send(user_id: str, thread_id: str, request: str, **extra: Any) -> dict[str, Any]:
    body = {"user_id": user_id, "thread_id": thread_id, "request": request, **extra}
    resp = handler.lambda_handler({"body": json.dumps(body)}, None)
    return {"status": resp["statusCode"], **json.loads(resp["body"])}


def test_follow_up_sees_previous_rewrite(model: _Recorder) -> None:
    thread = f"t-{uuid.uuid4().hex}"
    first = _send("u1", thread, "send me the report", inputs={"tone": "formal"})
    assert (first["status"], first["agent"]) == (200, "tone")

    follow_up = _send("u1", thread, "make it shorter")  # no tone, no agent
    assert (follow_up["status"], follow_up["agent"]) == (200, "tone")
    assert "User: send me the report" in model.prompts[1]
    assert "Assistant: rewrite 1" in model.prompts[1]


def test_threads_are_isolated_per_user(model: _Recorder) -> None:
    thread = f"t-{uuid.uuid4().hex}"
    _send("u1", thread, "send me the report", inputs={"tone": "formal"})
    other_user = _send("u2", thread, "make it shorter")
    assert other_user["agent"] == "hello"  # no history from u1, so no sticky routing
    assert len(model.prompts) == 1


def test_tone_follow_up_without_history_needs_a_tone(model: _Recorder) -> None:
    resp = _send("u1", f"t-{uuid.uuid4().hex}", "x", agent="tone")
    assert resp["status"] == 400


def test_last_agent_reads_latest_reply() -> None:
    state = make_state()
    assert last_agent(state) is None
    state["messages"] = [HumanMessage("a"), AIMessage("b", name="tone"), HumanMessage("c")]
    assert last_agent(state) == "tone"


def test_format_history_excludes_current_message() -> None:
    messages = [HumanMessage("hi"), AIMessage("Hello.", name="tone"), HumanMessage("shorter")]
    assert format_history(messages) == "User: hi\nAssistant: Hello."
    assert format_history([HumanMessage("only")]) == "(none)"


def _settings(table: str | None) -> Settings:
    return Settings(
        app_env="test",
        aws_region="us-east-1",
        log_level="INFO",
        bedrock_model_id=None,
        guardrail_id=None,
        guardrail_version="DRAFT",
        checkpoint_table=table,
        supervisor_enabled=False,
    )


def test_checkpointer_is_in_memory_without_table() -> None:
    assert isinstance(make_checkpointer(_settings(None)), InMemorySaver)


def test_checkpointer_uses_dynamodb_with_table() -> None:
    saver = make_checkpointer(_settings("agents-test-conversations"))
    assert isinstance(saver, DynamoDBSaver)
    assert saver.table_name == "agents-test-conversations"


def test_thread_key_includes_user() -> None:
    assert checkpoint_thread_id("u1", "t1") == "u1:t1"
