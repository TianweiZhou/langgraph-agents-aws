import json
import logging
import uuid
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

import handler
from shared.usage import UsageCallback, track_usage


def _result(input_tokens: int, output_tokens: int) -> LLMResult:
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_callback_adds_up_tokens_inside_tracking() -> None:
    callback = UsageCallback()
    with track_usage() as usage:
        callback.on_llm_end(_result(100, 20))
        callback.on_llm_end(_result(50, 5))
    assert (usage.model_calls, usage.input_tokens, usage.output_tokens) == (2, 150, 25)


def test_callback_outside_tracking_is_ignored() -> None:
    UsageCallback().on_llm_end(_result(1, 1))  # no error, nothing to record into


def _summaries(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        {k: getattr(r, k, None) for k in ("status", "agent", "error", "duration_ms", "user_id")}
        for r in caplog.records
        if r.getMessage() == "Request summary"
    ]


def _call(body: dict[str, Any]) -> None:
    handler.lambda_handler({"body": json.dumps(body)}, None)


def test_summary_logged_for_success(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        _call({"user_id": "u1", "thread_id": f"t-{uuid.uuid4().hex}", "request": "ping"})
    [summary] = _summaries(caplog)
    assert (summary["status"], summary["agent"], summary["user_id"]) == (200, "hello", "u1")
    assert isinstance(summary["duration_ms"], int)


def test_summary_names_agent_that_rejected(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        _call(
            {"user_id": "u1", "thread_id": f"t-{uuid.uuid4().hex}", "request": "x", "agent": "tone"}
        )
    [summary] = _summaries(caplog)
    assert (summary["status"], summary["agent"]) == (400, "tone")
    assert "tone" in summary["error"]


def test_summary_logged_for_bad_body(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        handler.lambda_handler({"body": "not json"}, None)
    [summary] = _summaries(caplog)
    assert (summary["status"], summary["error"]) == (400, "Invalid request body")
