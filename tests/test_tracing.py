import json
import logging
import uuid
from typing import Any

import pytest

import handler
from helpers import Script, decision
from orchestrator import supervisor as supervisor_module
from shared import tracing
from shared.config import get_settings


def _trace_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        {
            k: getattr(r, k, None)
            for k in ("step", "node", "next", "received", "produced", "request_id")
        }
        for r in caplog.records
        if r.getMessage() == "Trace step"
    ]


def _call(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, *decisions: Any):
    monkeypatch.setattr(supervisor_module, "decide_with_model", Script(decisions))
    body = {"user_id": "u1", "thread_id": f"t-{uuid.uuid4().hex}", "request": "hello there"}
    with caplog.at_level(logging.INFO):
        resp = handler.lambda_handler({"body": json.dumps(body)}, None)
    return json.loads(resp["body"]), _trace_lines(caplog)


def test_each_step_is_traced_in_order(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    body, steps = _call(monkeypatch, caplog, decision(action="respond", reply="Hi!"))
    assert [(s["step"], s["node"], s["next"]) for s in steps] == [
        (1, "supervisor", "remember"),
        (2, "remember", "end"),
    ]
    assert {s["request_id"] for s in steps} == {body["request_id"]}
    produced = json.loads(steps[0]["produced"])
    assert produced["decision"]["action"] == "respond"
    assert json.loads(steps[0]["received"])["request"] == "hello there"


def test_text_is_hidden_when_content_tracing_is_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("TRACE_CONTENT", "false")
    get_settings.cache_clear()
    try:
        _, steps = _call(monkeypatch, caplog, decision(action="respond", reply="Hi!"))
    finally:
        monkeypatch.delenv("TRACE_CONTENT")
        get_settings.cache_clear()
    received = json.loads(steps[0]["received"])
    assert received["request"] == "<11 chars>"


def test_long_text_is_cut() -> None:
    value = tracing._redact({"text": "x" * 500}, content=True)  # pyright: ignore[reportPrivateUsage]
    assert len(value["text"]) == tracing.MAX_TEXT + 1


def test_no_trace_outside_a_request(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        tracing.trace_step("x", received={}, produced={}, next_node="y", started=0.0)
    assert _trace_lines(caplog) == []
