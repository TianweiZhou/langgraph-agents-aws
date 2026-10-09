import json
import logging
from typing import Any

import pytest

from helpers import make_state
from orchestrator.graph import build_default_graph
from shared.logging import JsonFormatter, log_context


def _format(msg: str, **extra: Any) -> dict[str, Any]:
    logger = logging.getLogger("test")
    record = logger.makeRecord("test", logging.INFO, __file__, 1, msg, (), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return json.loads(JsonFormatter().format(record))


def test_log_line_has_thread_id_and_agent_fields() -> None:
    line = _format("hi")
    assert line["message"] == "hi"
    assert line["level"] == "INFO"
    assert line["thread_id"] is None
    assert line["agent"] is None


def test_log_context_sets_and_resets_fields() -> None:
    with log_context(thread_id="t1", agent="hello"):
        line = _format("inside")
    assert (line["thread_id"], line["agent"]) == ("t1", "hello")
    after = _format("outside")
    assert (after["thread_id"], after["agent"]) == (None, None)


def test_extra_fields_are_included() -> None:
    assert _format("hi", user_id="u1")["user_id"] == "u1"


def test_agent_logs_carry_thread_id_and_agent(caplog: pytest.LogCaptureFixture) -> None:
    formatter = JsonFormatter()
    with caplog.at_level(logging.INFO), log_context(thread_id="t1"):
        build_default_graph().invoke(make_state())
    lines = [json.loads(formatter.format(r)) for r in caplog.records]
    started = next(line for line in lines if line["message"] == "Agent started")
    assert (started["thread_id"], started["agent"]) == ("t1", "hello")
