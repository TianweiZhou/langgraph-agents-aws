import json
import logging
from typing import Any

import pytest

from helpers import make_state
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


def test_agent_logs_carry_request_thread_and_agent(caplog: pytest.LogCaptureFixture) -> None:
    from pydantic import BaseModel

    from agents.base import Agent
    from helpers import Script, decision
    from orchestrator.graph import build_graph
    from orchestrator.state import GraphState, StateUpdate
    from shared.logging import get_logger

    class _Model(BaseModel):
        pass

    def node(state: GraphState) -> StateUpdate:
        get_logger("test.agent").info("Agent working")
        return {"reply": "done"}

    agent = Agent(name="tone", description="t", input_model=_Model, output_model=_Model, node=node)
    graph = build_graph([agent], decide=Script([decision(action="route", agent="tone")]))
    formatter = JsonFormatter()
    with caplog.at_level(logging.INFO), log_context(request_id="r1", thread_id="t1"):
        graph.invoke(make_state())
    lines = [json.loads(formatter.format(r)) for r in caplog.records]
    working = next(line for line in lines if line["message"] == "Agent working")
    assert (working["request_id"], working["thread_id"], working["agent"]) == ("r1", "t1", "tone")
