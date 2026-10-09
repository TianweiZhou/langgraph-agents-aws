import pytest
from pydantic import BaseModel

from agents.base import Agent
from orchestrator.state import GraphState, StateUpdate


class _In(BaseModel):
    text: str


class _Out(BaseModel):
    text: str


def _node(state: GraphState) -> StateUpdate:
    return {"result": {"text": state["request"]}}


def _make(name: str = "echo", description: str = "Echoes the request.") -> Agent:
    return Agent(name=name, description=description, input_model=_In, output_model=_Out, node=_node)


def test_agent_holds_contract_fields() -> None:
    agent = _make()
    assert agent.name == "echo"
    assert agent.input_model is _In
    assert agent.output_model is _Out


def test_agent_node_returns_partial_update() -> None:
    state: GraphState = {
        "user_id": "u1",
        "thread_id": "t1",
        "request": "hi",
        "agent": "echo",
        "result": None,
    }
    assert _make().node(state) == {"result": {"text": "hi"}}


@pytest.mark.parametrize(("name", "description"), [("", "desc"), ("echo", "  ")])
def test_agent_rejects_blank_name_or_description(name: str, description: str) -> None:
    with pytest.raises(ValueError):
        _make(name, description)
