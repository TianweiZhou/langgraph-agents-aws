import pytest
from pydantic import BaseModel

from agents.base import Agent, InvalidAgentInput, parse_input
from helpers import make_state
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
    assert _make().node(make_state(request="hi")) == {"result": {"text": "hi"}}


@pytest.mark.parametrize(("name", "description"), [("", "desc"), ("echo", "  ")])
def test_agent_rejects_blank_name_or_description(name: str, description: str) -> None:
    with pytest.raises(ValueError):
        _make(name, description)


class _Needs(BaseModel):
    text: str
    tone: str


def test_parse_input_merges_request_and_inputs() -> None:
    parsed = parse_input(_Needs, make_state(request="hey", inputs={"tone": "formal"}))
    assert (parsed.text, parsed.tone) == ("hey", "formal")


def test_parse_input_names_missing_fields() -> None:
    with pytest.raises(InvalidAgentInput, match="tone"):
        parse_input(_Needs, make_state(request="hey"))
