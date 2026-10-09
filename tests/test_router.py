import pytest
from pydantic import BaseModel

from agents.base import Agent
from helpers import make_state
from orchestrator.router import RuleRouter, UnknownAgentError, make_router_node
from orchestrator.state import GraphState, StateUpdate


class _Model(BaseModel):
    pass


def _noop(state: GraphState) -> StateUpdate:
    return {}


def _agent(name: str) -> Agent:
    return Agent(
        name=name, description=f"{name} agent", input_model=_Model, output_model=_Model, node=_noop
    )


AGENTS = (_agent("hello"), _agent("tone"))


def test_falls_back_to_default_when_no_agent_requested() -> None:
    router = RuleRouter(AGENTS, default="hello")
    assert router.choose(make_state()) == "hello"


def test_uses_requested_agent() -> None:
    router = RuleRouter(AGENTS, default="hello")
    assert router.choose(make_state(agent="tone")) == "tone"


def test_rejects_unknown_requested_agent() -> None:
    router = RuleRouter(AGENTS, default="hello")
    with pytest.raises(UnknownAgentError):
        router.choose(make_state(agent="nope"))


def test_rejects_unknown_default() -> None:
    with pytest.raises(UnknownAgentError):
        RuleRouter(AGENTS, default="nope")


def test_router_node_writes_choice_to_state() -> None:
    node = make_router_node(RuleRouter(AGENTS, default="hello"))
    assert node(make_state()) == {"agent": "hello"}
