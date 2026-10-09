import pytest
from pydantic import BaseModel

from agents.base import Agent
from helpers import make_state
from orchestrator.graph import build_default_graph, build_graph
from orchestrator.router import RuleRouter, UnknownAgentError
from orchestrator.state import GraphState, StateUpdate


class _Model(BaseModel):
    pass


def _agent(name: str) -> Agent:
    def node(state: GraphState) -> StateUpdate:
        return {"result": {"handled_by": name}}

    return Agent(
        name=name, description=f"{name} agent", input_model=_Model, output_model=_Model, node=node
    )


def test_default_graph_runs_hello_end_to_end() -> None:
    final = build_default_graph().invoke(make_state(request="ping"))
    assert final["agent"] == "hello"
    assert final["result"] == {"message": "Hello from the hello agent! You said: ping"}
    assert final["user_id"] == "u1"
    assert final["thread_id"] == "t1"


def test_graph_routes_to_requested_agent() -> None:
    agents = (_agent("a"), _agent("b"))
    graph = build_graph(agents, RuleRouter(agents, default="a"))
    assert graph.invoke(make_state(agent="b"))["result"] == {"handled_by": "b"}


def test_graph_raises_for_unknown_agent() -> None:
    with pytest.raises(UnknownAgentError):
        build_default_graph().invoke(make_state(agent="nope"))


def test_build_graph_rejects_duplicate_names() -> None:
    agents = (_agent("a"), _agent("a"))
    with pytest.raises(ValueError, match="unique"):
        build_graph(agents, RuleRouter(agents, default="a"))


def test_build_graph_rejects_reserved_name() -> None:
    agents = (_agent("router"),)
    with pytest.raises(ValueError, match="reserved"):
        build_graph(agents, RuleRouter(agents, default="router"))
