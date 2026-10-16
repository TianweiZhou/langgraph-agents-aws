from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.registry import ALL_AGENTS, DEFAULT_AGENT
from helpers import make_state
from orchestrator.router import UnknownAgentError, make_router_node
from orchestrator.supervisor import RouteDecision, SupervisorRouter


class _FakeDecide:
    """Stands in for the model: returns a fixed decision and records the prompt."""

    def __init__(self, **decision: Any) -> None:
        self.decision = RouteDecision.model_validate(
            {"follow_up": False, "reason": "test", **decision}
        )
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> RouteDecision:
        self.prompts.append(prompt)
        return self.decision


def _router(decide: Any) -> SupervisorRouter:
    return SupervisorRouter(ALL_AGENTS, default=DEFAULT_AGENT, decide=decide)


def test_routes_to_model_choice_with_extracted_inputs() -> None:
    decide = _FakeDecide(agent="tone", tone="formal", text="hey send the report")
    route = _router(decide).choose(make_state(request="make this formal: hey send the report"))
    assert (route.agent, route.router) == ("tone", "supervisor")
    assert route.inputs == {"tone": "formal", "text": "hey send the report"}


def test_prompt_lists_agents_history_and_message() -> None:
    decide = _FakeDecide(agent="hello")
    state = make_state(request="ping")
    state["messages"] = [
        HumanMessage("hi"),
        AIMessage("Hello!", name="hello"),
        HumanMessage("ping"),
    ]
    _router(decide).choose(state)
    prompt = decide.prompts[0]
    assert "- tone:" in prompt and "- hello:" in prompt
    assert "User: hi\nAssistant: Hello!" in prompt
    assert "<message>\nping\n</message>" in prompt


def test_follow_up_goes_to_previous_agent() -> None:
    state = make_state(request="shorter")
    state["messages"] = [HumanMessage("x"), AIMessage("y", name="tone"), HumanMessage("shorter")]
    route = _router(_FakeDecide(agent="hello", follow_up=True)).choose(state)
    assert route.agent == "tone"


def test_greeting_is_not_forced_into_last_agent() -> None:
    """The bug the supervisor fixes: 'ping' after a rewrite went to the tone agent."""
    state = make_state(request="ping")
    state["messages"] = [HumanMessage("x"), AIMessage("y", name="tone"), HumanMessage("ping")]
    assert _router(_FakeDecide(agent="hello")).choose(state).agent == "hello"


def test_explicit_agent_skips_the_model() -> None:
    decide = _FakeDecide(agent="hello")
    route = _router(decide).choose(make_state(agent="tone"))
    assert route.agent == "tone"
    assert decide.prompts == []


def test_unknown_explicit_agent_still_raises() -> None:
    with pytest.raises(UnknownAgentError):
        _router(_FakeDecide(agent="hello")).choose(make_state(agent="nope"))


def test_model_error_falls_back_to_rules() -> None:
    def broken(prompt: str) -> RouteDecision:
        raise RuntimeError("bedrock down")

    route = _router(broken).choose(make_state(request="please rewrite this"))
    assert (route.agent, route.router) == ("tone", "rules")


def test_unknown_model_choice_falls_back_to_rules() -> None:
    route = _router(_FakeDecide(agent="poet")).choose(make_state(request="hi"))
    assert (route.agent, route.router) == ("hello", "rules")


def test_agent_name_is_normalized() -> None:
    assert _router(_FakeDecide(agent=" Tone ")).choose(make_state()).agent == "tone"


def test_caller_inputs_win_over_extracted_inputs() -> None:
    node = make_router_node(_router(_FakeDecide(agent="tone", tone="casual")))
    update = node(make_state(request="x", inputs={"tone": "formal"}))
    assert update == {"agent": "tone", "inputs": {"tone": "formal"}}
