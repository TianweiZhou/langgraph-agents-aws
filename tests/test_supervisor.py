"""The supervisor loop: routing, replying itself, finishing, multi-step, and fallbacks."""

import uuid
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from agents.base import Agent, InvalidAgentInput
from helpers import Script, decision, make_state
from orchestrator.graph import Graph, build_graph
from orchestrator.state import GraphState, StateUpdate, new_turn
from orchestrator.supervisor import FALLBACK_REPLY, MAX_AGENT_STEPS, Supervisor


class _Model(BaseModel):
    pass


def _echo_agent(name: str, calls: list[GraphState] | None = None) -> Agent:
    def node(state: GraphState) -> StateUpdate:
        if calls is not None:
            calls.append(state)
        return {"result": {"by": name}, "reply": f"{name} did it"}

    return Agent(
        name=name, description=f"{name} agent", input_model=_Model, output_model=_Model, node=node
    )


def _graph(script: Script, *agents: Agent) -> Graph:
    return build_graph(
        agents or (_echo_agent("tone"), _echo_agent("memory")), InMemorySaver(), script
    )


def _run(graph: Graph, request: str, thread: str | None = None) -> GraphState:
    config: Any = {"configurable": {"thread_id": thread or uuid.uuid4().hex}}
    return cast(GraphState, graph.invoke(new_turn("u1", "t1", request), config))


def test_supervisor_replies_itself_without_agents() -> None:
    final = _run(_graph(Script([decision(action="respond", reply="Hi! How can I help?")])), "hi")
    assert (final["reply"], final["done"]) == ("Hi! How can I help?", [])
    last = final["messages"][-1]
    assert (last.name, last.text) == ("supervisor", "Hi! How can I help?")


def test_single_step_finishes_without_second_model_call() -> None:
    script = Script([decision(action="route", agent="tone", tone="formal", text="send it")])
    final = _run(_graph(script), "make this formal: send it")
    assert final["reply"] == "tone did it"
    assert [d["agent"] for d in final["done"]] == ["tone"]
    assert len(script.prompts) == 1  # the planned single step ends the turn
    assert final["decision"] == {
        "action": "finish",
        "agent": None,
        "reason": "planned steps are done",
        "model_call": False,
    }
    assert final["messages"][-1].name == "tone"


def test_multi_step_reports_back_and_routes_again() -> None:
    calls: list[GraphState] = []
    script = Script(
        [
            decision(action="route", agent="memory", more_steps=True, task="remember Sara"),
            decision(action="route", agent="tone", recipient="Sara", text="sorry"),
        ]
    )
    graph = _graph(script, _echo_agent("tone", calls), _echo_agent("memory"))
    final = _run(graph, "Sara is my manager. Reply to her: sorry")
    assert [d["agent"] for d in final["done"]] == ["memory", "tone"]
    assert "- memory: memory did it" in script.prompts[1]  # the second decision sees the work
    assert final["reply"] == "tone did it"
    tone_inputs = calls[0]["inputs"] or {}
    assert tone_inputs.get("recipient") == "Sara"
    assert "task" not in tone_inputs  # the memory task does not leak into the tone step


def test_supervisor_can_finish_after_more_steps() -> None:
    script = Script(
        [decision(action="route", agent="tone", more_steps=True), decision(action="finish")]
    )
    final = _run(_graph(script), "x")
    assert (final["reply"], len(script.prompts)) == ("tone did it", 2)


def test_step_limit_stops_runaway_loops() -> None:
    loop = [decision(action="route", agent="tone", more_steps=True)] * (MAX_AGENT_STEPS + 2)
    final = _run(_graph(Script(loop)), "x")
    assert len(final["done"]) == MAX_AGENT_STEPS
    assert final["decision"] and final["decision"]["reason"] == "step limit reached"


def test_missing_input_becomes_a_question_for_the_user() -> None:
    def needs_more(state: GraphState) -> StateUpdate:
        raise InvalidAgentInput("Invalid inputs: tone", ask="Which tone should I use?")

    asker = Agent(
        name="tone", description="t", input_model=_Model, output_model=_Model, node=needs_more
    )
    final = _run(_graph(Script([decision(action="route", agent="tone")]), asker), "rewrite this")
    assert final["reply"] == "Which tone should I use?"


def test_follow_up_with_unknown_agent_goes_to_previous_agent() -> None:
    supervisor = Supervisor(
        [_echo_agent("tone")], Script([decision(action="route", agent="nope", follow_up=True)])
    )
    state = make_state("shorter")
    state["messages"] = [HumanMessage("x"), AIMessage("y", name="tone"), HumanMessage("shorter")]
    picked, model_call = supervisor.decide(state)
    assert (picked.agent, model_call) == ("tone", True)


def test_model_failure_falls_back_to_intent_rules() -> None:
    def broken(prompt: str) -> Any:
        raise RuntimeError("bedrock down")

    def noop(state: GraphState) -> StateUpdate:
        return {}

    tone = Agent(
        name="tone",
        description="t",
        input_model=_Model,
        output_model=_Model,
        node=noop,
        matches=lambda s: "rewrite" in s["request"],
    )
    supervisor = Supervisor([tone], broken)
    picked, model_call = supervisor.decide(make_state("please rewrite this"))
    assert (picked.action, picked.agent, model_call) == ("route", "tone", False)
    apology, _ = supervisor.decide(make_state("what's up"))
    assert (apology.action, apology.reply) == ("respond", FALLBACK_REPLY)


def test_unknown_agent_falls_back() -> None:
    supervisor = Supervisor([_echo_agent("tone")], Script([decision(action="route", agent="poet")]))
    picked, model_call = supervisor.decide(make_state("hi"))
    assert (picked.action, model_call) == ("respond", False)


def test_prompt_lists_agents_history_message_and_work() -> None:
    script = Script([decision(action="respond", reply="ok")])
    supervisor = Supervisor([_echo_agent("tone"), _echo_agent("memory")], script)
    state = make_state("ping")
    state["messages"] = [
        HumanMessage("hi"),
        AIMessage("Hello!", name="supervisor"),
        HumanMessage("ping"),
    ]
    supervisor.decide(state)
    prompt = script.prompts[0]
    assert "- tone: tone agent" in prompt and "- memory: memory agent" in prompt
    assert "User: hi\nAssistant: Hello!" in prompt
    assert "<message>\nping\n</message>" in prompt
    assert "(nothing yet)" in prompt


def test_history_carries_over_between_turns() -> None:
    script = Script(
        [
            decision(action="route", agent="tone"),
            decision(action="route", agent="tone", follow_up=True),
        ]
    )
    graph = _graph(script)
    thread = uuid.uuid4().hex
    _run(graph, "make this formal: x", thread)
    _run(graph, "shorter please", thread)
    assert "User: make this formal: x\nAssistant: tone did it" in script.prompts[1]


def test_reserved_names_rejected() -> None:
    for name in ("supervisor", "recall", "remember"):
        try:
            build_graph([_echo_agent(name)])
        except ValueError as exc:
            assert "reserved" in str(exc)
        else:
            raise AssertionError(f"{name} should be reserved")
