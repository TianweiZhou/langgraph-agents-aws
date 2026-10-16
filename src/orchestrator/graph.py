"""Builds the orchestrator graph.

    START -> supervisor -> recall -> <agent> -> supervisor -> ... -> remember -> END

The supervisor is the only node the user talks to. Each agent reports back to it;
it routes again, or finishes, after which remember saves what was learned.
Every step writes a trace line (shared/tracing.py) for the admin dashboard.
"""

import time
from collections.abc import Callable, Sequence
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.base import Agent, InvalidAgentInput
from agents.registry import ALL_AGENTS
from orchestrator.memory_steps import make_recall_node, make_remember_node, recipient_of
from orchestrator.state import GraphState, Node, StateUpdate
from orchestrator.supervisor import SUPERVISOR, Decide, Supervisor
from shared.logging import get_logger, log_context
from shared.tracing import trace_step

RECALL = "recall"
REMEMBER = "remember"
RESERVED = frozenset({SUPERVISOR, RECALL, REMEMBER})

logger = get_logger(__name__)

Graph = CompiledStateGraph[GraphState, None, GraphState, GraphState]
Describe = Callable[[GraphState], dict[str, Any]]


def _after_supervisor(state: GraphState) -> str:
    return RECALL if state["agent"] else REMEMBER


def _chosen_agent(state: GraphState) -> str:
    agent = state["agent"]
    if agent is None:
        raise RuntimeError("Recall reached without a chosen agent")
    return agent


def _traced(
    name: str, node: Node, received: Describe, next_node: Callable[[GraphState], str]
) -> Node:
    """Run `node` and write one trace line: what arrived, what it produced, where next."""

    def run(state: GraphState) -> StateUpdate:
        started = time.perf_counter()
        update = node(state)
        merged: GraphState = {**state, **update}  # pyright: ignore[reportAssignmentType]
        trace_step(
            name,
            received=received(state),
            produced={k: v for k, v in update.items() if k != "messages"},
            next_node=next_node(merged),
            started=started,
        )
        return update

    return run


def _agent_step(agent: Agent) -> Node:
    """Run an agent and record its reply for the supervisor."""

    def run(state: GraphState) -> StateUpdate:
        with log_context(agent=agent.name):
            update: StateUpdate
            try:
                update = agent.node(state)
            except InvalidAgentInput as exc:
                # Missing details come back to the supervisor as a question for the user.
                update = {"reply": exc.ask, "result": {"error": str(exc)}}
        update["done"] = [*state["done"], {"agent": agent.name, "reply": update.get("reply") or ""}]
        return update

    return run


def _supervisor_received(state: GraphState) -> dict[str, Any]:
    return {"request": state["request"], "done": state["done"], "inputs": state["inputs"]}


def _agent_received(state: GraphState) -> dict[str, Any]:
    return {"request": state["request"], "inputs": state["inputs"], "memory": state["memory"]}


def _memory_received(state: GraphState) -> dict[str, Any]:
    return {"recipient": recipient_of(state), "agents_run": [d["agent"] for d in state["done"]]}


def build_graph(
    agents: Sequence[Agent],
    checkpointer: BaseCheckpointSaver[str] | None = None,
    decide: Decide | None = None,
) -> Graph:
    names = [a.name for a in agents]
    if len(names) != len(set(names)):
        raise ValueError(f"Agent names must be unique: {names}")
    if reserved := RESERVED.intersection(names):
        raise ValueError(f"{sorted(reserved)} are reserved and cannot be agent names")

    supervisor = Supervisor(agents, decide)
    builder = StateGraph(GraphState)
    builder.add_node(
        SUPERVISOR,
        _traced(SUPERVISOR, supervisor.node(), _supervisor_received, _after_supervisor),
    )
    builder.add_node(RECALL, _traced(RECALL, make_recall_node(), _memory_received, _chosen_agent))
    builder.add_node(
        REMEMBER, _traced(REMEMBER, make_remember_node(), _memory_received, lambda s: "end")
    )
    for agent in agents:
        step = _traced(agent.name, _agent_step(agent), _agent_received, lambda s: SUPERVISOR)
        builder.add_node(agent.name, step)
        builder.add_edge(agent.name, SUPERVISOR)

    builder.add_edge(START, SUPERVISOR)
    builder.add_conditional_edges(
        SUPERVISOR, _after_supervisor, {RECALL: RECALL, REMEMBER: REMEMBER}
    )
    builder.add_conditional_edges(RECALL, _chosen_agent, {n: n for n in names})
    builder.add_edge(REMEMBER, END)
    return builder.compile(checkpointer=checkpointer)


def build_default_graph(checkpointer: BaseCheckpointSaver[str] | None = None) -> Graph:
    return build_graph(ALL_AGENTS, checkpointer)
