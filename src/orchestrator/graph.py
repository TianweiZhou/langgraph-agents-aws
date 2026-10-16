"""Builds the orchestrator graph: START -> router -> recall -> agent -> remember -> END."""

from collections.abc import Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.base import Agent
from agents.registry import ALL_AGENTS, DEFAULT_AGENT
from orchestrator.memory_steps import make_recall_node, make_remember_node
from orchestrator.router import Router, RuleRouter, make_router_node
from orchestrator.state import GraphState, Node, StateUpdate
from orchestrator.supervisor import SupervisorRouter
from shared.config import get_settings
from shared.logging import get_logger, log_context

ROUTER_NODE = "router"
RECALL_NODE = "recall"
REMEMBER_NODE = "remember"
RESERVED = frozenset({ROUTER_NODE, RECALL_NODE, REMEMBER_NODE})

logger = get_logger(__name__)

Graph = CompiledStateGraph[GraphState, None, GraphState, GraphState]


def _selected_agent(state: GraphState) -> str:
    agent = state["agent"]
    if agent is None:
        raise RuntimeError("Router did not choose an agent")
    return agent


def _with_agent_logging(agent: Agent) -> Node:
    """Run the agent's node with `agent` attached to every log line it writes."""

    def node(state: GraphState) -> StateUpdate:
        with log_context(agent=agent.name):
            logger.info("Agent started")
            update = agent.node(state)
            logger.info("Agent finished")
            return update

    return node


def build_graph(
    agents: Sequence[Agent],
    router: Router,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> Graph:
    names = [a.name for a in agents]
    if len(names) != len(set(names)):
        raise ValueError(f"Agent names must be unique: {names}")
    if reserved := RESERVED.intersection(names):
        raise ValueError(f"{sorted(reserved)} are reserved and cannot be agent names")

    builder = StateGraph(GraphState)
    builder.add_node(ROUTER_NODE, make_router_node(router))
    builder.add_node(RECALL_NODE, make_recall_node())
    builder.add_node(REMEMBER_NODE, make_remember_node())
    for agent in agents:
        builder.add_node(agent.name, _with_agent_logging(agent))
        builder.add_edge(agent.name, REMEMBER_NODE)

    builder.add_edge(START, ROUTER_NODE)
    builder.add_edge(ROUTER_NODE, RECALL_NODE)
    builder.add_conditional_edges(RECALL_NODE, _selected_agent, {n: n for n in names})
    builder.add_edge(REMEMBER_NODE, END)
    return builder.compile(checkpointer=checkpointer)


def default_router() -> Router:
    """LLM supervisor when SUPERVISOR_ENABLED is set, keyword rules otherwise."""
    if get_settings().supervisor_enabled:
        return SupervisorRouter(ALL_AGENTS, default=DEFAULT_AGENT)
    return RuleRouter(ALL_AGENTS, default=DEFAULT_AGENT)


def build_default_graph(checkpointer: BaseCheckpointSaver[str] | None = None) -> Graph:
    return build_graph(ALL_AGENTS, default_router(), checkpointer)
