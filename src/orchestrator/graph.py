"""Builds the orchestrator graph: START -> router -> chosen agent -> END."""

from collections.abc import Sequence

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.base import Agent
from agents.registry import ALL_AGENTS, DEFAULT_AGENT
from orchestrator.router import Router, RuleRouter, make_router_node
from orchestrator.state import GraphState, Node, StateUpdate
from shared.logging import get_logger, log_context

ROUTER_NODE = "router"

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


def build_graph(agents: Sequence[Agent], router: Router) -> Graph:
    names = [a.name for a in agents]
    if len(names) != len(set(names)):
        raise ValueError(f"Agent names must be unique: {names}")
    if ROUTER_NODE in names:
        raise ValueError(f"'{ROUTER_NODE}' is reserved and cannot be an agent name")

    builder = StateGraph(GraphState)
    builder.add_node(ROUTER_NODE, make_router_node(router))
    for agent in agents:
        builder.add_node(agent.name, _with_agent_logging(agent))
        builder.add_edge(agent.name, END)

    builder.add_edge(START, ROUTER_NODE)
    builder.add_conditional_edges(ROUTER_NODE, _selected_agent, {n: n for n in names})
    return builder.compile()


def build_default_graph() -> Graph:
    return build_graph(ALL_AGENTS, RuleRouter(ALL_AGENTS, default=DEFAULT_AGENT))
