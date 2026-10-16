"""Router: decides which agent handles a request.

The graph only depends on the `Router` protocol, so the simple rule-based
router below can later be swapped for an LLM-driven supervisor.
"""

from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import AIMessage

from agents.base import Agent
from orchestrator.state import GraphState, Node, StateUpdate


class UnknownAgentError(ValueError):
    """The request asked for an agent that is not registered."""


class Router(Protocol):
    def choose(self, state: GraphState) -> str:
        """Return the name of the agent that should handle `state`."""
        ...


def last_agent(state: GraphState) -> str | None:
    """Name of the agent that sent the most recent reply in this thread."""
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return message.name
    return None


class RuleRouter:
    """Picks, in order: the agent the caller asked for, the first agent whose
    intent check matches, the agent that answered last in this thread (so
    follow-ups like "make it shorter" stay with it), then the default agent."""

    def __init__(self, agents: Sequence[Agent], default: str) -> None:
        self._agents = tuple(agents)
        self._names = frozenset(a.name for a in agents)
        if default not in self._names:
            raise UnknownAgentError(f"Default agent '{default}' is not registered")
        self._default = default

    def choose(self, state: GraphState) -> str:
        requested = state["agent"]
        if requested is not None:
            if requested not in self._names:
                raise UnknownAgentError(f"Unknown agent '{requested}'")
            return requested
        for agent in self._agents:
            if agent.matches is not None and agent.matches(state):
                return agent.name
        previous = last_agent(state)
        if previous in self._names:
            return previous
        return self._default


def make_router_node(router: Router) -> Node:
    def route(state: GraphState) -> StateUpdate:
        return {"agent": router.choose(state)}

    return route
