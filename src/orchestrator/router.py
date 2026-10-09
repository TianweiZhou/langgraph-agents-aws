"""Router: decides which agent handles a request.

The graph only depends on the `Router` protocol, so the simple rule-based
router below can later be swapped for an LLM-driven supervisor.
"""

from collections.abc import Iterable
from typing import Protocol

from agents.base import Agent
from orchestrator.state import GraphState, Node, StateUpdate


class UnknownAgentError(ValueError):
    """The request asked for an agent that is not registered."""


class Router(Protocol):
    def choose(self, state: GraphState) -> str:
        """Return the name of the agent that should handle `state`."""
        ...


class RuleRouter:
    """Uses the agent the caller asked for, otherwise falls back to a default."""

    def __init__(self, agents: Iterable[Agent], default: str) -> None:
        self._names = frozenset(a.name for a in agents)
        if default not in self._names:
            raise UnknownAgentError(f"Default agent '{default}' is not registered")
        self._default = default

    def choose(self, state: GraphState) -> str:
        requested = state["agent"]
        if requested is None:
            return self._default
        if requested not in self._names:
            raise UnknownAgentError(f"Unknown agent '{requested}'")
        return requested


def make_router_node(router: Router) -> Node:
    def route(state: GraphState) -> StateUpdate:
        return {"agent": router.choose(state)}

    return route
