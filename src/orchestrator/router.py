"""Router: decides which agent handles a request.

The graph only depends on the `Router` protocol. Two implementations:
`RuleRouter` (keyword rules, no model call) here, and the LLM-driven
`SupervisorRouter` in `orchestrator/supervisor.py`.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage

from agents.base import Agent
from orchestrator.state import GraphState, Node, StateUpdate
from shared.logging import get_logger

logger = get_logger(__name__)


class UnknownAgentError(ValueError):
    """The request asked for an agent that is not registered."""


@dataclass(frozen=True)
class Route:
    """The router's decision: which agent, plus any inputs it worked out."""

    agent: str
    inputs: dict[str, Any] = field(default_factory=lambda: {})
    reason: str = ""
    router: str = "rules"


class Router(Protocol):
    def choose(self, state: GraphState) -> Route:
        """Decide which agent should handle `state`."""
        ...


def last_agent(state: GraphState) -> str | None:
    """Name of the agent that sent the most recent reply in this thread."""
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return message.name
    return None


def requested_agent(state: GraphState, names: frozenset[str]) -> str | None:
    """The agent the caller asked for by name, if any. Raises if it is unknown."""
    requested = state["agent"]
    if requested is not None and requested not in names:
        raise UnknownAgentError(f"Unknown agent '{requested}'")
    return requested


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

    @property
    def names(self) -> frozenset[str]:
        return self._names

    def choose(self, state: GraphState) -> Route:
        if requested := requested_agent(state, self._names):
            return Route(requested, reason="requested by caller")
        for agent in self._agents:
            if agent.matches is not None and agent.matches(state):
                return Route(agent.name, reason="intent rule matched")
        previous = last_agent(state)
        if previous in self._names:
            return Route(previous, reason="follow-up to last agent")
        return Route(self._default, reason="default agent")


def make_router_node(router: Router) -> Node:
    def route(state: GraphState) -> StateUpdate:
        decision = router.choose(state)
        logger.info(
            "Route chosen",
            extra={"router": decision.router, "chosen": decision.agent, "reason": decision.reason},
        )
        # Inputs the caller sent explicitly win over inputs the router worked out.
        inputs = {**decision.inputs, **(state["inputs"] or {})}
        return {"agent": decision.agent, "inputs": inputs or None}

    return route
