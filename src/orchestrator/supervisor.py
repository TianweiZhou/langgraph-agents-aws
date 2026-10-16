"""LLM supervisor: one structured model call decides the agent and extracts inputs.

Falls back to the rule-based router when the model call fails or returns an
agent that does not exist, so a bad model reply never breaks a request.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

from pydantic import BaseModel, Field

from agents.base import Agent
from orchestrator.router import Route, RuleRouter, last_agent, requested_agent
from orchestrator.state import GraphState
from shared.history import format_history
from shared.logging import get_logger
from shared.models import get_chat_model
from shared.prompts import load_prompt

logger = get_logger(__name__)


class RouteDecision(BaseModel):
    """What the supervisor model returns."""

    agent: str = Field(description="Name of exactly one agent from the list.")
    follow_up: bool = Field(description="True if the message refers to the previous reply.")
    tone: str | None = Field(default=None, description="Target tone if stated, else null.")
    text: str | None = Field(
        default=None, description="The text to work on, if mixed with an instruction."
    )
    reason: str = Field(description="One short sentence explaining the choice.")


Decide = Callable[[str], RouteDecision]


def decide_with_model(prompt: str) -> RouteDecision:
    model = get_chat_model().with_structured_output(RouteDecision)
    return cast(RouteDecision, model.invoke(prompt))


class SupervisorRouter:
    def __init__(
        self,
        agents: Sequence[Agent],
        default: str,
        decide: Decide = decide_with_model,
    ) -> None:
        self._agents = tuple(agents)
        self._fallback = RuleRouter(agents, default)
        self._default = default
        self._decide = decide

    def _prompt(self, state: GraphState) -> str:
        catalog = "\n".join(f"- {a.name}: {a.description}" for a in self._agents)
        return load_prompt("supervisor", kind="orchestrator").format(
            agents=catalog,
            history=format_history(state["messages"]),
            message=state["request"],
            default=self._default,
        )

    def choose(self, state: GraphState) -> Route:
        if requested := requested_agent(state, self._fallback.names):
            return Route(requested, reason="requested by caller", router="supervisor")
        try:
            decision = self._decide(self._prompt(state))
        except Exception:
            logger.warning("Supervisor call failed; using rules", exc_info=True)
            return self._fallback.choose(state)

        agent = decision.agent.strip().lower()
        if decision.follow_up and (previous := last_agent(state)) in self._fallback.names:
            agent = previous
        if agent not in self._fallback.names:
            logger.warning("Supervisor picked unknown agent; using rules", extra={"picked": agent})
            return self._fallback.choose(state)

        inputs: dict[str, Any] = {}
        if decision.tone:
            inputs["tone"] = decision.tone
        if decision.text:
            inputs["text"] = decision.text
        return Route(agent, inputs=inputs, reason=decision.reason, router="supervisor")
