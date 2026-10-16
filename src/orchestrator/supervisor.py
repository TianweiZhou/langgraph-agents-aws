"""The supervisor: the only assistant the user talks to.

It runs in a loop. Each time it is reached it decides to route work to an agent,
reply to the user itself, or finish with the work already done. Agents always
report back to it. When it planned a single step, it finishes right after that
step without another model call.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agents.base import Agent
from orchestrator.state import GraphState, Node, StateUpdate
from shared.history import format_history
from shared.logging import get_logger
from shared.model_fields import OptionalText
from shared.models import get_chat_model
from shared.prompts import load_prompt

logger = get_logger(__name__)

SUPERVISOR = "supervisor"
MAX_AGENT_STEPS = 4  # per user message; stops runaway loops
FALLBACK_REPLY = "Sorry, I couldn't work that out just now. Could you try again?"


class Decision(BaseModel):
    """What the supervisor model returns."""

    action: Literal["route", "respond", "finish"] = Field(
        description="route = send work to an agent, respond = reply yourself, "
        "finish = the work already done answers the user."
    )
    agent: OptionalText = Field(default=None, description="Agent name when routing.")
    task: OptionalText = Field(
        default=None, description="When routing: the part of the message this agent handles."
    )
    more_steps: bool = Field(
        default=False, description="When routing: true if another agent is needed after this."
    )
    reply: OptionalText = Field(default=None, description="Your reply when responding.")
    follow_up: bool = Field(
        default=False, description="True if the message refers to the previous reply."
    )
    tone: OptionalText = Field(default=None, description="Target tone if stated, else null.")
    text: OptionalText = Field(
        default=None, description="The text to work on, if mixed with an instruction."
    )
    recipient: OptionalText = Field(
        default=None, description="Name of the person the user is writing or replying to."
    )
    context: OptionalText = Field(
        default=None, description="The situation the user describes, e.g. what they said."
    )
    reason: str = Field(description="One short sentence explaining the decision.")


Decide = Callable[[str], Decision]


def decide_with_model(prompt: str) -> Decision:
    model = get_chat_model().with_structured_output(Decision)
    return cast(Decision, model.invoke(prompt))


def last_agent(state: GraphState) -> str | None:
    """Who wrote the most recent reply in this thread (an agent or the supervisor)."""
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return message.name
    return None


def _record(decision: Decision, *, model_call: bool) -> dict[str, Any]:
    return {
        "action": decision.action,
        "agent": decision.agent,
        "more_steps": decision.more_steps,
        "reason": decision.reason,
        "model_call": model_call,
    }


def _finish(
    state: GraphState, reply: str | None, *, own_reply: bool, decision: dict[str, Any]
) -> StateUpdate:
    """End the turn; the reply goes into the thread's history."""
    done = state["done"]
    author = done[-1]["agent"] if done and not own_reply else SUPERVISOR
    text = reply or FALLBACK_REPLY
    return {
        "agent": None,
        "more": False,
        "decision": decision,
        "reply": text,
        "messages": [AIMessage(text, name=author)],
    }


def _no_call(action: str, reason: str) -> dict[str, Any]:
    return {"action": action, "agent": None, "reason": reason, "model_call": False}


class Supervisor:
    def __init__(self, agents: Sequence[Agent], decide: Decide | None = None) -> None:
        self._agents = tuple(agents)
        self._names = frozenset(a.name for a in agents)
        self._decide = decide

    @property
    def agent_names(self) -> frozenset[str]:
        return self._names

    def prompt(self, state: GraphState) -> str:
        catalog = "\n".join(f"- {a.name}: {a.description}" for a in self._agents)
        done = "\n".join(f"- {d['agent']}: {d['reply']}" for d in state["done"]) or "(nothing yet)"
        return load_prompt("supervisor", kind="orchestrator").format(
            agents=catalog,
            history=format_history(state["messages"]),
            message=state["request"],
            done=done,
        )

    def fallback(self, state: GraphState) -> Decision:
        """No usable model decision: first agent whose intent rule matches, else apologize."""
        for agent in self._agents:
            if agent.matches is not None and agent.matches(state):
                return Decision(action="route", agent=agent.name, reason="intent rule (fallback)")
        return Decision(action="respond", reply=FALLBACK_REPLY, reason="no rule matched (fallback)")

    def decide(self, state: GraphState) -> tuple[Decision, bool]:
        """The decision for this point in the loop, and whether the model made it."""
        try:
            decision = (self._decide or decide_with_model)(self.prompt(state))
        except Exception:
            logger.warning("Supervisor model call failed; using fallback", exc_info=True)
            return self.fallback(state), False
        agent = (decision.agent or "").strip().lower()
        if decision.action == "route" and agent not in self._names:
            previous = last_agent(state)
            if decision.follow_up and previous in self._names:
                return decision.model_copy(update={"agent": previous}), True
            logger.warning("Supervisor picked an unknown agent", extra={"picked": agent})
            return self.fallback(state), False
        return decision.model_copy(update={"agent": agent or None}), True

    def _route(self, state: GraphState, decision: Decision, record: dict[str, Any]) -> StateUpdate:
        extracted = {
            k: v
            for k, v in (
                ("tone", decision.tone),
                ("text", decision.text),
                ("recipient", decision.recipient),
                ("context", decision.context),
            )
            if v
        }
        # Keep what earlier steps this turn worked out (e.g. the recipient), but the task
        # and follow-up flag belong to this routing step only.
        carried = {
            k: v for k, v in (state["inputs"] or {}).items() if k not in ("task", "follow_up")
        }
        inputs: dict[str, Any] = {**carried, **extracted}
        if decision.task:
            inputs["task"] = decision.task
        if decision.follow_up:
            inputs["follow_up"] = True
        return {
            "agent": decision.agent,
            "more": decision.more_steps,
            "decision": record,
            "inputs": inputs or None,
        }

    def node(self) -> Node:
        def supervise(state: GraphState) -> StateUpdate:
            done = state["done"]
            if done and not state["more"]:
                record = _no_call("finish", "planned steps are done")
                return _finish(state, state["reply"], own_reply=False, decision=record)
            if len(done) >= MAX_AGENT_STEPS:
                logger.warning("Supervisor hit the step limit")
                record = _no_call("finish", "step limit reached")
                return _finish(state, state["reply"], own_reply=False, decision=record)
            decision, model_call = self.decide(state)
            record = _record(decision, model_call=model_call)
            logger.info("Supervisor decided", extra={"decision": record})
            if decision.action == "route":
                return self._route(state, decision, record)
            if decision.action == "respond":
                return _finish(state, decision.reply, own_reply=True, decision=record)
            return _finish(state, state["reply"], own_reply=False, decision=record)

        return supervise
