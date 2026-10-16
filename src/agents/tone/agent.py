"""Tone agent: rewrites text in a requested tone without changing its meaning."""

import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from agents.base import Agent, InvalidAgentInput, parse_input
from agents.tone.schemas import ToneInput, ToneOutput
from orchestrator.state import GraphState, StateUpdate
from shared.history import format_history
from shared.models import get_chat_model
from shared.prompts import load_prompt

NAME = "tone"
DEFAULT_AUDIENCE = "a general audience"
_INTENT = re.compile(r"\b(tone|rewrite|rephrase|reword)\b", re.IGNORECASE)


class EmptyModelResponseError(RuntimeError):
    """The model returned no text."""


def matches(state: GraphState) -> bool:
    """Pick this agent when a target tone is given or the request asks for a rewrite."""
    inputs = state["inputs"] or {}
    return "tone" in inputs or bool(_INTENT.search(state["request"]))


def _has_previous_rewrite(messages: Sequence[AnyMessage]) -> bool:
    return any(isinstance(m, AIMessage) and m.name == NAME for m in messages)


def _recipient_block(state: GraphState, payload: ToneInput) -> str:
    """What we know about the recipient, from recall, for the prompt."""
    memory: dict[str, Any] = state["memory"] or {}
    if not payload.recipient:
        return "(not given)"
    lines = [f"Name: {payload.recipient}"]
    lines.append(f"Known: {memory['profile']}" if memory.get("profile") else "Known: nothing yet")
    lines.extend(f"Earlier: {event}" for event in memory.get("events", []))
    return "\n".join(lines)


def _tone_instruction(payload: ToneInput, state: GraphState) -> str:
    if payload.tone:
        return payload.tone
    if _has_previous_rewrite(state["messages"]):
        return "same as your previous rewrite, adjusted as asked"
    if payload.recipient:
        return "whatever suits the relationship with the recipient described below"
    raise InvalidAgentInput("Invalid inputs for ToneInput: tone")


def _unknown_recipient_note(payload: ToneInput, state: GraphState) -> str | None:
    if payload.recipient and not (state["memory"] or {}).get("known"):
        return (
            f"I don't know {payload.recipient} yet. Tell me who they are "
            f"(e.g. '{payload.recipient} is my manager') and I'll remember for next time."
        )
    return None


def run(state: GraphState) -> StateUpdate:
    payload = parse_input(ToneInput, state)
    prompt = load_prompt(NAME).format(
        history=format_history(state["messages"]),
        tone=_tone_instruction(payload, state),
        audience=payload.audience or payload.recipient or DEFAULT_AUDIENCE,
        recipient=_recipient_block(state, payload),
        situation=payload.context or "(not given)",
        text=payload.text,
    )
    response = get_chat_model().invoke([HumanMessage(prompt)])
    rewritten = response.text.strip()
    if not rewritten:
        raise EmptyModelResponseError("Model returned an empty rewrite")

    output = ToneOutput(
        rewritten_text=rewritten,
        tone=payload.tone,
        note=_unknown_recipient_note(payload, state),
    )
    return {"result": output.model_dump(), "messages": [AIMessage(rewritten, name=NAME)]}


agent = Agent(
    name=NAME,
    description=(
        "Rewrites a message or draft in a requested tone (formal, friendly, polite, ...) "
        "without changing its meaning; also adjusts its previous rewrite on request."
    ),
    input_model=ToneInput,
    output_model=ToneOutput,
    node=run,
    matches=matches,
)
