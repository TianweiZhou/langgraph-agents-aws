"""Tone agent: rewrites text in a requested tone without changing its meaning."""

import re
from collections.abc import Sequence

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from agents.base import Agent, InvalidAgentInput, parse_input
from agents.tone.schemas import ToneInput, ToneOutput
from orchestrator.state import GraphState, StateUpdate
from shared.models import get_chat_model
from shared.prompts import load_prompt

NAME = "tone"
DEFAULT_AUDIENCE = "a general audience"
HISTORY_TURNS = 6  # earlier messages shown to the model
_INTENT = re.compile(r"\b(tone|rewrite|rephrase|reword)\b", re.IGNORECASE)


class EmptyModelResponseError(RuntimeError):
    """The model returned no text."""


def matches(state: GraphState) -> bool:
    """Pick this agent when a target tone is given or the request asks for a rewrite."""
    inputs = state["inputs"] or {}
    return "tone" in inputs or bool(_INTENT.search(state["request"]))


def format_history(messages: Sequence[AnyMessage]) -> str:
    """Earlier turns of the thread (the current message excluded), oldest first."""
    earlier = messages[:-1][-HISTORY_TURNS:]
    lines = [f"{'Assistant' if isinstance(m, AIMessage) else 'User'}: {m.text}" for m in earlier]
    return "\n".join(lines) or "(none)"


def _has_previous_rewrite(messages: Sequence[AnyMessage]) -> bool:
    return any(isinstance(m, AIMessage) and m.name == NAME for m in messages)


def run(state: GraphState) -> StateUpdate:
    payload = parse_input(ToneInput, state)
    messages = state["messages"]
    if payload.tone is None and not _has_previous_rewrite(messages):
        raise InvalidAgentInput("Invalid inputs for ToneInput: tone")
    prompt = load_prompt(NAME).format(
        history=format_history(messages),
        tone=payload.tone or "same as your previous rewrite, adjusted as asked",
        audience=payload.audience or DEFAULT_AUDIENCE,
        text=payload.text,
    )
    response = get_chat_model().invoke([HumanMessage(prompt)])
    rewritten = response.text.strip()
    if not rewritten:
        raise EmptyModelResponseError("Model returned an empty rewrite")

    output = ToneOutput(rewritten_text=rewritten, tone=payload.tone)
    return {"result": output.model_dump(), "messages": [AIMessage(rewritten, name=NAME)]}


agent = Agent(
    name=NAME,
    description="Rewrites text in a requested tone (e.g. formal, friendly) keeping its meaning.",
    input_model=ToneInput,
    output_model=ToneOutput,
    node=run,
    matches=matches,
)
