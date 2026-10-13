"""Tone agent: rewrites text in a requested tone without changing its meaning."""

import re

from langchain_core.messages import HumanMessage

from agents.base import Agent, parse_input
from agents.tone.schemas import ToneInput, ToneOutput
from orchestrator.state import GraphState, StateUpdate
from shared.logging import get_logger
from shared.models import get_chat_model
from shared.prompts import load_prompt

NAME = "tone"
DEFAULT_AUDIENCE = "a general audience"
_INTENT = re.compile(r"\b(tone|rewrite|rephrase|reword)\b", re.IGNORECASE)

logger = get_logger(__name__)


class EmptyModelResponseError(RuntimeError):
    """The model returned no text."""


def matches(state: GraphState) -> bool:
    """Pick this agent when a target tone is given or the request asks for a rewrite."""
    inputs = state["inputs"] or {}
    return "tone" in inputs or bool(_INTENT.search(state["request"]))


def run(state: GraphState) -> StateUpdate:
    payload = parse_input(ToneInput, state)
    prompt = load_prompt(NAME).format(
        tone=payload.tone,
        audience=payload.audience or DEFAULT_AUDIENCE,
        text=payload.text,
    )
    response = get_chat_model().invoke([HumanMessage(prompt)])
    rewritten = response.text.strip()
    if not rewritten:
        raise EmptyModelResponseError("Model returned an empty rewrite")

    usage = response.usage_metadata
    logger.info(
        "Model call finished",
        extra={
            "input_tokens": usage["input_tokens"] if usage else None,
            "output_tokens": usage["output_tokens"] if usage else None,
        },
    )
    output = ToneOutput(rewritten_text=rewritten, tone=payload.tone)
    return {"result": output.model_dump()}


agent = Agent(
    name=NAME,
    description="Rewrites text in a requested tone (e.g. formal, friendly) keeping its meaning.",
    input_model=ToneInput,
    output_model=ToneOutput,
    node=run,
    matches=matches,
)
