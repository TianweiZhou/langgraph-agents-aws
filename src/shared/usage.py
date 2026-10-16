"""Counts model tokens per request, for every agent, via a LangChain callback."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Usage:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


_current: ContextVar[Usage | None] = ContextVar("usage", default=None)


@contextmanager
def track_usage() -> Generator[Usage]:
    """Collect token usage for model calls made inside the block."""
    usage = Usage()
    token = _current.set(usage)
    try:
        yield usage
    finally:
        _current.reset(token)


def record_usage(input_tokens: int, output_tokens: int) -> None:
    usage = _current.get()
    if usage is not None:
        usage.model_calls += 1
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens


class UsageCallback(BaseCallbackHandler):
    """Attached to chat models; records and logs tokens after each call."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generation in (g for batch in response.generations for g in batch):
            if not isinstance(generation, ChatGeneration):
                continue
            message = generation.message
            usage = message.usage_metadata if isinstance(message, AIMessage) else None
            if usage is None:
                continue
            record_usage(usage["input_tokens"], usage["output_tokens"])
            logger.info(
                "Model call finished",
                extra={
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                },
            )
