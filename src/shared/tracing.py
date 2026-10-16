"""One "Trace step" log line per graph step, so the admin dashboard can replay a request.

Each line carries the request id, step number, node, what arrived, what the node
produced, and where the flow goes next. Message text is included only when
TRACE_CONTENT is on (dev), cut to a short length.
"""

import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from shared.config import get_settings
from shared.logging import get_logger

logger = get_logger(__name__)

MAX_TEXT = 300  # characters kept per text value when content tracing is on
TEXT_KEYS = frozenset(
    {"request", "text", "reply", "context", "rewritten_text", "message", "profile", "event"}
)


@dataclass
class _Trace:
    step: int = 0


_current: ContextVar[_Trace | None] = ContextVar("trace", default=None)


@contextmanager
def trace_request() -> Generator[None]:
    """Number the steps of one request from 1."""
    token = _current.set(_Trace())
    try:
        yield
    finally:
        _current.reset(token)


def _redact(value: Any, *, content: bool, key: str | None = None) -> Any:
    if isinstance(value, dict):
        items: dict[Any, Any] = value  # pyright: ignore[reportUnknownVariableType]
        return {k: _redact(v, content=content, key=str(k)) for k, v in items.items()}
    if isinstance(value, list):
        values: list[Any] = value  # pyright: ignore[reportUnknownVariableType]
        return [_redact(v, content=content, key=key) for v in values]
    if isinstance(value, str) and key in TEXT_KEYS:
        if not content:
            return f"<{len(value)} chars>"
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "…"
    return value


def trace_step(
    node: str,
    *,
    received: dict[str, Any],
    produced: dict[str, Any],
    next_node: str,
    started: float,
) -> None:
    trace = _current.get()
    if trace is None:
        return
    trace.step += 1
    content = get_settings().trace_content
    logger.info(
        "Trace step",
        extra={
            "step": trace.step,
            "node": node,
            "next": next_node,
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "received": json.dumps(_redact(received, content=content), default=str),
            "produced": json.dumps(_redact(produced, content=content), default=str),
        },
    )
