"""Structured JSON logging with `thread_id` and `agent` on every line."""

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

_thread_id: ContextVar[str | None] = ContextVar("thread_id", default=None)
_agent: ContextVar[str | None] = ContextVar("agent", default=None)

_CONTEXT_FIELDS = ("thread_id", "agent")
_STANDARD_ATTRS = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_base_factory = logging.getLogRecordFactory()


def _record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    """Stamp the current context onto each record when it is created."""
    record = _base_factory(*args, **kwargs)
    record.thread_id = _thread_id.get()
    record.agent = _agent.get()
    return record


logging.setLogRecordFactory(_record_factory)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **{f: getattr(record, f, None) for f in _CONTEXT_FIELDS},
        }
        # Anything passed via `extra={...}` becomes a top-level field.
        entry.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Send all logs to stdout as JSON. Safe to call more than once."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def log_context(*, thread_id: str | None = None, agent: str | None = None) -> Generator[None]:
    """Attach `thread_id` and/or `agent` to every log line inside the block."""
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    if thread_id is not None:
        tokens.append((_thread_id, _thread_id.set(thread_id)))
    if agent is not None:
        tokens.append((_agent, _agent.set(agent)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
