"""Lambda entrypoint: API Gateway (HTTP API) event -> graph -> JSON response."""

import base64
import json
import time
import uuid
from functools import cache
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, ValidationError

from agents.base import InvalidAgentInput
from orchestrator.graph import Graph, build_default_graph
from orchestrator.router import UnknownAgentError
from orchestrator.state import GraphState
from shared.checkpointer import checkpoint_thread_id, make_checkpointer
from shared.config import get_settings
from shared.logging import configure_logging, get_logger, log_context
from shared.usage import Usage, track_usage

configure_logging(get_settings().log_level)
logger = get_logger(__name__)


class InvokeRequest(BaseModel):
    user_id: str = Field(min_length=1)
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1)
    request: str = Field(min_length=1)
    agent: str | None = None  # optional: ask for a specific agent
    inputs: dict[str, Any] | None = None  # agent-specific fields, e.g. {"tone": "formal"}


class InvokeResponse(BaseModel):
    thread_id: str
    agent: str
    result: dict[str, Any]


@cache
def _graph() -> Graph:
    # Built once per Lambda container and reused across warm invocations.
    return build_default_graph(make_checkpointer(get_settings()))


def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _parse_body(event: dict[str, Any]) -> InvokeRequest:
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode("utf-8")
    return InvokeRequest.model_validate_json(raw)


def _config(req: InvokeRequest) -> RunnableConfig:
    return {"configurable": {"thread_id": checkpoint_thread_id(req.user_id, req.thread_id)}}


def _run(req: InvokeRequest) -> InvokeResponse:
    state: GraphState = {
        "user_id": req.user_id,
        "thread_id": req.thread_id,
        "request": req.request,
        "inputs": req.inputs,
        "agent": req.agent,
        "result": None,
        "messages": [HumanMessage(req.request)],  # appended to the thread's history
    }
    final = _graph().invoke(state, _config(req))
    return InvokeResponse(thread_id=req.thread_id, agent=final["agent"], result=final["result"])


def _agent_after_failure(req: InvokeRequest) -> str | None:
    """The agent the router picked before a failure, for the summary log."""
    try:
        return _graph().get_state(_config(req)).values.get("agent")
    except Exception:
        return None


def _log_summary(
    *,
    started: float,
    status: int,
    agent: str | None,
    usage: Usage,
    user_id: str | None = None,
    error: str | None = None,
) -> None:
    """One line per request: what the admin dashboard reads."""
    with log_context(agent=agent):
        logger.info(
            "Request summary",
            extra={
                "user_id": user_id,
                "status": status,
                "duration_ms": round((time.perf_counter() - started) * 1000),
                "model_calls": usage.model_calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "error": error,
            },
        )


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        req = _parse_body(event)
    except (ValidationError, ValueError) as exc:
        logger.warning("Invalid request", extra={"error": str(exc)})
        _log_summary(
            started=started, status=400, agent=None, usage=Usage(), error="Invalid request body"
        )
        return _response(400, {"error": "Invalid request body"})

    with log_context(thread_id=req.thread_id), track_usage() as usage:
        logger.info("Request received", extra={"user_id": req.user_id})
        try:
            resp = _run(req)
        except (UnknownAgentError, InvalidAgentInput) as exc:
            logger.warning("Rejected request", extra={"error": str(exc)})
            status, body = 400, {"error": str(exc)}
        except Exception:
            logger.exception("Request failed")
            status, body = 500, {"error": "Internal error"}
        else:
            _log_summary(
                started=started, status=200, agent=resp.agent, usage=usage, user_id=req.user_id
            )
            return _response(200, resp.model_dump())

        _log_summary(
            started=started,
            status=status,
            agent=_agent_after_failure(req),
            usage=usage,
            user_id=req.user_id,
            error=body["error"],
        )
        return _response(status, body)
