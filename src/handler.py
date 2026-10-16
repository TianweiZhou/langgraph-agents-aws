"""Lambda entrypoint: API Gateway (HTTP API) event -> graph -> JSON response."""

import base64
import json
import time
import uuid
from functools import cache
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, ValidationError

from orchestrator.graph import Graph, build_default_graph
from orchestrator.state import new_turn
from shared.checkpointer import checkpoint_thread_id, make_checkpointer
from shared.config import get_settings
from shared.logging import configure_logging, get_logger, log_context
from shared.tracing import trace_request
from shared.usage import Usage, track_usage

configure_logging(get_settings().log_level)
logger = get_logger(__name__)


class InvokeRequest(BaseModel):
    user_id: str = Field(min_length=1)
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1)
    request: str = Field(min_length=1)
    # Optional hints for the supervisor, e.g. {"tone": "formal", "recipient": "Sara"}.
    inputs: dict[str, Any] | None = None


class InvokeResponse(BaseModel):
    thread_id: str
    request_id: str
    reply: str  # what the supervisor says back
    route: list[str]  # agents that worked on this message, in order
    note: str | None  # a hint for the user, if any (e.g. unknown recipient)
    result: dict[str, Any] | None  # the last agent's full output


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


def _run(req: InvokeRequest, request_id: str) -> InvokeResponse:
    state = new_turn(req.user_id, req.thread_id, req.request)
    state["inputs"] = req.inputs
    final = _graph().invoke(state, _config(req))
    result: dict[str, Any] | None = final["result"]
    done: list[dict[str, Any]] = final["done"]
    return InvokeResponse(
        thread_id=req.thread_id,
        request_id=request_id,
        reply=final["reply"] or "",
        route=[str(d["agent"]) for d in done],
        note=(result or {}).get("note"),
        result=result,
    )


def _log_summary(
    *,
    started: float,
    status: int,
    route: list[str],
    usage: Usage,
    user_id: str | None = None,
    error: str | None = None,
) -> None:
    """One line per request: what the admin dashboard reads."""
    with log_context(agent=route[-1] if route else None):
        logger.info(
            "Request summary",
            extra={
                "user_id": user_id,
                "route": " > ".join(route) or "supervisor",
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
    request_id = getattr(context, "aws_request_id", None) or uuid.uuid4().hex
    try:
        req = _parse_body(event)
    except (ValidationError, ValueError) as exc:
        with log_context(request_id=request_id):
            logger.warning("Invalid request", extra={"error": str(exc)})
            _log_summary(
                started=started, status=400, route=[], usage=Usage(), error="Invalid request body"
            )
        return _response(400, {"error": "Invalid request body", "request_id": request_id})

    with (
        log_context(request_id=request_id, thread_id=req.thread_id),
        track_usage() as usage,
        trace_request(),
    ):
        logger.info("Request received", extra={"user_id": req.user_id})
        try:
            resp = _run(req, request_id)
        except Exception:
            logger.exception("Request failed")
            _log_summary(
                started=started,
                status=500,
                route=[],
                usage=usage,
                user_id=req.user_id,
                error="Internal error",
            )
            return _response(500, {"error": "Internal error", "request_id": request_id})
        _log_summary(
            started=started, status=200, route=resp.route, usage=usage, user_id=req.user_id
        )
        return _response(200, resp.model_dump())
