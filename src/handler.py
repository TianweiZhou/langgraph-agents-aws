"""Lambda entrypoint: API Gateway (HTTP API) event -> graph -> JSON response."""

import base64
import json
import uuid
from functools import cache
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from orchestrator.graph import Graph, build_default_graph
from orchestrator.router import UnknownAgentError
from orchestrator.state import GraphState
from shared.config import get_settings
from shared.logging import configure_logging, get_logger, log_context

configure_logging(get_settings().log_level)
logger = get_logger(__name__)


class InvokeRequest(BaseModel):
    user_id: str = Field(min_length=1)
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1)
    request: str = Field(min_length=1)
    agent: str | None = None  # optional: ask for a specific agent


class InvokeResponse(BaseModel):
    thread_id: str
    agent: str
    result: dict[str, Any]


@cache
def _graph() -> Graph:
    # Built once per Lambda container and reused across warm invocations.
    return build_default_graph()


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


def _run(req: InvokeRequest) -> InvokeResponse:
    state: GraphState = {
        "user_id": req.user_id,
        "thread_id": req.thread_id,
        "request": req.request,
        "agent": req.agent,
        "result": None,
    }
    final = _graph().invoke(state)
    return InvokeResponse(thread_id=req.thread_id, agent=final["agent"], result=final["result"])


def lambda_handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    try:
        req = _parse_body(event)
    except (ValidationError, ValueError) as exc:
        logger.warning("Invalid request", extra={"error": str(exc)})
        return _response(400, {"error": "Invalid request body"})

    with log_context(thread_id=req.thread_id):
        logger.info("Request received", extra={"user_id": req.user_id})
        try:
            resp = _run(req)
        except UnknownAgentError as exc:
            logger.warning("Unknown agent", extra={"error": str(exc)})
            return _response(400, {"error": str(exc)})
        except Exception:
            logger.exception("Request failed")
            return _response(500, {"error": "Internal error"})
        with log_context(agent=resp.agent):
            logger.info("Request completed")
        return _response(200, resp.model_dump())
