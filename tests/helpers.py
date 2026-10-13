from typing import Any

from orchestrator.state import GraphState


def make_state(
    request: str = "hi", agent: str | None = None, inputs: dict[str, Any] | None = None
) -> GraphState:
    return {
        "user_id": "u1",
        "thread_id": "t1",
        "request": request,
        "inputs": inputs,
        "agent": agent,
        "result": None,
    }
