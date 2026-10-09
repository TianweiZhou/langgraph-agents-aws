"""Typed state shared by every node in the orchestrator graph."""

from typing import Any, TypedDict


class GraphState(TypedDict):
    """Full graph state.

    `user_id` and `thread_id` are carried from day one so memory can be added
    later without changing the request schema.
    """

    user_id: str
    thread_id: str
    request: str
    agent: str | None  # chosen by the router
    result: dict[str, Any] | None  # agent output


class StateUpdate(TypedDict, total=False):
    """Partial update a node returns. LangGraph merges it into `GraphState`."""

    agent: str | None
    result: dict[str, Any] | None
