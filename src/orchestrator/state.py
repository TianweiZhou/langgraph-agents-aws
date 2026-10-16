"""Typed state shared by every node in the orchestrator graph."""

from typing import Annotated, Any, Protocol, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """Full graph state, saved per thread by the checkpointer.

    `user_id` and `thread_id` are carried from day one so memory can be added
    later without changing the request schema.
    """

    user_id: str
    thread_id: str
    request: str
    inputs: dict[str, Any] | None  # agent-specific fields, checked by each agent
    agent: str | None  # chosen by the router
    result: dict[str, Any] | None  # agent output
    messages: Annotated[list[AnyMessage], add_messages]  # conversation so far


class StateUpdate(TypedDict, total=False):
    """Partial update a node returns. LangGraph merges it into `GraphState`."""

    agent: str | None
    inputs: dict[str, Any] | None
    result: dict[str, Any] | None
    messages: list[AnyMessage]


class Node(Protocol):
    """A graph node: reads the full state, returns a partial update."""

    def __call__(self, state: GraphState) -> StateUpdate: ...
