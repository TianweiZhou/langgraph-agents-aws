"""Typed state shared by every node in the orchestrator graph."""

from typing import Annotated, Any, Protocol, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """Full graph state, saved per thread by the checkpointer.

    Every field except `user_id`, `thread_id`, and `messages` is per turn: the
    handler resets it with `new_turn` on each request.
    """

    user_id: str
    thread_id: str
    request: str  # the user's message
    inputs: dict[str, Any] | None  # what the supervisor worked out (tone, text, recipient, ...)
    agent: str | None  # next agent to run; None once the supervisor finishes
    more: bool  # the supervisor planned another step after the current agent
    decision: dict[str, Any] | None  # the supervisor's latest decision, for tracing
    done: list[dict[str, Any]]  # agents run this turn: {"agent", "reply"}
    reply: str | None  # latest user-facing text (from an agent or the supervisor)
    result: dict[str, Any] | None  # latest agent's full output
    memory: dict[str, Any] | None  # what recall found about the recipient
    saved: dict[str, Any] | None  # what remember saved
    messages: Annotated[list[AnyMessage], add_messages]  # conversation so far


class StateUpdate(TypedDict, total=False):
    """Partial update a node returns. LangGraph merges it into `GraphState`."""

    inputs: dict[str, Any] | None
    agent: str | None
    more: bool
    decision: dict[str, Any] | None
    done: list[dict[str, Any]]
    reply: str | None
    result: dict[str, Any] | None
    memory: dict[str, Any] | None
    saved: dict[str, Any] | None
    messages: list[AnyMessage]


class Node(Protocol):
    """A graph node: reads the full state, returns a partial update."""

    def __call__(self, state: GraphState) -> StateUpdate: ...


def new_turn(user_id: str, thread_id: str, request: str) -> GraphState:
    """Graph input for a new user message: resets every per-turn field."""
    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "request": request,
        "inputs": None,
        "agent": None,
        "more": False,
        "decision": None,
        "done": [],
        "reply": None,
        "result": None,
        "memory": None,
        "saved": None,
        "messages": [HumanMessage(request)],  # appended to the thread's history
    }
