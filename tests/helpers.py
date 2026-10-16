from collections.abc import Iterable
from typing import Any

from orchestrator.state import GraphState, new_turn
from orchestrator.supervisor import Decision


def make_state(request: str = "hi", inputs: dict[str, Any] | None = None) -> GraphState:
    state = new_turn("u1", "t1", request)
    state["inputs"] = inputs
    return state


def decision(**fields: Any) -> Decision:
    return Decision.model_validate({"reason": "test", **fields})


class Script:
    """Fake supervisor model: returns the given decisions in order and records prompts."""

    def __init__(self, decisions: Iterable[Decision]) -> None:
        self._decisions = list(decisions)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> Decision:
        self.prompts.append(prompt)
        if not self._decisions:
            raise AssertionError("Supervisor asked for more decisions than scripted")
        return self._decisions.pop(0)
