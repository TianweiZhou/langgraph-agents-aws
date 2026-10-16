"""Contract every agent must follow to be plugged into the orchestrator."""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from orchestrator.state import GraphState, Node

IntentMatcher = Callable[[GraphState], bool]


class InvalidAgentInput(ValueError):
    """The request does not carry what the chosen agent needs.

    `ask` is a short question for the user about what is missing.
    """

    def __init__(self, message: str, ask: str | None = None) -> None:
        super().__init__(message)
        self.ask = ask or "I need a bit more detail to do that. Could you say more?"


@dataclass(frozen=True)
class Agent:
    """An agent the orchestrator can route to.

    Each agent module builds one of these and keeps its own prompt, schemas,
    and tests, so adding an agent never touches another agent.
    """

    name: str  # unique id, also used as the graph node name
    description: str  # short text the router can use to pick this agent
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    node: Node
    matches: IntentMatcher | None = None  # optional intent check for the router

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent name must not be empty")
        if not self.description.strip():
            raise ValueError(f"Agent '{self.name}' needs a description")


def parse_input[T: BaseModel](model: type[T], state: GraphState) -> T:
    """Build an agent's input from the request text plus its `inputs` fields."""
    data = {"text": state["request"], **(state["inputs"] or {})}
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        fields = ", ".join(str(err["loc"][0]) for err in exc.errors() if err["loc"])
        raise InvalidAgentInput(f"Invalid inputs for {model.__name__}: {fields}") from exc
