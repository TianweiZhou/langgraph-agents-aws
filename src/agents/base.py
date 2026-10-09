"""Contract every agent must follow to be plugged into the orchestrator."""

from dataclasses import dataclass

from pydantic import BaseModel

from orchestrator.state import Node


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

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent name must not be empty")
        if not self.description.strip():
            raise ValueError(f"Agent '{self.name}' needs a description")
