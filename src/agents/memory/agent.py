"""Memory agent: explicit commands about people ("remember", "forget", "who is", "list")."""

from collections.abc import Callable
from typing import cast

from agents.base import Agent, InvalidAgentInput, parse_input
from agents.memory.schemas import MemoryCommand, MemoryInput, MemoryOutput
from orchestrator.state import GraphState, Node, StateUpdate
from shared.memory import PeopleMemory, get_people_memory
from shared.models import get_chat_model
from shared.prompts import load_prompt

NAME = "memory"

Interpret = Callable[[str], MemoryCommand]


def interpret_with_model(prompt: str) -> MemoryCommand:
    model = get_chat_model().with_structured_output(MemoryCommand)
    return cast(MemoryCommand, model.invoke(prompt))


def carry_out(command: MemoryCommand, people: PeopleMemory, user_id: str) -> str:
    """Do what the command asks and return the reply for the user."""
    if command.action == "list":
        contacts = people.list_contacts(user_id)
        if not contacts:
            return "I don't know anyone yet. Tell me about someone, e.g. 'Sara is my manager'."
        return "People I know:\n" + "\n".join(f"- {c.describe()}" for c in contacts)

    if not command.name:
        raise InvalidAgentInput(
            "Invalid inputs for memory: name", ask="Who do you mean? Please give me their name."
        )

    if command.action == "remember":
        contact = people.update_contact(
            user_id,
            command.name,
            relationship=command.relationship,
            tone_notes=command.tone_notes,
            facts=command.facts,
        )
        return f"Got it. {contact.describe()}."

    if command.action == "forget":
        if people.forget(user_id, command.name):
            return f"Done. I've forgotten {command.name} and our history with them."
        return f"I didn't know anyone called {command.name}."

    contact = people.get_contact(user_id, command.name)
    if contact is None:
        return (
            f"I don't know {command.name} yet. "
            f"Tell me who they are, e.g. '{command.name} is my manager'."
        )
    events = people.recent_events(user_id, command.name)
    history = "\n".join(f"- {e.at[:10]}: {e.summary}" for e in events)
    return contact.describe() + (f"\nRecently:\n{history}" if history else "")


def make_node(
    interpret: Interpret | None = None,
    memory: Callable[[], PeopleMemory] | None = None,
) -> Node:
    """Defaults are looked up at call time so tests can swap them."""

    def run(state: GraphState) -> StateUpdate:
        payload = parse_input(MemoryInput, state)
        # The supervisor hands over just the memory part of a mixed message.
        task = (state["inputs"] or {}).get("task")
        prompt = load_prompt(NAME).format(text=task or payload.text)
        command = (interpret or interpret_with_model)(prompt)
        reply = carry_out(command, (memory or get_people_memory)(), state["user_id"])
        output = MemoryOutput(message=reply, action=command.action)
        return {"result": output.model_dump(), "reply": reply}

    return run


agent = Agent(
    name=NAME,
    description=(
        "Saves, updates, forgets, or looks up what the user tells you about the people they "
        "write to (relationships, how to sound with them), e.g. 'Sara is my manager', "
        "'forget Mike', 'who is Sara?', 'who do you know?'."
    ),
    input_model=MemoryInput,
    output_model=MemoryOutput,
    node=make_node(),
)
