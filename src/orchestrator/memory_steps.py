"""Graph steps that run around every agent: recall before, remember after.

Recall loads what is known about the recipient (no model call). Remember asks
the model what the user just revealed about them and saves it. Neither step
can fail a request: errors are logged and the reply still goes out.
"""

from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel, Field

from orchestrator.state import GraphState, Node, StateUpdate
from shared.logging import get_logger
from shared.memory import PeopleMemory, get_people_memory
from shared.model_fields import OptionalText, TextList
from shared.models import get_chat_model
from shared.prompts import load_prompt

logger = get_logger(__name__)

MEMORY_AGENT = "memory"  # handles its own saving; remember skips it


def recipient_of(state: GraphState) -> str | None:
    recipient = (state["inputs"] or {}).get("recipient")
    return str(recipient).strip() or None if recipient else None


def make_recall_node(memory: Callable[[], PeopleMemory] | None = None) -> Node:
    def recall(state: GraphState) -> StateUpdate:
        name = recipient_of(state)
        if name is None:
            return {"memory": None}
        try:
            people = (memory or get_people_memory)()
            contact = people.get_contact(state["user_id"], name)
            events = people.recent_events(state["user_id"], name)
        except Exception:
            logger.warning("Recall failed; continuing without memory", exc_info=True)
            return {"memory": None}
        logger.info("Recall", extra={"recipient": name, "known": contact is not None})
        return {
            "memory": {
                "recipient": name,
                "known": contact is not None,
                "profile": contact.describe() if contact else None,
                "events": [f"{e.at[:10]}: {e.summary}" for e in events],
            }
        }

    return recall


class MemoryNote(BaseModel):
    """What the user revealed in this turn about the person they are writing to."""

    relationship: OptionalText = Field(
        default=None, description="Their relationship to the user, if stated (e.g. 'manager')."
    )
    tone_notes: OptionalText = Field(
        default=None, description="How the user wants to sound with them, if stated."
    )
    new_facts: TextList = Field(
        default_factory=lambda: [], description="Other lasting facts about them the user stated."
    )
    event: OptionalText = Field(
        default=None, description="One line on what happened in this exchange, or null."
    )


Extract = Callable[[str], MemoryNote]


def extract_with_model(prompt: str) -> MemoryNote:
    model = get_chat_model().with_structured_output(MemoryNote)
    return cast(MemoryNote, model.invoke(prompt))


def _remember_prompt(state: GraphState, name: str) -> str:
    known: dict[str, Any] = state["memory"] or {}
    inputs = state["inputs"] or {}
    reply = state["messages"][-1].text if state["messages"] else ""
    return load_prompt("remember", kind="orchestrator").format(
        recipient=name,
        profile=known.get("profile") or "(nothing yet)",
        message=state["request"],
        situation=inputs.get("context") or "(not given)",
        reply=reply,
    )


def make_remember_node(
    memory: Callable[[], PeopleMemory] | None = None,
    extract: Extract | None = None,
) -> Node:
    def remember(state: GraphState) -> StateUpdate:
        name = recipient_of(state)
        if name is None or state["agent"] == MEMORY_AGENT:
            return {}
        try:
            note = (extract or extract_with_model)(_remember_prompt(state, name))
            people = (memory or get_people_memory)()
            profile_changed = bool(note.relationship or note.tone_notes or note.new_facts)
            if profile_changed:
                people.update_contact(
                    state["user_id"],
                    name,
                    relationship=note.relationship,
                    tone_notes=note.tone_notes,
                    facts=note.new_facts,
                )
            if note.event:
                people.add_event(state["user_id"], name, note.event, state["thread_id"])
            logger.info(
                "Remembered",
                extra={"recipient": name, "profile_changed": profile_changed, "event": note.event},
            )
        except Exception:
            logger.warning("Remember failed; reply already sent", exc_info=True)
        return {}

    return remember
