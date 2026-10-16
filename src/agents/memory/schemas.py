from typing import Literal

from pydantic import BaseModel, Field

from shared.model_fields import OptionalText, TextList


class MemoryInput(BaseModel):
    text: str = Field(min_length=1, description="The user's message about a person.")


class MemoryCommand(BaseModel):
    """What the user wants done with their memory about people (model output)."""

    action: Literal["remember", "forget", "lookup", "list"] = Field(
        description="remember = save details, forget = delete a person, "
        "lookup = what is known about a person, list = everyone known."
    )
    name: OptionalText = Field(default=None, description="The person's name, if one is named.")
    relationship: OptionalText = Field(default=None, description="Relationship, if stated.")
    tone_notes: OptionalText = Field(default=None, description="How to sound with them, if stated.")
    facts: TextList = Field(
        default_factory=lambda: [], description="Other details about them the user stated."
    )


class MemoryOutput(BaseModel):
    message: str = Field(description="Confirmation or answer for the user.")
    action: str = Field(description="The action that was carried out.")
