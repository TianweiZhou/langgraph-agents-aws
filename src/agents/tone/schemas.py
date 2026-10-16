from pydantic import BaseModel, Field


class ToneInput(BaseModel):
    text: str = Field(min_length=1, description="The text to rewrite.")
    tone: str | None = Field(
        default=None,
        min_length=1,
        description="Target tone, e.g. 'formal'. Optional on a follow-up in the same thread.",
    )
    audience: str | None = Field(default=None, description="Who will read the result.")
    recipient: str | None = Field(default=None, description="The person being written to.")
    context: str | None = Field(
        default=None, description="The situation, e.g. what the recipient said."
    )


class ToneOutput(BaseModel):
    rewritten_text: str = Field(description="The text in the target tone, same meaning.")
    tone: str | None = Field(description="The tone that was asked for, if any.")
    note: str | None = Field(default=None, description="A hint for the user, if any.")
