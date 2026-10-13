from pydantic import BaseModel, Field


class ToneInput(BaseModel):
    text: str = Field(min_length=1, description="The text to rewrite.")
    tone: str = Field(min_length=1, description="Target tone, e.g. 'formal' or 'friendly'.")
    audience: str | None = Field(default=None, description="Who will read the result.")


class ToneOutput(BaseModel):
    rewritten_text: str = Field(description="The text in the target tone, same meaning.")
    tone: str = Field(description="The tone that was applied.")
