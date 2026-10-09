from pydantic import BaseModel, Field


class HelloInput(BaseModel):
    text: str = Field(description="Whatever the user sent.")


class HelloOutput(BaseModel):
    message: str = Field(description="A greeting that echoes the input.")
