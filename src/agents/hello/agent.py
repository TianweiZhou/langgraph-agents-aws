"""Hello-world agent: proves routing and wiring end to end without calling a model."""

from agents.base import Agent
from agents.hello.schemas import HelloInput, HelloOutput
from orchestrator.state import GraphState, StateUpdate
from shared.prompts import load_prompt

NAME = "hello"


def run(state: GraphState) -> StateUpdate:
    payload = HelloInput(text=state["request"])
    template = load_prompt(NAME)
    output = HelloOutput(message=template.format(text=payload.text))
    return {"result": output.model_dump()}


agent = Agent(
    name=NAME,
    description="Fallback agent that greets the user and echoes their request.",
    input_model=HelloInput,
    output_model=HelloOutput,
    node=run,
)
