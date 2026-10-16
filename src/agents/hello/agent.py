"""Hello-world agent: proves routing and wiring end to end without calling a model."""

from langchain_core.messages import AIMessage

from agents.base import Agent, parse_input
from agents.hello.schemas import HelloInput, HelloOutput
from orchestrator.state import GraphState, StateUpdate
from shared.prompts import load_prompt

NAME = "hello"


def run(state: GraphState) -> StateUpdate:
    payload = parse_input(HelloInput, state)
    template = load_prompt(NAME)
    output = HelloOutput(message=template.format(text=payload.text))
    return {"result": output.model_dump(), "messages": [AIMessage(output.message, name=NAME)]}


agent = Agent(
    name=NAME,
    description="Greets the user and handles small talk or anything no other agent covers.",
    input_model=HelloInput,
    output_model=HelloOutput,
    node=run,
)
