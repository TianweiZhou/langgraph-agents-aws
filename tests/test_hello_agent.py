from agents.hello.agent import agent, run
from agents.hello.schemas import HelloOutput
from helpers import make_state


def test_hello_echoes_request() -> None:
    update = run(make_state(request="ping"))
    assert update == {"result": {"message": "Hello from the hello agent! You said: ping"}}


def test_hello_output_matches_schema() -> None:
    update = run(make_state(request="ping"))
    assert update.get("result") is not None
    HelloOutput.model_validate(update.get("result"))


def test_hello_keeps_braces_in_request() -> None:
    update = run(make_state(request="{not a placeholder}"))
    assert update == {
        "result": {"message": "Hello from the hello agent! You said: {not a placeholder}"}
    }


def test_hello_contract() -> None:
    assert agent.name == "hello"
    assert agent.output_model is HelloOutput
