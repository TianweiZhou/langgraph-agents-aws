"""Check the supervisor's routing decisions against real Bedrock.

Usage (from the repo root):
    AWS_PROFILE=dev BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0 \
        PYTHONPATH=src uv run python -m evals.supervisor.run
"""

import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from pydantic import BaseModel, Field

from agents.registry import ALL_AGENTS, DEFAULT_AGENT
from orchestrator.state import GraphState
from orchestrator.supervisor import SupervisorRouter

HERE = Path(__file__).resolve().parent
PASS_RATE = 0.9
UNSET = "unset"  # a case field left out is not checked


class Case(BaseModel):
    id: str
    message: str
    history: list[tuple[str, str]] = Field(default_factory=list[tuple[str, str]])  # (speaker, text)
    agent: str
    follow_up: bool | None = None
    tone: str | None = UNSET
    text_contains: str | None = None


def load_cases() -> list[Case]:
    lines = (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    return [Case.model_validate_json(line) for line in lines if line.strip()]


def to_state(case: Case) -> GraphState:
    messages: list[AnyMessage] = [
        HumanMessage(text) if speaker == "user" else AIMessage(text, name=speaker)
        for speaker, text in case.history
    ]
    messages.append(HumanMessage(case.message))
    return {
        "user_id": "eval",
        "thread_id": f"eval-{case.id}",
        "request": case.message,
        "inputs": None,
        "agent": None,
        "result": None,
        "messages": messages,
    }


def check(case: Case, router: SupervisorRouter) -> list[str]:
    """Problems with the decision for `case`; empty means pass."""
    route = router.choose(to_state(case))
    problems: list[str] = []
    if route.router != "supervisor":
        problems.append("fell back to rules")
    if route.agent != case.agent:
        problems.append(f"agent={route.agent}, expected {case.agent}")
    tone = route.inputs.get("tone")
    if case.tone != UNSET:
        expected = case.tone.lower() if case.tone else None
        got = tone.lower() if isinstance(tone, str) else None
        if (expected is None) != (got is None) or (expected and got and expected not in got):
            problems.append(f"tone={tone!r}, expected {case.tone!r}")
    text = route.inputs.get("text")
    if case.text_contains and case.text_contains not in str(text):
        problems.append(f"text={text!r}, expected to contain {case.text_contains!r}")
    print(f"{'PASS' if not problems else 'FAIL'}  {case.id}  -> {route.agent}  ({route.reason})")
    for problem in problems:
        print(f"      {problem}")
    return problems


def main() -> int:
    router = SupervisorRouter(ALL_AGENTS, default=DEFAULT_AGENT)
    cases = load_cases()
    passed = sum(not check(case, router) for case in cases)
    rate = passed / len(cases)
    print(json.dumps({"passed": passed, "total": len(cases), "pass_rate": round(rate, 2)}))
    return 0 if rate >= PASS_RATE else 1


if __name__ == "__main__":
    sys.exit(main())
