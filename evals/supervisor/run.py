"""Check the supervisor's first decision for each message against real Bedrock.

Usage (from the repo root):
    AWS_PROFILE=dev BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0 \
        PYTHONPATH=src uv run python -m evals.supervisor.run
"""

import json
import sys
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from pydantic import BaseModel, Field

from agents.registry import ALL_AGENTS
from orchestrator.state import GraphState, new_turn
from orchestrator.supervisor import Supervisor

HERE = Path(__file__).resolve().parent
PASS_RATE = 0.9
UNSET = "unset"  # a case field left out is not checked


class Case(BaseModel):
    id: str
    message: str
    history: list[tuple[str, str]] = Field(default_factory=list[tuple[str, str]])
    action: Literal["route", "respond"] = "route"
    agent: str | None = None
    more_steps: bool | None = None
    tone: str | None = UNSET
    text_contains: str | None = None
    recipient: str | None = UNSET


def load_cases() -> list[Case]:
    lines = (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    return [Case.model_validate_json(line) for line in lines if line.strip()]


def to_state(case: Case) -> GraphState:
    state = new_turn("eval", f"eval-{case.id}", case.message)
    earlier: list[AnyMessage] = [
        HumanMessage(text) if speaker == "user" else AIMessage(text, name=speaker)
        for speaker, text in case.history
    ]
    state["messages"] = [*earlier, *state["messages"]]
    return state


def _matches(expected: str | None, got: str | None) -> bool:
    if expected is None or got is None:
        return expected is None and got is None
    return expected.lower() in got.lower()


def check(case: Case, supervisor: Supervisor) -> list[str]:
    """Problems with the decision for `case`; empty means pass."""
    decision, model_call = supervisor.decide(to_state(case))
    problems: list[str] = []
    if not model_call:
        problems.append("fell back (no usable model decision)")
    if decision.action != case.action:
        problems.append(f"action={decision.action}, expected {case.action}")
    if case.action == "route" and decision.agent != case.agent:
        problems.append(f"agent={decision.agent}, expected {case.agent}")
    if case.more_steps is not None and decision.more_steps != case.more_steps:
        problems.append(f"more_steps={decision.more_steps}, expected {case.more_steps}")
    if case.tone != UNSET and not _matches(case.tone, decision.tone):
        problems.append(f"tone={decision.tone!r}, expected {case.tone!r}")
    if case.text_contains and case.text_contains not in str(decision.text):
        problems.append(f"text={decision.text!r}, expected to contain {case.text_contains!r}")
    if case.recipient != UNSET and not _matches(case.recipient, decision.recipient):
        problems.append(f"recipient={decision.recipient!r}, expected {case.recipient!r}")
    target = decision.agent or decision.action
    print(f"{'PASS' if not problems else 'FAIL'}  {case.id}  -> {target}  ({decision.reason})")
    for problem in problems:
        print(f"      {problem}")
    return problems


def main() -> int:
    supervisor = Supervisor(ALL_AGENTS)
    cases = load_cases()
    passed = sum(not check(case, supervisor) for case in cases)
    rate = passed / len(cases)
    print(json.dumps({"passed": passed, "total": len(cases), "pass_rate": round(rate, 2)}))
    return 0 if rate >= PASS_RATE else 1


if __name__ == "__main__":
    sys.exit(main())
