"""Run the tone eval set against the real Bedrock model and score the results.

Usage (from the repo root):
    AWS_PROFILE=dev BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0 \
        PYTHONPATH=src uv run python -m evals.tone.run
"""

import json
import sys
from functools import cache
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from orchestrator.graph import Graph, build_default_graph
from orchestrator.state import GraphState
from shared.models import get_chat_model

HERE = Path(__file__).resolve().parent
PASS_SCORE = 4  # judge scores at or above this pass
PASS_RATE = 0.8  # share of cases that must pass


class Case(BaseModel):
    id: str
    text: str
    tone: str
    audience: str | None = None
    must_keep: list[str] = Field(default_factory=list)


class JudgeScore(BaseModel):
    """Scores for one rewrite."""

    tone_match: int = Field(ge=1, le=5, description="How well the requested tone is matched.")
    meaning_kept: int = Field(ge=1, le=5, description="Whether all facts are kept, none added.")
    reason: str = Field(description="One sentence explaining the scores.")


class CaseResult(BaseModel):
    id: str
    rewritten: str
    missing_terms: list[str]
    unchanged: bool
    score: JudgeScore

    @property
    def passed(self) -> bool:
        return (
            not self.missing_terms
            and not self.unchanged
            and self.score.tone_match >= PASS_SCORE
            and self.score.meaning_kept >= PASS_SCORE
        )


def load_cases() -> list[Case]:
    lines = (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    return [Case.model_validate_json(line) for line in lines if line.strip()]


@cache
def _graph() -> Graph:
    return build_default_graph()


def rewrite(case: Case) -> str:
    inputs: dict[str, Any] = {"tone": case.tone}
    if case.audience:
        inputs["audience"] = case.audience
    state: GraphState = {
        "user_id": "eval",
        "thread_id": f"eval-{case.id}",
        "request": case.text,
        "inputs": inputs,
        "agent": "tone",
        "result": None,
        "memory": None,  # recall fills this in each turn
        "messages": [HumanMessage(case.text)],
    }
    final = _graph().invoke(state)
    return str(final["result"]["rewritten_text"])


def judge(case: Case, rewritten: str) -> JudgeScore:
    prompt = (
        (HERE / "judge.md")
        .read_text(encoding="utf-8")
        .format(
            text=case.text,
            tone=case.tone,
            audience=case.audience or "a general audience",
            rewritten=rewritten,
        )
    )
    grader = get_chat_model().with_structured_output(JudgeScore)
    return cast(JudgeScore, grader.invoke(prompt))


def evaluate(case: Case) -> CaseResult:
    rewritten = rewrite(case)
    return CaseResult(
        id=case.id,
        rewritten=rewritten,
        missing_terms=[t for t in case.must_keep if t.lower() not in rewritten.lower()],
        unchanged=rewritten.strip() == case.text.strip(),
        score=judge(case, rewritten),
    )


def main() -> int:
    results = [evaluate(case) for case in load_cases()]
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"{mark}  {r.id}  tone={r.score.tone_match} meaning={r.score.meaning_kept}")
        print(f"      {r.rewritten}")
        if r.missing_terms:
            print(f"      missing: {r.missing_terms}")
        if not r.passed:
            print(f"      judge: {r.score.reason}")

    passed = sum(r.passed for r in results)
    rate = passed / len(results)
    print(json.dumps({"passed": passed, "total": len(results), "pass_rate": round(rate, 2)}))
    return 0 if rate >= PASS_RATE else 1


if __name__ == "__main__":
    sys.exit(main())
