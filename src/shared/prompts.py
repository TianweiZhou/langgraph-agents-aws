"""Single entry point for loading prompts.

Agent prompts live in `agents/<agent>/prompt.md`; orchestrator prompts (the
supervisor) in `orchestrator/prompts/<name>.md`. Later this can load a pinned
version from Bedrock Prompt Management without changing calling code.
"""

from functools import cache
from pathlib import Path
from typing import Literal

SRC_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = SRC_DIR / "agents"
ORCHESTRATOR_PROMPTS_DIR = SRC_DIR / "orchestrator" / "prompts"
LOCAL_VERSION = "local"

PromptKind = Literal["agent", "orchestrator"]


def _path(name: str, kind: PromptKind) -> Path:
    if kind == "agent":
        return AGENTS_DIR / name / "prompt.md"
    return ORCHESTRATOR_PROMPTS_DIR / f"{name}.md"


@cache
def load_prompt(name: str, version: str = LOCAL_VERSION, *, kind: PromptKind = "agent") -> str:
    """Return the prompt text for `name` at `version`."""
    if version != LOCAL_VERSION:
        raise NotImplementedError(f"Only '{LOCAL_VERSION}' prompts are supported, got '{version}'")
    path = _path(name, kind)
    if not path.is_file():
        raise FileNotFoundError(f"No {kind} prompt '{name}' at {path}")
    return path.read_text(encoding="utf-8").strip()
