"""Single entry point for loading agent prompts.

Prompts are read from `agents/<agent>/prompt.md` today. Later this can load a
pinned version from Bedrock Prompt Management without changing agent code.
"""

from functools import cache
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"
LOCAL_VERSION = "local"


@cache
def load_prompt(agent: str, version: str = LOCAL_VERSION) -> str:
    """Return the prompt text for `agent` at `version`."""
    if version != LOCAL_VERSION:
        raise NotImplementedError(f"Only '{LOCAL_VERSION}' prompts are supported, got '{version}'")
    path = AGENTS_DIR / agent / "prompt.md"
    if not path.is_file():
        raise FileNotFoundError(f"No prompt for agent '{agent}' at {path}")
    return path.read_text(encoding="utf-8").strip()
