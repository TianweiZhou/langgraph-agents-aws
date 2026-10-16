"""Formats a thread's earlier turns for a prompt."""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, AnyMessage


def format_history(messages: Sequence[AnyMessage], turns: int = 6) -> str:
    """Earlier turns of the thread (the current message excluded), oldest first."""
    earlier = messages[:-1][-turns:]
    lines = [f"{'Assistant' if isinstance(m, AIMessage) else 'User'}: {m.text}" for m in earlier]
    return "\n".join(lines) or "(none)"
