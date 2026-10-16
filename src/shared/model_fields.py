"""Field types for structured model output.

Models sometimes return the text "null" or "none" instead of a real null. These
types turn such placeholders into None (or drop them from lists) so they are
never saved as data.
"""

from typing import Annotated, Any

from pydantic import BeforeValidator

_PLACEHOLDERS = {"", "null", "none", "n/a", "na", "unknown", "not stated", "not given"}


def _none_if_placeholder(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in _PLACEHOLDERS:
        return None
    return value


def _drop_placeholders(value: Any) -> Any:
    if isinstance(value, list):
        items: list[Any] = [_none_if_placeholder(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
        return [v for v in items if v is not None]
    return _none_if_placeholder(value) or []


OptionalText = Annotated[str | None, BeforeValidator(_none_if_placeholder)]
TextList = Annotated[list[str], BeforeValidator(_drop_placeholders)]
