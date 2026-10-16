from pydantic import BaseModel

from shared.model_fields import OptionalText, TextList


class _Out(BaseModel):
    name: OptionalText = None
    facts: TextList = []


def test_placeholder_text_becomes_none() -> None:
    for placeholder in ("null", "None", " N/A ", "", "unknown"):
        assert _Out.model_validate({"name": placeholder}).name is None
    assert _Out.model_validate({"name": "Sara"}).name == "Sara"


def test_placeholders_dropped_from_lists() -> None:
    assert _Out.model_validate({"facts": ["null", "likes coffee", ""]}).facts == ["likes coffee"]
    assert _Out.model_validate({"facts": "null"}).facts == []
