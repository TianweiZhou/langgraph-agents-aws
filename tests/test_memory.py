"""Long-term memory about people: store, recall/remember steps, memory agent, tone use."""

import json
import uuid
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.store.memory import InMemoryStore

import handler
from agents.base import InvalidAgentInput
from agents.memory.agent import carry_out, make_node
from agents.memory.schemas import MemoryCommand
from agents.tone import agent as tone_module
from helpers import make_state
from orchestrator import memory_steps
from orchestrator.memory_steps import MemoryNote, make_recall_node, make_remember_node
from shared.memory import Contact, PeopleMemory, get_people_memory, name_key


def _people() -> PeopleMemory:
    return PeopleMemory(InMemoryStore())


# --- store ---------------------------------------------------------------


def test_name_key_normalizes() -> None:
    assert name_key("  Sara   Lee ") == name_key("sara lee") == "sara lee"


def test_update_contact_merges_and_dedupes_facts() -> None:
    people = _people()
    people.update_contact("u1", "Sara", relationship="manager", facts=["runs Q3 reviews"])
    people.update_contact("u1", "sara", tone_notes="formal but warm", facts=["runs Q3 reviews"])
    contact = people.get_contact("u1", "SARA")
    assert contact == Contact("Sara", "manager", "formal but warm", ["runs Q3 reviews"])


def test_contacts_are_per_user() -> None:
    people = _people()
    people.update_contact("u1", "Sara", relationship="manager")
    assert people.get_contact("u2", "Sara") is None
    assert people.list_contacts("u2") == []


def test_events_keep_order_and_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shared.memory.MAX_EVENTS_PER_CONTACT", 3)
    people = _people()
    for i in range(5):
        people.add_event("u1", "Sara", f"event {i}", "t1")
    assert [e.summary for e in people.recent_events("u1", "Sara", limit=10)] == [
        "event 2",
        "event 3",
        "event 4",
    ]


def test_forget_removes_contact_and_events() -> None:
    people = _people()
    people.update_contact("u1", "Mike", relationship="brother")
    people.add_event("u1", "Mike", "asked about dinner", "t1")
    assert people.forget("u1", "mike") is True
    assert people.get_contact("u1", "Mike") is None
    assert people.recent_events("u1", "Mike") == []
    assert people.forget("u1", "Mike") is False


# --- recall / remember steps ----------------------------------------------


def test_recall_loads_profile_and_events() -> None:
    people = _people()
    people.update_contact("u1", "Sara", relationship="manager", tone_notes="formal")
    people.add_event("u1", "Sara", "asked about the late report", "t1")
    recall = make_recall_node(lambda: people)
    update = recall(make_state(inputs={"recipient": "Sara"}))
    memory = update.get("memory") or {}
    assert memory["known"] is True
    assert memory["profile"] == "Sara: manager; prefers formal"
    assert memory["events"][0].endswith("asked about the late report")


def test_recall_without_recipient_clears_memory() -> None:
    assert make_recall_node(_people)(make_state()) == {"memory": None}


def test_recall_unknown_recipient() -> None:
    update = make_recall_node(_people)(make_state(inputs={"recipient": "Zoe"}))
    assert (update.get("memory") or {})["known"] is False


def test_remember_saves_what_the_model_found() -> None:
    people = _people()
    note = MemoryNote(relationship="manager", event="Sara asked about the report")
    remember = make_remember_node(lambda: people, lambda prompt: note)
    state = make_state(request="replying to Sara, my manager", inputs={"recipient": "Sara"})
    state["agent"] = "tone"
    remember(state)
    assert (people.get_contact("u1", "Sara") or Contact("x")).relationship == "manager"
    assert [e.summary for e in people.recent_events("u1", "Sara")] == [
        "Sara asked about the report"
    ]


def test_remember_skips_memory_agent_and_missing_recipient() -> None:
    calls: list[str] = []

    def extract(prompt: str) -> MemoryNote:
        calls.append(prompt)
        return MemoryNote()

    remember = make_remember_node(_people, extract)
    no_recipient = make_state()
    no_recipient["agent"] = "tone"
    remember(no_recipient)
    memory_turn = make_state(inputs={"recipient": "Sara"})
    memory_turn["agent"] = "memory"
    remember(memory_turn)
    assert calls == []


def test_remember_never_fails_the_request() -> None:
    def broken(prompt: str) -> MemoryNote:
        raise RuntimeError("bedrock down")

    state = make_state(inputs={"recipient": "Sara"})
    state["agent"] = "tone"
    assert make_remember_node(_people, broken)(state) == {}


# --- memory agent --------------------------------------------------------


def _cmd(**fields: Any) -> MemoryCommand:
    return MemoryCommand.model_validate(fields)


def test_memory_agent_remember_lookup_list_forget() -> None:
    people = _people()
    remembered = carry_out(
        _cmd(action="remember", name="Sara", relationship="manager", tone_notes="formal"),
        people,
        "u1",
    )
    assert remembered == "Got it. Sara: manager; prefers formal."
    assert carry_out(_cmd(action="lookup", name="sara"), people, "u1").startswith("Sara: manager")
    assert "- Sara: manager" in carry_out(_cmd(action="list"), people, "u1")
    assert "forgotten Sara" in carry_out(_cmd(action="forget", name="Sara"), people, "u1")
    assert "don't know anyone" in carry_out(_cmd(action="list"), people, "u1")


def test_memory_agent_lookup_unknown_person() -> None:
    reply = carry_out(_cmd(action="lookup", name="Zoe"), _people(), "u1")
    assert "don't know Zoe yet" in reply


def test_memory_agent_needs_a_name() -> None:
    with pytest.raises(InvalidAgentInput):
        carry_out(_cmd(action="remember"), _people(), "u1")


def test_memory_agent_node_replies_and_records_message() -> None:
    people = _people()
    node = make_node(lambda prompt: _cmd(action="remember", name="Mike"), lambda: people)
    update = node(make_state(request="Mike is my brother"))
    assert (update.get("result") or {})["action"] == "remember"
    assert update.get("messages", [])[-1].name == "memory"


# --- tone agent uses memory ------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.prompt = ""

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self.prompt = messages[-1].text
        return AIMessage(content="Hi Sara, the report will be with you tomorrow.")


def test_tone_prompt_includes_recipient_profile_and_situation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(tone_module, "get_chat_model", lambda: recorder)
    state = make_state(
        request="sorry got busy, will send tmrw",
        inputs={"recipient": "Sara", "context": "she asked why the report is late"},
    )
    state["memory"] = {
        "recipient": "Sara",
        "known": True,
        "profile": "Sara: manager; prefers formal but warm",
        "events": ["2026-10-09: asked about Q2 numbers"],
    }
    result = tone_module.run(state).get("result") or {}
    assert "Known: Sara: manager; prefers formal but warm" in recorder.prompt
    assert "Earlier: 2026-10-09: asked about Q2 numbers" in recorder.prompt
    assert "she asked why the report is late" in recorder.prompt
    assert "suits the relationship" in recorder.prompt  # no tone given, recipient known
    assert result["note"] is None


def test_tone_notes_unknown_recipient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tone_module, "get_chat_model", lambda: _Recorder())
    state = make_state(request="hey", inputs={"recipient": "Zoe"})
    state["memory"] = {"recipient": "Zoe", "known": False, "profile": None, "events": []}
    result = tone_module.run(state).get("result") or {}
    assert "don't know Zoe yet" in str(result["note"])


# --- end to end through the handler ------------------------------------------


def _send(user_id: str, request: str, **extra: Any) -> dict[str, Any]:
    body = {"user_id": user_id, "thread_id": f"t-{uuid.uuid4().hex}", "request": request, **extra}
    resp = handler.lambda_handler({"body": json.dumps(body)}, None)
    return {"status": resp["statusCode"], **json.loads(resp["body"])}


def test_memory_carries_across_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Told once in one thread, used in another: the point of long-term memory."""
    user = f"u-{uuid.uuid4().hex}"

    def interpret(prompt: str) -> MemoryCommand:
        return _cmd(action="remember", name="Sara", relationship="manager")

    def extract(prompt: str) -> MemoryNote:
        return MemoryNote()

    monkeypatch.setattr("agents.memory.agent.interpret_with_model", interpret)
    told = _send(user, "Sara is my manager", agent="memory")
    assert (told["status"], told["agent"]) == (200, "memory")

    recorder = _Recorder()
    monkeypatch.setattr(tone_module, "get_chat_model", lambda: recorder)
    monkeypatch.setattr(memory_steps, "extract_with_model", extract)
    reply = _send(user, "will send tmrw", agent="tone", inputs={"recipient": "Sara"})
    assert reply["status"] == 200
    assert "Known: Sara: manager" in recorder.prompt
    assert get_people_memory().get_contact(user, "Sara") is not None
