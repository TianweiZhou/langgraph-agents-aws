"""Long-term memory about the people a user writes to, kept across all threads.

Stored in a LangGraph `BaseStore`: DynamoDB when MEMORY_TABLE is set (deployed),
in-memory otherwise (tests, local runs). Swapping in another store (for example
Bedrock AgentCore Memory) only changes `make_store`.

Layout, per user:
    ("users", <user_id>, "contacts")          key = name key  -> Contact
    ("users", <user_id>, "events", <name key>) key = timestamp -> Event
"""

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import cache
from typing import Any

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph_checkpoint_aws import DynamoDBStore

from shared.config import Settings, get_settings

MAX_EVENTS_PER_CONTACT = 20  # oldest are dropped beyond this
LIST_LIMIT = 500


def name_key(name: str) -> str:
    """Stable key for a person's name: 'Sara  Lee' and 'sara lee' match."""
    return re.sub(r"\s+", " ", name).strip().lower()


@dataclass
class Contact:
    name: str
    relationship: str | None = None
    tone_notes: str | None = None
    facts: list[str] = field(default_factory=lambda: [])

    def describe(self) -> str:
        parts = [f"{self.name}: {self.relationship or 'relationship unknown'}"]
        if self.tone_notes:
            parts.append(f"prefers {self.tone_notes}")
        parts.extend(self.facts)
        return "; ".join(parts)


@dataclass(frozen=True)
class Event:
    summary: str
    at: str  # ISO timestamp
    thread_id: str | None = None


def _contacts_ns(user_id: str) -> tuple[str, ...]:
    return ("users", user_id, "contacts")


def _events_ns(user_id: str, name: str) -> tuple[str, ...]:
    return ("users", user_id, "events", name_key(name))


def _contact(value: dict[str, Any]) -> Contact:
    return Contact(
        name=str(value["name"]),
        relationship=value.get("relationship"),
        tone_notes=value.get("tone_notes"),
        facts=[str(f) for f in value.get("facts", [])],
    )


class PeopleMemory:
    def __init__(self, store: BaseStore) -> None:
        self.store = store

    def get_contact(self, user_id: str, name: str) -> Contact | None:
        item = self.store.get(_contacts_ns(user_id), name_key(name))
        return _contact(item.value) if item else None

    def save_contact(self, user_id: str, contact: Contact) -> None:
        self.store.put(_contacts_ns(user_id), name_key(contact.name), asdict(contact))

    def update_contact(
        self,
        user_id: str,
        name: str,
        *,
        relationship: str | None = None,
        tone_notes: str | None = None,
        facts: list[str] | None = None,
    ) -> Contact:
        """Merge new details into a contact, creating it if needed."""
        contact = self.get_contact(user_id, name) or Contact(name=name.strip())
        if relationship:
            contact.relationship = relationship
        if tone_notes:
            contact.tone_notes = tone_notes
        for fact in facts or []:
            if fact and fact not in contact.facts:
                contact.facts.append(fact)
        self.save_contact(user_id, contact)
        return contact

    def list_contacts(self, user_id: str) -> list[Contact]:
        items = self.store.search(_contacts_ns(user_id), limit=LIST_LIMIT)
        return sorted((_contact(i.value) for i in items), key=lambda c: c.name.lower())

    def forget(self, user_id: str, name: str) -> bool:
        """Delete a contact and everything that happened with them."""
        existed = self.get_contact(user_id, name) is not None
        for item in self.store.search(_events_ns(user_id, name), limit=LIST_LIMIT):
            self.store.delete(_events_ns(user_id, name), item.key)
        self.store.delete(_contacts_ns(user_id), name_key(name))
        return existed

    def add_event(self, user_id: str, name: str, summary: str, thread_id: str | None) -> None:
        now = datetime.now(UTC).isoformat()
        event = Event(summary=summary, at=now, thread_id=thread_id)
        key = f"{now}#{uuid.uuid4().hex[:6]}"  # sorts by time
        self.store.put(_events_ns(user_id, name), key, asdict(event))
        items = sorted(
            self.store.search(_events_ns(user_id, name), limit=LIST_LIMIT), key=lambda i: i.key
        )
        for old in items[:-MAX_EVENTS_PER_CONTACT]:
            self.store.delete(_events_ns(user_id, name), old.key)

    def recent_events(self, user_id: str, name: str, limit: int = 5) -> list[Event]:
        """Most recent events with this person, oldest first."""
        items = sorted(
            self.store.search(_events_ns(user_id, name), limit=LIST_LIMIT), key=lambda i: i.key
        )
        return [
            Event(
                summary=str(i.value["summary"]),
                at=str(i.value["at"]),
                thread_id=i.value.get("thread_id"),
            )
            for i in items[-limit:]
        ]


def make_store(settings: Settings) -> BaseStore:
    if settings.memory_table is None:
        return InMemoryStore()
    return DynamoDBStore(table_name=settings.memory_table, region_name=settings.aws_region)


@cache
def get_people_memory() -> PeopleMemory:
    """The configured memory, built once per process."""
    return PeopleMemory(make_store(get_settings()))
