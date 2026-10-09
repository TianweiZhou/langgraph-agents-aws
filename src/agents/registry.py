"""The agents the orchestrator knows about. Add new agents here."""

from agents.base import Agent
from agents.hello.agent import agent as hello

ALL_AGENTS: tuple[Agent, ...] = (hello,)
DEFAULT_AGENT = hello.name
