"""The agents the orchestrator knows about. Add new agents here."""

from agents.base import Agent
from agents.hello.agent import agent as hello
from agents.memory.agent import agent as memory
from agents.tone.agent import agent as tone

# Order matters: the router checks intent in this order before using the default.
ALL_AGENTS: tuple[Agent, ...] = (tone, memory, hello)
DEFAULT_AGENT = hello.name
