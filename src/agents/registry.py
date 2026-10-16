"""The agents the supervisor can route to. Add new agents here."""

from agents.base import Agent
from agents.memory.agent import agent as memory
from agents.tone.agent import agent as tone

ALL_AGENTS: tuple[Agent, ...] = (tone, memory)
