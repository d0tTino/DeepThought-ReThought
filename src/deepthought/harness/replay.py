"""Replay recorded traces for evaluation."""

from typing import Iterable, Protocol

from .record import TraceEvent


class Agent(Protocol):
    async def act(self, state: str) -> str:
        """Return an action in response to ``state``."""
        ...


async def replay(trace: Iterable[TraceEvent], agent: Agent) -> None:
    for event in trace:
        _ = await agent.act(event.state)
