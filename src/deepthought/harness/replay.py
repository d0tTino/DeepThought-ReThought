"""Utilities for replaying previously recorded agent traces."""

import asyncio
from typing import Iterable, Protocol

from .record import TraceEvent


class Agent(Protocol):
    async def act(self, state: str) -> str:
        """Return an action in response to ``state``."""
        ...


async def replay(trace: Iterable[TraceEvent], agent: Agent) -> None:
    """Replay ``trace`` by invoking ``agent`` for each recorded state.

    The ``latency`` field of each :class:`~deepthought.harness.record.TraceEvent`
    indicates the delay before the next action should be replayed. This allows
    re-simulating the timing of the original interaction when running tests or
    evaluations.
    """

    for event in trace:
        await agent.act(event.state)
        if event.latency:
            await asyncio.sleep(event.latency)
