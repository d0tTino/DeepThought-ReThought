"""Replay recorded traces for evaluation."""

import asyncio
from typing import Iterable, Optional, Protocol

from ..eda.publisher import Publisher
from .record import TraceEvent


class Agent(Protocol):
    async def act(self, state: str) -> str:
        """Return an action in response to ``state``."""
        ...


async def replay(trace: Iterable[TraceEvent], agent: Agent, publisher: Optional[Publisher] = None) -> None:
    """Replay a trace, preserving original timing."""
    for event in trace:
        await asyncio.sleep(event.timestamp_delta)
        if publisher is not None:
            await publisher.publish("chat.raw", event.state)
        _ = await agent.act(event.state)
