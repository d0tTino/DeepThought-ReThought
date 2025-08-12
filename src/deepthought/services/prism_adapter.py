"""Adapter for translating Prism analytics into social graph updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .social_graph_memory import SocialGraphMemory


@dataclass
class PrismEvent:
    """Structured representation of a Prism analytics payload."""

    source: str
    target: Optional[str]
    sentiment: float
    reply_latency: Optional[float] = None
    emoji_counts: Dict[str, int] | None = None


class PrismAdapter:
    """Convert raw Prism events into calls on :class:`SocialGraphMemory`."""

    def __init__(self, memory: "SocialGraphMemory") -> None:
        self._memory = memory

    def translate(self, payload: Dict) -> PrismEvent:
        """Return a :class:`PrismEvent` from a raw payload."""

        latency = payload.get("reply_latency")
        return PrismEvent(
            source=str(payload.get("source")),
            target=payload.get("target"),
            sentiment=float(payload.get("sentiment", 0.0)),
            reply_latency=float(latency) if latency is not None else None,
            emoji_counts=payload.get("emoji_counts") or {},
        )

    async def ingest(self, payload: Dict) -> None:
        """Translate and forward a Prism event to the memory backend."""

        event = self.translate(payload)
        await self._memory.ingest_prism_event(event)

