"""Record agent interactions to an in-memory list."""

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class TraceEvent:
    state: str
    action: str
    reward: float
    latency: float
    timestamp: datetime
    timestamp_delta: float


def record_event(trace: List[TraceEvent], state: str, action: str, reward: float, latency: float) -> None:
    """Append a :class:`TraceEvent` to ``trace`` with a computed timestamp delta."""
    now = datetime.utcnow()
    delta = (now - trace[-1].timestamp).total_seconds() if trace else 0.0
    trace.append(
        TraceEvent(
            state=state,
            action=action,
            reward=reward,
            latency=latency,
            timestamp=now,
            timestamp_delta=delta,
        )
    )
