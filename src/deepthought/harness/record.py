"""Record agent interactions to an in-memory list."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple


@dataclass
class TraceEvent:
    state: str
    action: str
    reward: float
    latency: float
    timestamp: datetime


def record_event(trace: List[TraceEvent], state: str, action: str, reward: float, latency: float) -> None:
    trace.append(TraceEvent(state=state, action=action, reward=reward, latency=latency, timestamp=datetime.utcnow()))
