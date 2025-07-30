"""Simple token-bucket rate limiter keyed by user ID."""

from __future__ import annotations

import time
from typing import Dict, Tuple


class UserRateLimiter:
    """Token bucket limiter for individual users."""

    def __init__(self, capacity: int, refill_time: float) -> None:
        self.capacity = capacity
        self.refill_time = refill_time
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def allow(self, user_id: str, tokens: int = 1) -> bool:
        """Return ``True`` if ``tokens`` may be consumed for ``user_id``."""
        now = time.monotonic()
        tokens_left, last = self._buckets.get(user_id, (self.capacity, now))
        tokens_left = min(self.capacity, tokens_left + (now - last) * self.capacity / self.refill_time)
        if tokens_left >= tokens:
            tokens_left -= tokens
            allowed = True
        else:
            allowed = False
        self._buckets[user_id] = (tokens_left, now)
        return allowed

    def clear(self, user_id: str) -> None:
        """Reset the bucket for ``user_id``."""
        self._buckets.pop(user_id, None)
