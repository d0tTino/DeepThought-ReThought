from __future__ import annotations

"""Track social affinity scores per persona."""

from typing import Dict


class AffinityTracker:
    """Simple in-memory affinity tracker."""

    def __init__(self) -> None:
        self._scores: Dict[str, int] = {}

    def update(self, persona: str, delta: int = 1) -> None:
        """Increment ``persona`` affinity by ``delta``."""
        self._scores[persona] = self._scores.get(persona, 0) + delta

    def get(self, persona: str) -> int:
        """Return current affinity score for ``persona``."""
        return self._scores.get(persona, 0)
