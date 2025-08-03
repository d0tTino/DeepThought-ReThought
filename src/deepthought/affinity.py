from __future__ import annotations

"""Track social affinity scores per persona."""

from typing import Dict, List, Tuple


class AffinityTracker:
    """Simple in-memory affinity tracker."""

    def __init__(self) -> None:
        self._scores: Dict[str, int] = {}
        self._interactions: Dict[Tuple[str, str], int] = {}
        self._mutual: Dict[Tuple[str, str], int] = {}

    def update(self, persona: str, delta: int = 1) -> None:
        """Increment ``persona`` affinity by ``delta``."""
        self._scores[persona] = self._scores.get(persona, 0) + delta

    def get(self, persona: str) -> int:
        """Return current affinity score for ``persona``."""
        return self._scores.get(persona, 0)

    @staticmethod
    def _pair_key(a: str, b: str) -> Tuple[str, str]:
        return tuple(sorted((a, b)))

    def update_interaction(self, a: str, b: str, delta: int = 1) -> None:
        """Increment interaction count between ``a`` and ``b``."""
        key = self._pair_key(a, b)
        self._interactions[key] = self._interactions.get(key, 0) + delta

    def interaction_count(self, a: str, b: str) -> int:
        """Return interaction count between ``a`` and ``b``."""
        key = self._pair_key(a, b)
        return self._interactions.get(key, 0)

    def update_mutual_affinity(self, a: str, b: str, delta: int = 1) -> None:
        """Increment mutual affinity between ``a`` and ``b`` by ``delta``."""
        key = self._pair_key(a, b)
        self._mutual[key] = self._mutual.get(key, 0) + delta

    def mutual_affinity(self, a: str, b: str) -> int:
        """Return mutual affinity between ``a`` and ``b``."""
        key = self._pair_key(a, b)
        return self._mutual.get(key, 0)

    def top_friends(self, user: str, n: int = 3) -> List[str]:
        """Return up to ``n`` users with highest mutual affinity with ``user``."""
        pairs = [
            (other, score)
            for (a, b), score in self._mutual.items()
            if user in (a, b)
            for other in ([b] if a == user else [a])
        ]
        return [other for other, _ in sorted(pairs, key=lambda x: x[1], reverse=True)[:n]]

    def top_rivals(self, user: str, n: int = 3) -> List[str]:
        """Return up to ``n`` users with lowest mutual affinity with ``user``."""
        pairs = [
            (other, score)
            for (a, b), score in self._mutual.items()
            if user in (a, b)
            for other in ([b] if a == user else [a])
        ]
        return [other for other, _ in sorted(pairs, key=lambda x: x[1])[:n]]
