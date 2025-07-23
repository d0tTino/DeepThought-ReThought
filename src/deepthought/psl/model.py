from __future__ import annotations

"""Simplified PSL model wrapper."""

from typing import Mapping


class PSLModel:
    """Lightweight container for predicate weights."""

    def __init__(self, weights: Mapping[str, float]) -> None:
        self._weights = dict(weights)

    @classmethod
    def from_config(cls, config: Mapping[str, Mapping[str, float]]) -> "PSLModel":
        """Create a model from a configuration mapping."""
        weights = config.get("weights", {})
        return cls(weights)

    def infer(self, evidence: Mapping[str, float]) -> float:
        """Return the weighted sum score for ``evidence``."""
        return sum(
            self._weights.get(k, 0.0) * evidence.get(k, 0.0) for k in self._weights
        )
