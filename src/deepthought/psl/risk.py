from __future__ import annotations

"""Commit risk scoring utilities."""

from pathlib import Path
from typing import Mapping

import yaml

from .model import PSLModel


class RiskScorer:
    """Score code changes using a simple PSL model."""

    def __init__(self, model: PSLModel, threshold: float = 0.5) -> None:
        self._model = model
        self._threshold = threshold

    @classmethod
    def from_file(cls, path: str | Path) -> "RiskScorer":
        """Load model configuration from ``path``."""
        data = yaml.safe_load(Path(path).read_text())
        model = PSLModel.from_config(data.get("model", {}))
        threshold = float(data.get("threshold", 0.5))
        return cls(model, threshold)

    def score(self, evidence: Mapping[str, float]) -> float:
        """Return a risk score based on ``evidence``."""
        return self._model.infer(evidence)

    def is_high_risk(self, evidence: Mapping[str, float]) -> bool:
        """Return ``True`` if the score meets or exceeds the threshold."""
        return self.score(evidence) >= self._threshold
