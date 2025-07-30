"""Simple heuristic-based manipulation detector."""
from __future__ import annotations

from typing import Dict

MANIPULATIVE_PHRASES: Dict[str, float] = {
    "trust me": 0.8,
    "for your own good": 0.7,
    "if you": 0.5,
    "only": 0.3,
}


def manipulation_score(text: str, phrases: Dict[str, float] | None = None) -> float:
    """Return a manipulation score between 0 and 1 based on keyword matches."""
    if not isinstance(text, str):
        return 0.0
    lowered = text.lower()
    scores = phrases or MANIPULATIVE_PHRASES
    for phrase, score in scores.items():
        if phrase in lowered:
            return score
    return 0.0
