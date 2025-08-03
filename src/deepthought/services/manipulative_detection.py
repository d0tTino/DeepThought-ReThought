"""Simple heuristic-based manipulation detector."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from deepthought.perception.manipulative_detection import detect_manipulation
from deepthought.perception.manipulative_phrases import (  # noqa: F401
    CATEGORY_PHRASES,
    DEFLECTION_PHRASES,
    FLATTERY_PHRASES,
    GASLIGHTING_PHRASES,
    GUILT_TRIP_PHRASES,
    THREAT_PHRASES,
)


def manipulation_score(text: str, phrases: Dict[str, Iterable[str]] | None = None) -> Optional[str]:
    """Return the detected manipulation category."""
    if not isinstance(text, str):
        return None

    if phrases is None:
        return detect_manipulation(text)

    lowered = text.lower()
    for category, plist in phrases.items():
        for phrase in plist:
            if phrase in lowered:
                return category

    return None
