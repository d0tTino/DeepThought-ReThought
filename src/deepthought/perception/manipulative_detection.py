from __future__ import annotations

"""Heuristic detection of manipulative language in text."""

from typing import Optional

# Simple lists of phrases for each manipulation tactic
GUILT_TRIP_PHRASES = [
    "after all i've done for you",
    "you owe me",
    "i thought you cared",
    "if you really loved me",
    "how could you do this",
]

THREAT_PHRASES = [
    "or else",
    "you'll regret",
    "i'll make you",
    "i will hurt",
    "i will harm",
    "i'm going to report",
]

FLATTERY_PHRASES = [
    "you're the best",
    "no one is as",
    "you're amazing",
    "you're incredible",
    "you're perfect",
]


def detect_manipulation(text: str) -> Optional[str]:
    """Return the manipulation category if any heuristic matches."""
    lower = text.lower()
    for phrase in GUILT_TRIP_PHRASES:
        if phrase in lower:
            return "guilt_tripping"
    for phrase in THREAT_PHRASES:
        if phrase in lower:
            return "threat"
    for phrase in FLATTERY_PHRASES:
        if phrase in lower:
            return "excessive_flattery"
    return None
