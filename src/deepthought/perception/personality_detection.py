from __future__ import annotations

"""Infer Big Five personality traits from chat messages.

This module uses simple keyword heuristics to approximate the Big Five
personality traits: openness, conscientiousness, extraversion, agreeableness
and neuroticism. If no indicative keywords are present, a neutral score of
``0.5`` is returned for each trait.
"""

from collections import Counter
from typing import Dict, Iterable

# Basic keywords loosely associated with each trait.
TRAIT_KEYWORDS = {
    "openness": {
        "curious",
        "imaginative",
        "creative",
        "adventurous",
        "inventive",
        "artistic",
    },
    "conscientiousness": {
        "organized",
        "prepared",
        "responsible",
        "efficient",
        "plan",
        "orderly",
    },
    "extraversion": {
        "outgoing",
        "energetic",
        "talkative",
        "social",
        "assertive",
        "party",
    },
    "agreeableness": {
        "kind",
        "trusting",
        "generous",
        "cooperative",
        "helpful",
        "warm",
    },
    "neuroticism": {
        "anxious",
        "moody",
        "nervous",
        "worry",
        "insecure",
        "fearful",
    },
}

NEUTRAL_SCORE = 0.5


def _heuristic_scores(text: str) -> Dict[str, float]:
    """Return normalized trait scores using keyword counts."""
    counts = Counter({trait: 0 for trait in TRAIT_KEYWORDS})
    for trait, words in TRAIT_KEYWORDS.items():
        counts[trait] = sum(1 for word in words if word in text)
    total = sum(counts.values())
    if total == 0:
        return {trait: NEUTRAL_SCORE for trait in TRAIT_KEYWORDS}
    return {trait: counts[trait] / total for trait in TRAIT_KEYWORDS}


def infer_personality(messages: Iterable[str]) -> Dict[str, float]:
    """Infer Big Five trait scores from a sequence of ``messages``."""
    if not messages:
        return {trait: NEUTRAL_SCORE for trait in TRAIT_KEYWORDS}
    text = " ".join(messages).lower()
    return _heuristic_scores(text)
