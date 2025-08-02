from __future__ import annotations

"""Lightweight emotion classifier with optional Text2Emotion support.

If the `text2emotion` package is available, it is used to compute
probabilities for five basic emotions. Otherwise simple keyword
heuristics provide normalized scores.
"""

from collections import Counter
from typing import Dict

try:  # pragma: no cover - optional dependency
    import text2emotion as _t2e
except Exception:  # pragma: no cover - optional dependency
    _t2e = None

HAPPY_WORDS = {
    "happy",
    "joy",
    "joyful",
    "glad",
    "excited",
    "delighted",
    "pleased",
    "thrilled",
}

SAD_WORDS = {
    "sad",
    "down",
    "depressed",
    "unhappy",
    "miserable",
    "gloomy",
    "blue",
}

ANGRY_WORDS = {
    "angry",
    "mad",
    "furious",
    "irritated",
    "annoyed",
    "rage",
}

FEAR_WORDS = {
    "scared",
    "afraid",
    "terrified",
    "fear",
    "worried",
}

SURPRISE_WORDS = {
    "surprised",
    "astonished",
    "amazed",
    "shocked",
}

EMOTION_KEYWORDS = {
    "Happy": HAPPY_WORDS,
    "Sad": SAD_WORDS,
    "Angry": ANGRY_WORDS,
    "Fear": FEAR_WORDS,
    "Surprise": SURPRISE_WORDS,
}


def _heuristic_scores(text: str) -> Dict[str, float]:
    """Return normalized emotion scores using keyword counts."""
    lower = text.lower()
    counts = Counter({emotion: 0 for emotion in EMOTION_KEYWORDS})
    for emotion, words in EMOTION_KEYWORDS.items():
        counts[emotion] = sum(1 for w in words if w in lower)
    total = sum(counts.values()) or 1
    return {emotion: count / total for emotion, count in counts.items()}


def detect_emotions(text: str) -> Dict[str, float]:
    """Return a mapping of emotion label to score for ``text``.

    Parameters
    ----------
    text:
        The input text to analyze.
    """
    if _t2e is not None:
        try:  # pragma: no cover - optional dependency
            return _t2e.get_emotion(text)
        except Exception:
            pass
    return _heuristic_scores(text)
