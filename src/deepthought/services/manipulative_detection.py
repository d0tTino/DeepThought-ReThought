"""Simple heuristic-based manipulation detector."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from deepthought.perception.manipulative_detection import detect_manipulation

# Phrase lists grouped by manipulation category
GUILT_TRIP_PHRASES: Iterable[str] = [
    "after all i've done for you",
    "you owe me",
    "i thought you cared",
    "if you really loved me",
    "how could you do this",
    "don't you care",
    "think of everything i've sacrificed",
]

THREAT_PHRASES: Iterable[str] = [
    "or else",
    "you'll regret",
    "i'll make you",
    "i will hurt",
    "i will harm",
    "i'm going to report",
    "you will be sorry",
    "i'll ruin you",
]

FLATTERY_PHRASES: Iterable[str] = [
    "you're the best",
    "no one is as",
    "you're amazing",
    "you're incredible",
    "you're perfect",
    "trust me",
    "you're so talented",
    "i admire you so much",
]

# Mapping of category names to their phrases
CATEGORY_PHRASES: Dict[str, Iterable[str]] = {
    "guilt_tripping": GUILT_TRIP_PHRASES,
    "threat": THREAT_PHRASES,
    "excessive_flattery": FLATTERY_PHRASES,
}


def manipulation_score(
    text: str, phrases: Dict[str, Iterable[str]] | None = None
) -> Optional[str]:
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
