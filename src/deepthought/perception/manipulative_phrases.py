"""Phrase lists for detecting manipulative language."""

from typing import Dict, Iterable

GUILT_TRIP_PHRASES = [
    "after all i've done for you",
    "you owe me",
    "i thought you cared",
    "if you really loved me",
    "how could you do this",
    "don't you care",
    "think of everything i've sacrificed",
]

THREAT_PHRASES = [
    "or else",
    "you'll regret",
    "i'll make you",
    "i will hurt",
    "i will harm",
    "i'm going to report",
    "you will be sorry",
    "i'll ruin you",
]

FLATTERY_PHRASES = [
    "you're the best",
    "no one is as",
    "you're amazing",
    "you're incredible",
    "you're perfect",
    "trust me",
    "you're so talented",
    "i admire you so much",
]

DEFLECTION_PHRASES = [
    "you're overreacting",
    "you're taking this too seriously",
    "let's not dwell on",
    "that's not important",
]

GASLIGHTING_PHRASES = [
    "that never happened",
    "you're imagining things",
    "you're making things up",
    "i never said that",
]

CATEGORY_PHRASES: Dict[str, Iterable[str]] = {
    "guilt_tripping": GUILT_TRIP_PHRASES,
    "threat": THREAT_PHRASES,
    "excessive_flattery": FLATTERY_PHRASES,
    "deflection": DEFLECTION_PHRASES,
    "gaslighting": GASLIGHTING_PHRASES,
}
