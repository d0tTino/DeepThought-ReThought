import logging
from typing import Iterable

logger = logging.getLogger(__name__)

# Simple list of banned phrases that should not be processed further
BANNED_PHRASES = ["banned", "prohibited"]


def is_allowed(text: str, banned_phrases: Iterable[str] | None = None) -> bool:
    """Return ``True`` if ``text`` does not contain any banned phrases."""
    if not isinstance(text, str):
        return False
    phrases = list(banned_phrases) if banned_phrases is not None else BANNED_PHRASES
    lowered = text.lower()
    return not any(phrase in lowered for phrase in phrases)
