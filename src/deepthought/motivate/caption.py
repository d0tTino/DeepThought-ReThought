"""Very small abstractive caption generator stub."""

from typing import Iterable


def summarise_message(message: str, max_words: int = 5) -> str:
    """Return the first ``max_words`` words of ``message`` as a caption."""
    words: Iterable[str] = message.split()
    return " ".join(list(words)[:max_words])
