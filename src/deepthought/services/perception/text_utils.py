"""Utilities for handling text tokens used by perception components."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

Token = Tuple[str, float, float]

# Regular expressions matching common personally identifiable information (PII).
PII_PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
]


def scrub_text(text: str) -> str:
    """Redact common PII patterns from ``text``."""

    for pattern in PII_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def scrub_tokens(tokens: Iterable[Sequence[object]]) -> List[Token]:
    """Return tokens with their textual component scrubbed for PII."""

    sanitized: List[Token] = []
    for raw in tokens:
        if len(raw) != 3:
            continue
        text, start, end = raw
        sanitized.append((scrub_text(str(text)), float(start), float(end)))
    return sanitized


def hop_aligned_tokens(text: str, hop_seconds: float) -> List[Token]:
    """Tokenize ``text`` into hop-aligned spans using ``hop_seconds``."""

    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive")

    scrubbed = scrub_text(text)
    if not scrubbed:
        return []

    words = scrubbed.split()
    tokens: List[Token] = []
    for index, word in enumerate(words):
        start = index * hop_seconds
        end = (index + 1) * hop_seconds
        tokens.append((word, start, end))
    return tokens
