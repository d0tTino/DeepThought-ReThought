from __future__ import annotations

"""Simple persona selection utilities."""

from typing import Dict, Sequence


class PersonaSelector:
    """Select a persona based on keyword matches."""

    def __init__(self, personas: Dict[str, Sequence[str]]) -> None:
        self._personas = {k: list(v) for k, v in personas.items()}

    def choose(self, text: str) -> str | None:
        """Return the persona with the most keyword matches."""
        text_low = text.lower()
        best = None
        best_count = -1
        for name, keywords in self._personas.items():
            count = sum(text_low.count(kw.lower()) for kw in keywords)
            if count > best_count:
                best = name
                best_count = count
        return best
