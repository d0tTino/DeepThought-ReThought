from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MemoryTier(StrEnum):
    EPHEMERAL = "ephemeral"
    WORKING = "working"
    LONG_TERM = "long_term"


@dataclass(frozen=True)
class ScoredMemoryEvent:
    text: str
    salience: float
    tier: MemoryTier
    reason_tags: tuple[str, ...]


class MemoryLifecyclePolicy:
    """Score memory events by conversational salience and assign a tier."""

    _PREFERENCE_MARKERS = ("prefer", "favorite", "like", "love", "dislike", "hate")
    _IDENTITY_MARKERS = ("i am", "i'm", "my name is", "call me", "i work", "i live")
    _COMMITMENT_MARKERS = ("i will", "i'll", "i promise", "i commit", "remind me")
    _UNRESOLVED_TASK_MARKERS = (
        "todo",
        "to-do",
        "need to",
        "pending",
        "follow up",
        "later",
    )

    def score_event(self, text: str) -> ScoredMemoryEvent:
        raw = (text or "").strip()
        lowered = raw.lower()
        score = 0.05
        reasons: list[str] = []

        if any(marker in lowered for marker in self._PREFERENCE_MARKERS):
            score += 0.35
            reasons.append("user_preference")
        if any(marker in lowered for marker in self._IDENTITY_MARKERS):
            score += 0.35
            reasons.append("identity_fact")
        if any(marker in lowered for marker in self._COMMITMENT_MARKERS):
            score += 0.30
            reasons.append("commitment")
        if any(marker in lowered for marker in self._UNRESOLVED_TASK_MARKERS):
            score += 0.30
            reasons.append("unresolved_task")

        tier = self._tier_from_score(score)
        return ScoredMemoryEvent(
            text=raw, salience=min(score, 1.0), tier=tier, reason_tags=tuple(reasons)
        )

    @staticmethod
    def _tier_from_score(score: float) -> MemoryTier:
        if score >= 0.65:
            return MemoryTier.LONG_TERM
        if score >= 0.30:
            return MemoryTier.WORKING
        return MemoryTier.EPHEMERAL
