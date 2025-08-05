"""High level helpers for recording and querying social interactions."""

from __future__ import annotations

import logging
from typing import Optional

from textblob import TextBlob

from .db_manager import DBManager

logger = logging.getLogger(__name__)

FRIEND_THRESHOLD = 0.5
RIVAL_THRESHOLD = -0.5
MIN_INTERACTIONS = 3


class SocialGraphMemory:
    """Record messages and sentiment using a :class:`DBManager` backend."""

    def __init__(self, db_manager: Optional[DBManager] = None) -> None:
        self._db = db_manager or DBManager()

    async def record_message(self, source: str, text: str, target: Optional[str] = None) -> None:
        """Analyze sentiment of ``text`` and store the interaction."""
        try:
            score = float(TextBlob(text).sentiment.polarity)
        except Exception:  # pragma: no cover - TextBlob failure
            logger.exception("Sentiment analysis failed")
            score = 0.0
        await self._db.log_interaction(source, target, sentiment_score=score)
        if target is not None:
            await self._update_relationship_type(source, target)

    async def get_affinity(self, user_id: str) -> int:
        return await self._db.get_affinity(user_id)

    async def get_friendliness(self, source: str, target: str) -> float:
        return await self._db.get_friendliness(source, target)

    async def get_hostility(self, source: str, target: str) -> float:
        return await self._db.get_hostility(source, target)

    async def get_mutual_affinity(self, user_a: str, user_b: str) -> float:
        return await self._db.get_pair_mutual_affinity(user_a, user_b)

    async def get_relationship_stats(self, user_a: str, user_b: str) -> dict:
        """Return a summary of interactions and sentiment between two users."""
        ab = await self._db.get_relationship(user_a, user_b)
        ba = await self._db.get_relationship(user_b, user_a)

        def _stats(row: tuple | None) -> dict:
            if not row:
                return {
                    "count": 0,
                    "sentiment_sum": 0.0,
                    "avg_sentiment": 0.0,
                    "interaction_weight": 0.0,
                    "last_interaction": None,
                }
            count, sentiment_sum, weight, last = row
            avg = float(sentiment_sum) / count if count else 0.0
            return {
                "count": int(count),
                "sentiment_sum": float(sentiment_sum),
                "avg_sentiment": avg,
                "interaction_weight": float(weight),
                "last_interaction": last,
            }

        stats_a = _stats(ab)
        stats_b = _stats(ba)
        mutual = await self._db.get_pair_mutual_affinity(user_a, user_b)
        return {
            "pair": (user_a, user_b),
            "a_to_b": stats_a,
            "b_to_a": stats_b,
            "mutual_affinity": int(mutual),
        }

    async def _update_relationship_type(self, user_a: str, user_b: str) -> None:
        ab = await self._db.get_relationship(user_a, user_b) or (0, 0.0, 0.0, None)
        ba = await self._db.get_relationship(user_b, user_a) or (0, 0.0, 0.0, None)
        total_count = int(ab[0] or 0) + int(ba[0] or 0)
        total_sentiment = float(ab[1] or 0.0) + float(ba[1] or 0.0)
        status = "neutral"
        if total_count >= MIN_INTERACTIONS and total_count:
            avg = total_sentiment / total_count
            if avg >= FRIEND_THRESHOLD:
                status = "friend"
            elif avg <= RIVAL_THRESHOLD:
                status = "rival"
        await self._db.set_relationship_type(user_a, user_b, status)

    async def get_relationship_status(self, user_a: str, user_b: str) -> str | None:
        return await self._db.get_relationship_type(user_a, user_b)

    async def close(self) -> None:
        await self._db.close()

