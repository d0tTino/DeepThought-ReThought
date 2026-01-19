"""High level helpers for recording and querying social interactions."""

from __future__ import annotations

import logging
from typing import Optional

from .prism_adapter import PrismEvent

try:  # Optional dependency
    from textblob import TextBlob  # type: ignore
except Exception:  # pragma: no cover - dependency missing
    TextBlob = None  # type: ignore

from .db_manager import DBManager

logger = logging.getLogger(__name__)

FRIEND_THRESHOLD = 0.5
RIVAL_THRESHOLD = -0.5
MIN_INTERACTIONS = 3

POSITIVE_EMOJIS = {"❤️", "👍"}
NEGATIVE_EMOJIS = {"💔", "👎"}
LATENCY_THRESHOLD = 5.0


class SocialGraphMemory:
    """Record messages and sentiment using a :class:`DBManager` backend."""

    def __init__(self, db_manager: Optional[DBManager] = None) -> None:
        self._db = db_manager or DBManager()

    async def record_message(self, source: str, text: str, target: Optional[str] = None) -> None:
        """Analyze sentiment of ``text`` and store the interaction."""
        try:
            score = float(TextBlob(text).sentiment.polarity) if TextBlob else 0.0
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

    async def set_personality(self, user_id: str, traits) -> None:
        await self._db.set_user_profile(user_id, traits)

    async def get_personality(self, user_id: str):
        return await self._db.get_user_profile(user_id)

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
        if status == "friend":
            await self._db.update_edge(user_a, user_b, "ally", 1.0)
        elif status == "rival":
            await self._db.update_edge(user_a, user_b, "rival", 1.0)

    async def update_relationship_type(self, user_a: str, user_b: str) -> None:
        """Public wrapper for updating relationship type between two users."""
        await self._update_relationship_type(user_a, user_b)

    async def get_relationship_status(self, user_a: str, user_b: str) -> str | None:
        return await self._db.get_relationship_type(user_a, user_b)

    async def update_edge(
        self, source: str, target: str, edge_type: str, weight: float = 1.0
    ) -> None:
        """Add or update a typed edge between ``source`` and ``target``."""
        await self._db.update_edge(source, target, edge_type, weight)

    async def get_edge_weight(self, source: str, target: str, edge_type: str) -> float:
        """Return the current weight of a typed edge."""
        return await self._db.get_edge_weight(source, target, edge_type)

    async def discover_factions(self, edge_type: str = "ally") -> list[list[str]]:
        """Cluster users into factions based on their positive edges.

        Uses ``networkx``'s greedy modularity communities algorithm to find
        clusters. Only edges with positive weight are considered.
        """
        try:  # pragma: no cover - networkx may be missing
            import importlib

            nx = importlib.import_module("networkx")
        except Exception as exc:  # pragma: no cover - defensive
            raise ImportError("networkx is required for faction discovery") from exc

        edges = await self._db.get_edges(edge_type=edge_type)
        graph = nx.Graph()
        for src, tgt, weight in edges:
            if weight > 0:
                graph.add_edge(src, tgt, weight=weight)
        if graph.number_of_nodes() == 0:
            return []
        communities = nx.algorithms.community.greedy_modularity_communities(
            graph, weight="weight"
        )
        return [sorted(list(c)) for c in communities]

    async def ingest_prism_event(self, event: PrismEvent) -> None:
        """Update the social graph using a preprocessed Prism event."""

        await self._db.log_interaction(
            event.source, event.target, sentiment_score=event.sentiment
        )
        if event.target is not None:
            await self._update_relationship_type(event.source, event.target)

        if event.reply_latency is not None:
            delta = 1 if event.reply_latency <= LATENCY_THRESHOLD else -1
            await self._db.adjust_affinity(event.source, delta)

        if event.target is not None and event.emoji_counts:
            for emoji, count in event.emoji_counts.items():
                if emoji in POSITIVE_EMOJIS:
                    await self.update_edge(event.source, event.target, "ally", float(count))
                elif emoji in NEGATIVE_EMOJIS:
                    await self.update_edge(event.source, event.target, "rival", float(count))

    async def close(self) -> None:
        await self._db.close()
