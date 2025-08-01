import logging
from typing import Optional, Tuple

from .file_graph_dal import FileGraphDAL

logger = logging.getLogger(__name__)


class UserGraphDAL(FileGraphDAL):
    """File-backed graph tracking interactions between users."""

    def __init__(self, graph_file: str = "user_graph.json") -> None:
        super().__init__(graph_file)

    def add_message(
        self,
        source: str,
        target: Optional[str] = None,
        sentiment_score: float | None = None,
    ) -> None:
        """Record a message from ``source`` to ``target`` with optional sentiment."""
        self._graph.add_node(source, affinity=self.get_affinity(source) + 1)
        if target is not None:
            self._graph.add_node(target, affinity=self.get_affinity(target))
            score = float(sentiment_score or 0.0)
            for s, t in ((source, target), (target, source)):
                self._update_edge(s, t, score)
        self._write_graph()

    def _update_edge(self, source: str, target: str, score: float) -> None:
        data = self._graph.get_edge_data(source, target, default={})
        count = data.get("interaction_count", 0) + 1
        sentiment_sum = data.get("sentiment_sum", 0.0) + score
        self._graph.add_edge(
            source,
            target,
            interaction_count=count,
            sentiment_sum=sentiment_sum,
        )

    def get_affinity(self, user_id: str) -> int:
        """Return how many messages ``user_id`` has sent."""
        return int(self._graph.nodes.get(user_id, {}).get("affinity", 0))

    def get_relationship(self, source: str, target: str) -> Tuple[int, float]:
        """Return ``(interaction_count, sentiment_sum)`` for the edge."""
        data = self._graph.get_edge_data(source, target)
        if not data:
            return 0, 0.0
        return int(data.get("interaction_count", 0)), float(data.get("sentiment_sum", 0.0))

    def _avg_sentiment(self, source: str, target: str) -> float:
        count, ssum = self.get_relationship(source, target)
        if not count:
            return 0.0
        return ssum / count

    def get_friendliness(self, source: str, target: str) -> float:
        """Return the average positive sentiment from ``source`` to ``target``."""
        avg = self._avg_sentiment(source, target)
        return max(0.0, avg)

    def get_hostility(self, source: str, target: str) -> float:
        """Return the average negative sentiment from ``source`` to ``target``."""
        avg = self._avg_sentiment(source, target)
        return min(0.0, avg)

    def get_mutual_affinity(self, user_a: str, user_b: str) -> int:
        """Return how many messages have passed between the pair."""
        ab_count, _ = self.get_relationship(user_a, user_b)
        ba_count, _ = self.get_relationship(user_b, user_a)
        # each message updates both directional edges, so average the counts
        return int((ab_count + ba_count) / 2)

    def get_relationship_stats(self, user_a: str, user_b: str) -> dict:
        """Return a summary of interactions and sentiment between two users."""
        ab = self.get_relationship(user_a, user_b)
        ba = self.get_relationship(user_b, user_a)
        return {
            "pair": (user_a, user_b),
            "a_to_b": {
                "count": ab[0],
                "sentiment_sum": ab[1],
                "avg_sentiment": self._avg_sentiment(user_a, user_b),
            },
            "b_to_a": {
                "count": ba[0],
                "sentiment_sum": ba[1],
                "avg_sentiment": self._avg_sentiment(user_b, user_a),
            },
            # average counts because each interaction increments both directions
            "mutual_affinity": int((ab[0] + ba[0]) / 2),
        }
