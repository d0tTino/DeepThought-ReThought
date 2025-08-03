import logging
import time
from typing import Optional, Tuple

from .file_graph_dal import FileGraphDAL

logger = logging.getLogger(__name__)


class UserGraphDAL(FileGraphDAL):
    """File-backed graph tracking interactions between users.

    The graph stores additional parameters ``weight_decay`` and
    ``sentiment_decay`` which are applied exponentially based on the time
    elapsed between interactions. These parameters are persisted in the graph
    metadata so that decay behaviour is reproducible across runs.
    """

    def __init__(
        self,
        graph_file: str = "user_graph.json",
        weight_decay: float = 1.0,
        sentiment_decay: float = 1.0,
    ) -> None:
        self._weight_decay = float(weight_decay)
        self._sentiment_decay = float(sentiment_decay)
        super().__init__(graph_file)
        # Persist decay parameters for reproducibility
        g = self._graph.graph
        if "weight_decay" in g:
            self._weight_decay = float(g["weight_decay"])
        else:
            g["weight_decay"] = self._weight_decay
        if "sentiment_decay" in g:
            self._sentiment_decay = float(g["sentiment_decay"])
        else:
            g["sentiment_decay"] = self._sentiment_decay
        self._write_graph()

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
        now = time.time()
        last = float(data.get("last_interaction", now))
        elapsed = max(0.0, now - last)
        # Apply decay to existing values before updating
        sentiment_sum = float(data.get("sentiment_sum", 0.0)) * (
            self._sentiment_decay ** elapsed
        ) + score
        weight = float(data.get("interaction_weight", 0.0)) * (
            self._weight_decay ** elapsed
        ) + 1.0
        count = int(data.get("interaction_count", 0)) + 1
        self._graph.add_edge(
            source,
            target,
            interaction_count=count,
            sentiment_sum=sentiment_sum,
            interaction_weight=weight,
            last_interaction=now,
        )

    def get_affinity(self, user_id: str) -> int:
        """Return how many messages ``user_id`` has sent."""
        return int(self._graph.nodes.get(user_id, {}).get("affinity", 0))

    def get_relationship(self, source: str, target: str) -> Tuple[int, float, float, float]:
        """Return ``(interaction_count, sentiment_sum, weight, last_interaction)`` for the edge."""
        data = self._graph.get_edge_data(source, target)
        if not data:
            return 0, 0.0, 0.0, 0.0
        now = time.time()
        last = float(data.get("last_interaction", now))
        elapsed = max(0.0, now - last)
        sentiment_sum = float(data.get("sentiment_sum", 0.0)) * (
            self._sentiment_decay ** elapsed
        )
        weight = float(data.get("interaction_weight", 0.0)) * (
            self._weight_decay ** elapsed
        )
        return (
            int(data.get("interaction_count", 0)),
            sentiment_sum,
            weight,
            last,
        )

    def _avg_sentiment(self, source: str, target: str) -> float:
        rel = self.get_relationship(source, target)
        count, ssum = rel[0], rel[1]
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

    def get_mutual_affinity(self, user_a: str, user_b: str) -> float:
        """Return decayed interaction weight between the pair."""
        ab_weight = self.get_relationship(user_a, user_b)[2]
        ba_weight = self.get_relationship(user_b, user_a)[2]
        return (ab_weight + ba_weight) / 2

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
                "interaction_weight": ab[2],
                "last_interaction": ab[3],
            },
            "b_to_a": {
                "count": ba[0],
                "sentiment_sum": ba[1],
                "avg_sentiment": self._avg_sentiment(user_b, user_a),
                "interaction_weight": ba[2],
                "last_interaction": ba[3],
            },
            # average weights because each message updates both directions
            "mutual_affinity": (ab[2] + ba[2]) / 2,
        }

    def get_interaction_weight(self, source: str, target: str) -> float:
        return self.get_relationship(source, target)[2]

    def get_last_interaction(self, source: str, target: str) -> float:
        return self.get_relationship(source, target)[3]
