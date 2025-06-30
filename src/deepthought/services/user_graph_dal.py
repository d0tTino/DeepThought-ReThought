import json
import logging
import os
from typing import Optional, Tuple

import networkx as nx

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
            data = self._graph.get_edge_data(source, target, default={})
            count = data.get("interaction_count", 0) + 1
            sentiment_sum = data.get("sentiment_sum", 0.0) + float(
                sentiment_score or 0.0
            )
            self._graph.add_edge(
                source,
                target,
                interaction_count=count,
                sentiment_sum=sentiment_sum,
            )
        self._write_graph()

    def get_affinity(self, user_id: str) -> int:
        """Return how many messages ``user_id`` has sent."""
        return int(self._graph.nodes.get(user_id, {}).get("affinity", 0))

    def get_relationship(self, source: str, target: str) -> Tuple[int, float]:
        """Return ``(interaction_count, sentiment_sum)`` for the edge."""
        data = self._graph.get_edge_data(source, target)
        if not data:
            return 0, 0.0
        return int(data.get("interaction_count", 0)), float(
            data.get("sentiment_sum", 0.0)
        )

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
