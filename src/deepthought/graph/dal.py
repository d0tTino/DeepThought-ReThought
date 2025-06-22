from __future__ import annotations

from typing import Any, Dict, List, Optional

from .connector import GraphConnector


class GraphDAL:
    """Simple data access layer using :class:`GraphConnector`."""

    def __init__(self, connector: GraphConnector) -> None:
        self._connector = connector

    def add_entity(self, label: str, properties: Dict[str, Any]) -> None:
        """Create or merge a node with the given ``label`` and ``properties``."""
        query = f"MERGE (n:{label} $props)"
        self._connector.execute(query, {"props": properties})

    def add_relationship(
        self,
        start_id: Any,
        end_id: Any,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create or merge a relationship between two nodes by ``id``."""
        query = (
            "MATCH (a {id: $start_id}), (b {id: $end_id}) "
            f"MERGE (a)-[r:{rel_type} $props]->(b)"
        )
        self._connector.execute(
            query,
            {"start_id": start_id, "end_id": end_id, "props": properties or {}},
        )

    def get_entity(self, label: str, key: str, value: Any) -> Optional[Dict[str, Any]]:
        """Return the first node matching ``label`` and property ``key``."""
        query = f"MATCH (n:{label} {{{key}: $value}}) RETURN n"
        result = self._connector.execute(query, {"value": value})
        return result[0] if result else None

    def query_subgraph(self, query: str, params: Optional[Dict[str, Any]] = None) -> List:
        """Execute an arbitrary Cypher ``query`` and return the results."""
        return self._connector.execute(query, params)
