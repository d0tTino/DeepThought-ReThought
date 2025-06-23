from __future__ import annotations

from .connector import GraphConnector


class GraphDAL:
    """Data access layer providing high level graph operations."""

    def __init__(self, connector: GraphConnector) -> None:
        self._connector = connector

    def merge_entity(self, name: str) -> None:
        """Ensure an Entity node exists with the given ``name``."""
        self._connector.execute("MERGE (:Entity {name: $name})", {"name": name})

    def merge_next_edge(self, src: str, dst: str) -> None:
        """Ensure a NEXT edge exists from ``src`` to ``dst``."""
        self._connector.execute(
            "MATCH (a:Entity {name: $src}), (b:Entity {name: $dst}) MERGE (a)-[:NEXT]->(b)",
            {"src": src, "dst": dst},
        )

    # --- Methods used in tests ---
    def add_entity(self, label: str, props: dict) -> None:
        """Create or update a node with ``label`` and ``props``."""
        query = f"MERGE (n:{label} $props)"
        self._connector.execute(query, {"props": props})

    def add_relationship(self, start_id: int, end_id: int, rel_type: str, props: dict | None = None) -> None:
        """Create or update a relationship of type ``rel_type`` between nodes."""
        props = props or {}
        query = f"MATCH (a {{id: $start_id}}), (b {{id: $end_id}}) MERGE (a)-[r:{rel_type} $props]->(b)"
        self._connector.execute(query, {"start_id": start_id, "end_id": end_id, "props": props})

    def get_entity(self, label: str, key: str, value: str):
        """Return a single node matching ``label`` and ``key``."""
        query = f"MATCH (n:{label} {{{key}: $value}}) RETURN n"
        rows = self._connector.execute(query, {"value": value})
        return rows[0] if rows else None

    def query_subgraph(self, query: str, params: dict | None = None):
        """Run an arbitrary query and return the results."""
        return self._connector.execute(query, params or {})
