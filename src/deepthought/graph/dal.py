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

    # Methods required by unit tests

    def add_entity(self, label: str, props: dict) -> None:
        """Create or merge a node with ``label`` and ``props``."""
        self._connector.execute(f"MERGE (n:{label} $props)", {"props": props})

    def add_relationship(
        self, start_id: int, end_id: int, rel: str, props: dict | None = None
    ) -> None:
        """Create or merge a relationship of type ``rel`` between two IDs."""
        self._connector.execute(
            f"MATCH (a {{id: $start_id}}), (b {{id: $end_id}}) MERGE (a)-[r:{rel} $props]->(b)",
            {"start_id": start_id, "end_id": end_id, "props": props or {}},
        )

    def get_entity(self, label: str, key: str, value: str):
        """Return the first node of ``label`` where ``key`` equals ``value``."""
        rows = self._connector.execute(
            f"MATCH (n:{label} {{{key}: $value}}) RETURN n",
            {"value": value},
        )
        return rows[0] if rows else None

    def query_subgraph(self, query: str, params: dict | None = None):
        """Execute an arbitrary Cypher query and return the results."""
        return self._connector.execute(query, params or {})
