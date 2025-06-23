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

    # ------------------------------------------------------------------
    # Additional helper methods used by tests
    # ------------------------------------------------------------------

    def add_entity(self, label: str, props: dict) -> None:
        """Create or merge an entity node with arbitrary properties."""
        self._connector.execute(f"MERGE (n:{label} $props)", {"props": props})

    def add_relationship(self, start_id: int, end_id: int, rel_type: str, props: dict) -> None:
        """Create or merge a relationship between two nodes."""
        self._connector.execute(
            f"MATCH (a {{id: $start_id}}), (b {{id: $end_id}}) MERGE (a)-[r:{rel_type} $props]->(b)",
            {"start_id": start_id, "end_id": end_id, "props": props},
        )

    def get_entity(self, label: str, prop: str, value: str) -> dict | None:
        """Retrieve the first entity matching ``prop``."""
        result = self._connector.execute(
            f"MATCH (n:{label} {{{prop}: $value}}) RETURN n",
            {"value": value},
        )
        return result[0] if result else None

    def query_subgraph(self, query: str, params: dict) -> list:
        """Run an arbitrary Cypher query and return the results."""
        return self._connector.execute(query, params)
