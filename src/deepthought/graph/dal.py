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
