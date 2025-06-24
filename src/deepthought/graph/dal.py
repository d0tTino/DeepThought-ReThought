from __future__ import annotations

import re

from .connector import GraphConnector


class GraphDAL:
    """Data access layer providing high level graph operations.

    This class exposes convenience helpers for inserting entities and
    relationships as well as running small read queries against the
    underlying graph database via :class:`GraphConnector`.
    """

    def __init__(self, connector: GraphConnector) -> None:
        self._connector = connector
        self._safe_label_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def _validate_label(self, label: str) -> str:
        """Return ``label`` if it matches :attr:`_safe_label_re`, else raise."""
        if not self._safe_label_re.fullmatch(label):
            raise ValueError(f"Invalid label: {label}")
        return label

    def _validate_rel_type(self, rel_type: str) -> str:
        """Return ``rel_type`` if it matches :attr:`_safe_label_re`, else raise."""
        if not self._safe_label_re.fullmatch(rel_type):
            raise ValueError(f"Invalid relationship type: {rel_type}")
        return rel_type

    _IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    @classmethod
    def _validate_identifier(cls, value: str) -> None:
        if not cls._IDENT_RE.match(value):
            raise ValueError(f"Invalid identifier: {value!r}")

    def merge_entity(self, name: str) -> None:
        """Ensure an Entity node exists with the given ``name``."""
        self._connector.execute("MERGE (:Entity {name: $name})", {"name": name})

    def merge_next_edge(self, src: str, dst: str) -> None:
        """Ensure a NEXT edge exists from ``src`` to ``dst``."""
        self._connector.execute(
            "MATCH (a:Entity {name: $src}), (b:Entity {name: $dst}) MERGE (a)-[:NEXT]->(b)",
            {"src": src, "dst": dst},
        )

    # New methods expected by unit tests

    def add_entity(self, label: str, props: dict) -> None:
        """Create or merge a node with ``label`` and ``props``."""
        self._validate_identifier(label)
        query = f"MERGE (n:{label} $props)"

        self._connector.execute(query, {"props": props})

    def add_relationship(self, start_id: int, end_id: int, rel_type: str, props: dict) -> None:
        """Create or merge a relationship of ``rel_type`` between two nodes."""
        self._validate_identifier(rel_type)
        query = "MATCH (a {id: $start_id}), (b {id: $end_id}) MERGE (a)-[r:" f"{rel_type} $props]->(b)"

        self._connector.execute(
            query,
            {"start_id": start_id, "end_id": end_id, "props": props},
        )

    def get_entity(self, label: str, field: str, value):
        """Return the first node of ``label`` where ``field`` equals ``value``."""
        query = f"MATCH (n:{label} {{{field}: $value}}) RETURN n"
        rows = self._connector.execute(query, {"value": value})
        return rows[0] if rows else None

    def query_subgraph(self, query: str, params: dict):
        """Execute an arbitrary query and return the resulting rows."""
        return self._connector.execute(query, params)
