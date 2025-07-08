from __future__ import annotations

"""Graph backend strategy interfaces."""

import uuid
from abc import ABC, abstractmethod
from typing import Any, List

from .connector import GraphConnector, Neo4jConnector
from .dal import GraphDAL


class GraphBackend(ABC):
    """Abstract interface used by :class:`TieredMemory` for graph operations."""

    @abstractmethod
    def query_subgraph(self, query: str, params: dict) -> List[Any]:
        """Execute ``query`` and return rows."""

    @abstractmethod
    def merge_entity(self, name: str) -> None:
        """Ensure an entity with ``name`` exists."""


class GraphDALBackend(GraphBackend):
    """Adapter delegating to :class:`GraphDAL`."""

    def __init__(self, dal: GraphDAL) -> None:
        self._dal = dal

    def query_subgraph(self, query: str, params: dict) -> List[Any]:
        return self._dal.query_subgraph(query, params)

    def merge_entity(self, name: str) -> None:
        self._dal.merge_entity(name)


class Neo4jBackend(GraphBackend):
    """Direct backend using :class:`Neo4jConnector`."""

    def __init__(self, connector: Neo4jConnector) -> None:
        self._connector = connector

    def query_subgraph(self, query: str, params: dict) -> List[Any]:
        return self._connector.execute(query, params)

    def merge_entity(self, name: str) -> None:
        entity_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name))
        self._connector.execute(
            "MERGE (:Entity {id: $id, name: $name})",
            {"id": entity_id, "name": name},
        )


class NoOpGraphBackend(GraphBackend):
    """Graph backend that does nothing."""

    def query_subgraph(self, query: str, params: dict) -> List[Any]:
        return []

    def merge_entity(self, name: str) -> None:  # pragma: no cover - trivial
        pass


def create_graph_backend(backend: str = "memgraph", **params: Any) -> GraphBackend:
    """Return a :class:`GraphBackend` implementation for ``backend``."""
    lower = backend.lower()
    if lower == "memgraph":
        connector = GraphConnector(**params)
        dal = GraphDAL(connector)
        return GraphDALBackend(dal)
    if lower == "neo4j":
        connector = Neo4jConnector(**params)
        return Neo4jBackend(connector)
    if lower in {"none", "noop", "stub"}:
        return NoOpGraphBackend()
    raise ValueError(f"Unknown graph backend: {backend}")
