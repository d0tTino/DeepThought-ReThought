from __future__ import annotations

"""Graph backend strategy interfaces."""

from abc import ABC, abstractmethod
from typing import Any, List
import os

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


class NoOpGraphBackend(GraphBackend):
    """Graph backend that does nothing."""

    def query_subgraph(self, query: str, params: dict) -> List[Any]:
        return []

    def merge_entity(self, name: str) -> None:  # pragma: no cover - trivial
        pass


def create_graph_backend(name: str = "memgraph", **params: Any) -> GraphBackend:
    """Return a :class:`GraphBackend` implementation based on ``name``."""
    lower = name.lower()
    if lower == "memgraph":
        host = params.get("host", os.getenv("MG_HOST", "localhost"))
        port = int(params.get("port", os.getenv("MG_PORT", 7687)))
        username = params.get("username", os.getenv("MG_USER", ""))
        password = params.get("password", os.getenv("MG_PASSWORD", ""))
        connector = GraphConnector(
            host=host, port=port, username=username, password=password
        )
        dal = GraphDAL(connector)
        return GraphDALBackend(dal)
    if lower == "neo4j":
        host = params.get("host", os.getenv("NEO4J_HOST", "localhost"))
        port = int(params.get("port", os.getenv("NEO4J_PORT", 7687)))
        username = params.get("username", os.getenv("NEO4J_USER", "neo4j"))
        password = params.get("password", os.getenv("NEO4J_PASSWORD", "neo4j"))
        connector = Neo4jConnector(
            host=host, port=port, username=username, password=password
        )
        dal = GraphDAL(connector)
        return GraphDALBackend(dal)
    if lower in {"none", "noop", "stub"}:
        return NoOpGraphBackend()
    raise ValueError(f"Unknown graph backend: {name}")

