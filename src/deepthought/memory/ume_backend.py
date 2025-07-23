from __future__ import annotations

"""Tiered memory backend that also logs events to UME."""

from dataclasses import dataclass
import time
from typing import List, Any, Optional

from ume import Event, apply_event_to_graph, MockGraph

from .tiered import TieredMemory
from .vector_store import create_vector_store, VectorStore
from ..graph.backend import GraphBackend, NoOpGraphBackend


class UMEGraphBackend(GraphBackend):
    """Minimal graph backend backed by a UME graph."""

    def __init__(self, graph: Optional[MockGraph] = None) -> None:
        self._graph = graph or MockGraph()
        self.events: list[Event] = []

    def query_subgraph(self, query: str, params: dict) -> List[Any]:  # pragma: no cover - basic demo
        # Support simple queries used by TieredMemory tests
        limit = params.get("limit")
        if query.startswith("MATCH (n:Entity)"):
            nodes = self._graph.get_all_node_ids()
            if limit:
                nodes = nodes[: int(limit)]
            return [{"fact": n} for n in nodes]
        if "MATCH (a:Entity {name: $src})-[:NEXT]->(b:Entity {name: $dst})" in query:
            src = params.get("src")
            dst = params.get("dst")
            for s, t, label in self._graph.get_all_edges():
                if s == src and t == dst and label == "NEXT":
                    return [{"src": s, "dst": t}]
            return []
        return []

    def merge_entity(self, name: str) -> None:  # pragma: no cover - trivial
        evt = Event(
            event_type="CREATE_NODE",
            timestamp=int(time.time()),
            payload={"node_id": name, "attributes": {"name": name}},
        )
        apply_event_to_graph(evt, self._graph)
        self.events.append(evt)


@dataclass
class UMEMemory(TieredMemory):
    """Tiered memory that logs interactions to a UME graph."""

    ume_backend: UMEGraphBackend

    def store_interaction(self, text: str) -> None:  # pragma: no cover - simple
        super().store_interaction(text)
        self.ume_backend.merge_entity(text)


def create_ume_memory(settings=None) -> UMEMemory:
    """Return :class:`UMEMemory` configured from ``Settings``."""
    from ..config import get_settings
    from ..memory import load_memory_settings

    settings = settings or get_settings()
    ms = load_memory_settings(settings)
    store = create_vector_store(
        backend=ms.vector_backend,
        persist_directory=None,
        use_gpu=ms.vector_use_gpu,
    )
    backend = UMEGraphBackend()
    return UMEMemory(
        store,
        backend,
        capacity=ms.memory_capacity,
        top_k=ms.memory_top_k,
        ume_backend=backend,
    )

