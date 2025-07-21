"""Small demo storing graph facts and retrieving them."""

from __future__ import annotations

import logging
import os
import time

from deepthought.graph import GraphConnector, GraphDAL, Neo4jConnector
from deepthought.memory.hierarchical import HierarchicalMemory
from deepthought.memory.vector_store import create_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class _NoOpConnector:
    def execute(self, query: str, params: dict | None = None) -> list:
        return []


def _create_dal(backend: str) -> GraphDAL:
    backend = backend.lower()
    if backend == "neo4j":
        connector = Neo4jConnector()
    elif backend == "memgraph":
        connector = GraphConnector()
    elif backend in {"noop", "none"}:
        connector = _NoOpConnector()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return GraphDAL(connector)

def main() -> None:
    backend_name = os.getenv("DT_GRAPH_BACKEND", "memgraph")
    store = create_vector_store("faiss", collection_name="graph_demo")
    dal = _create_dal(backend_name)
    memory = HierarchicalMemory(store, dal)

    facts = [
        "Alice met Bob",
        "Bob chatted with Carol",
        "Carol saw Dave",
    ]

    store.add_texts(facts)
    for fact in facts:
        try:
            src, dst = fact.split()[0], fact.split()[-1]
            dal.merge_entity(src)
            dal.merge_entity(dst)
            dal.merge_next_edge(src, dst)
        except Exception:  # pragma: no cover - demo only
            logger.error("Failed to store %s", fact, exc_info=True)

    start = time.perf_counter()
    ctx = memory.retrieve_context("Alice")
    latency = time.perf_counter() - start
    logger.info("Retrieved context: %s", ctx)
    logger.info("Retrieval latency: %.3f seconds", latency)


if __name__ == "__main__":
    main()
