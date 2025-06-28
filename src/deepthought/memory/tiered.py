from __future__ import annotations

"""Tiered memory mixing short-term vectors with long-term graph storage."""

import logging
from collections import OrderedDict
from typing import List, Sequence

from ..graph import GraphDAL
from .vector_store import VectorStore, create_vector_store

logger = logging.getLogger(__name__)


class TieredMemory:
    """Short term vector memory backed by long term graph storage."""

    def __init__(
        self,
        vector_store: VectorStore,
        graph_dal: GraphDAL,
        capacity: int = 100,
        top_k: int = 3,
    ) -> None:
        self._store = vector_store
        self._dal = graph_dal
        self._capacity = capacity
        self._top_k = top_k
        self._counter = 0
        self._lru: OrderedDict[str, str] = OrderedDict()

    @classmethod
    def from_chroma(
        cls,
        graph_dal: GraphDAL,
        collection_name: str = "deepthought",
        persist_directory: str | None = None,
        capacity: int = 100,
        top_k: int = 3,
    ) -> "TieredMemory":
        store = create_vector_store(collection_name, persist_directory)
        return cls(store, graph_dal, capacity=capacity, top_k=top_k)

    # internal helpers
    def _evict_if_needed(self) -> None:
        while len(self._lru) > self._capacity:
            text, doc_id = self._lru.popitem(last=False)
            try:
                self._store.collection.delete([doc_id])
            except Exception:  # pragma: no cover - defensive
                logger.error(
                    "Failed to delete %s from vector store", doc_id, exc_info=True
                )

    def _add_to_vector(self, text: str) -> None:
        if text in self._lru:
            self._lru.move_to_end(text)
            return
        doc_id = str(self._counter)
        self._counter += 1
        try:
            self._store.add_texts([text], ids=[doc_id])
            self._lru[text] = doc_id
            self._evict_if_needed()
        except Exception:  # pragma: no cover - defensive
            logger.error("Failed to add text to vector store", exc_info=True)

    def _vector_matches(self, prompt: str) -> List[str]:
        try:
            result = self._store.query([prompt], n_results=self._top_k)
            docs: Sequence | None = None
            if isinstance(result, dict):
                docs = result.get("documents")
            elif isinstance(result, Sequence):
                docs = result
            if not docs:
                return []
            matches: List[str] = []
            for doc in docs:
                if isinstance(doc, list):
                    for d in doc:
                        text = str(getattr(d, "page_content", d))
                        matches.append(text)
                        if text in self._lru:
                            self._lru.move_to_end(text)
                else:
                    text = str(getattr(doc, "page_content", doc))
                    matches.append(text)
                    if text in self._lru:
                        self._lru.move_to_end(text)
            return matches
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Vector store query failed: %s", exc, exc_info=True)
            return []

    def _graph_facts(self, limit: int) -> List[str]:
        try:
            rows = self._dal.query_subgraph(
                "MATCH (n:Entity) RETURN n.name AS fact LIMIT $limit",
                {"limit": limit},
            )
            return [str(r.get("fact")) for r in rows if r.get("fact")]
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Graph query failed: %s", exc, exc_info=True)
            return []

    def store_interaction(self, text: str) -> None:
        self._add_to_vector(text)
        try:
            self._dal.merge_entity(text)
        except Exception:  # pragma: no cover - defensive
            logger.error("Failed to store interaction in graph", exc_info=True)

    def retrieve_context(self, prompt: str) -> List[str]:
        """Return relevant facts, loading from graph when needed."""
        vector = self._vector_matches(prompt)
        if len(vector) < self._top_k:
            graph = self._graph_facts(self._top_k - len(vector))
            for fact in graph:
                self._add_to_vector(fact)
            vector.extend(graph)
        seen = set()
        merged: List[str] = []
        for item in vector:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        return merged
