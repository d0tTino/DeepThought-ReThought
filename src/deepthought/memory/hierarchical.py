from __future__ import annotations

"""Hierarchical memory retrieval mixing vector and graph lookups."""

import logging
from typing import Any, List, Sequence

from ..graph import GraphDAL

logger = logging.getLogger(__name__)


class HierarchicalMemory:
    """Combine vector store search with graph facts."""

    def __init__(self, vector_store: Any, graph_dal: GraphDAL, top_k: int = 3) -> None:
        self._vector_store = vector_store
        self._dal = graph_dal
        self._top_k = top_k

    def _vector_matches(self, prompt: str) -> List[str]:
        if self._vector_store is None:
            return []
        try:
            result = self._vector_store.query(query_texts=[prompt], n_results=self._top_k)
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
                        matches.append(str(getattr(d, "page_content", d)))
                else:
                    matches.append(str(getattr(doc, "page_content", doc)))
            return matches
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Vector store query failed: %s", exc, exc_info=True)
            return []

    def _graph_facts(self) -> List[str]:
        try:
            rows = self._dal.query_subgraph(
                "MATCH (n:Entity) RETURN n.name AS fact LIMIT $limit",
                {"limit": self._top_k},
            )
            return [str(r.get("fact")) for r in rows if r.get("fact")]
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Graph query failed: %s", exc, exc_info=True)
            return []

    def retrieve_context(self, prompt: str) -> List[str]:
        """Return merged vector matches and graph facts."""
        vector = self._vector_matches(prompt)
        graph = self._graph_facts()
        seen = set()
        merged: List[str] = []
        for item in vector + graph:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        return merged
