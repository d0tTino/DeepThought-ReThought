"""Lightweight wrapper around chromadb collections."""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import chromadb
    from chromadb.api.types import EmbeddingFunction
except Exception:  # pragma: no cover - chromadb not installed
    chromadb = None  # type: ignore
    EmbeddingFunction = object  # type: ignore


class SimpleEmbeddingFunction(EmbeddingFunction):
    """Deterministic embedding function using SHA1 hashes."""

    def __call__(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha1(text.encode("utf-8")).digest()[:8]
            vectors.append([b / 255 for b in digest])
        return vectors


class VectorStore:
    """Small helper around a Chroma collection."""

    def __init__(
        self,
        collection_name: str = "deepthought",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[EmbeddingFunction] = None,
    ) -> None:
        if chromadb is None:
            raise ImportError("chromadb is required for VectorStore")
        if persist_directory:
            client = chromadb.PersistentClient(path=persist_directory)
        else:
            client = chromadb.Client()
        self._client = client
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function or SimpleEmbeddingFunction(),
        )

    @property
    def collection(self):
        return self._collection

    def add_texts(
        self,
        texts: Sequence[str],
        ids: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[dict]] = None,
    ) -> None:
        ids = list(ids) if ids is not None else [str(i) for i in range(len(texts))]
        self._collection.add(documents=list(texts), ids=ids, metadatas=list(metadatas) if metadatas else None)

    def query(self, query_texts: Sequence[str], n_results: int = 3):
        return self._collection.query(query_texts=list(query_texts), n_results=n_results)


def create_vector_store(
    collection_name: str = "deepthought",
    persist_directory: Optional[str] = None,
    embedding_function: Optional[EmbeddingFunction] = None,
) -> VectorStore:
    """Convenience initializer returning a :class:`VectorStore`."""
    return VectorStore(collection_name, persist_directory, embedding_function)
