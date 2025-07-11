"""Vector store interfaces and implementations."""

from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import chromadb
    from chromadb.api.types import EmbeddingFunction
except Exception:  # pragma: no cover - chromadb not installed
    chromadb = None  # type: ignore
    EmbeddingFunction = object  # type: ignore

    class _DummyCollection:
        def __init__(self) -> None:
            self.docs: dict[str, str] = {}

        def add(self, documents, ids, metadatas=None):  # type: ignore[override]
            for i, doc in zip(ids, documents):
                self.docs[str(i)] = doc

        def query(self, query_texts, n_results=3):  # type: ignore[override]
            docs = [list(self.docs.values())[:n_results] for _ in query_texts]
            return {"documents": docs}

        def count(self) -> int:
            return len(self.docs)

    class _DummyClient:
        def get_or_create_collection(self, name, embedding_function=None):
            return _DummyCollection()

    def _create_client(path=None):
        return _DummyClient()

else:

    def _create_client(path=None):
        if path:
            return chromadb.PersistentClient(path=path)
        return chromadb.Client()


class SimpleEmbeddingFunction(EmbeddingFunction):
    """Deterministic embedding function using SHA1 hashes."""

    def __call__(self, input: List[str]) -> List[List[float]]:  # type: ignore[override]
        vectors: List[List[float]] = []
        for text in input:
            digest = hashlib.sha1(text.encode("utf-8")).digest()[:8]
            vectors.append([b / 255 for b in digest])
        return vectors


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @property
    @abstractmethod
    def collection(self) -> Any:  # pragma: no cover - abstract method
        """Return the underlying collection object."""

    @abstractmethod
    def add_texts(
        self,
        texts: Sequence[str],
        ids: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[dict]] = None,
    ) -> None:
        """Add text documents to the store."""

    @abstractmethod
    def query(self, query_texts: Sequence[str], n_results: int = 3):
        """Query the vector store for matching texts."""


class ChromaVectorStore(VectorStore):
    """Thin wrapper around a Chroma collection."""

    def __init__(
        self,
        collection_name: str = "deepthought",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[EmbeddingFunction] = None,
    ) -> None:
        self._client = _create_client(persist_directory)
        self._collection = self._client.get_or_create_collection(
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
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        else:
            ids = list(ids)
        self._collection.add(
            documents=list(texts),
            ids=ids,
            metadatas=list(metadatas) if metadatas else None,
        )

    def query(self, query_texts: Sequence[str], n_results: int = 3):
        return self._collection.query(query_texts=list(query_texts), n_results=n_results)


try:  # pragma: no cover - optional dependency
    import faiss
except Exception:  # pragma: no cover - faiss not installed
    faiss = None


class FaissVectorStore(VectorStore):
    """In-memory FAISS vector store with optional GPU support."""

    def __init__(
        self,
        embedding_function: Optional[EmbeddingFunction] = None,
        use_gpu: bool = False,
    ) -> None:
        if faiss is None:  # pragma: no cover - defensive
            raise RuntimeError("faiss is not installed")

        self._embedding = embedding_function or SimpleEmbeddingFunction()
        # derive dimensionality from a sample embedding
        dim = len(self._embedding(["sample"])[0])
        index = faiss.IndexFlatL2(dim)
        if use_gpu and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        self._index = index
        self._texts: list[Optional[str]] = []
        self._id_to_idx: dict[str, int] = {}

    @property
    def collection(self):  # pragma: no cover - simple accessor
        class _Coll:
            def __init__(self, outer: "FaissVectorStore") -> None:
                self._outer = outer

            def delete(self, ids: Sequence[str]):
                self._outer._delete(ids)

        return _Coll(self)

    def _delete(self, ids: Sequence[str]) -> None:
        for _id in ids:
            idx = self._id_to_idx.pop(str(_id), None)
            if idx is not None:
                self._texts[idx] = None

    def add_texts(
        self,
        texts: Sequence[str],
        ids: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[dict]] = None,
    ) -> None:  # noqa: D401 - docstring inherited
        ids = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in texts]
        vectors = self._embedding(list(texts))
        import numpy as np

        self._index.add(np.asarray(vectors, dtype="float32"))
        for text, _id in zip(texts, ids):
            self._texts.append(text)
            self._id_to_idx[str(_id)] = len(self._texts) - 1

    def query(self, query_texts: Sequence[str], n_results: int = 3):
        import numpy as np

        vecs = self._embedding(list(query_texts))
        distances, indices = self._index.search(np.asarray(vecs, dtype="float32"), n_results)
        docs: list[list[str]] = []
        for inds in indices:
            hits: list[str] = []
            for i in inds:
                if 0 <= i < len(self._texts):
                    text = self._texts[i]
                    if text is not None:
                        hits.append(text)
            docs.append(hits)
        return {"documents": docs}


def create_vector_store(
    backend: str = "chroma",
    collection_name: str = "deepthought",
    persist_directory: Optional[str] = None,
    embedding_function: Optional[EmbeddingFunction] = None,
    use_gpu: bool = False,
) -> VectorStore:
    """Return a vector store implementation based on ``backend``."""
    if backend == "faiss":
        return FaissVectorStore(embedding_function=embedding_function, use_gpu=use_gpu)
    return ChromaVectorStore(collection_name, persist_directory, embedding_function)
