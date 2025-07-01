import uuid
from typing import Optional, Sequence

import faiss
import numpy as np

from .vector_store import SimpleEmbeddingFunction


class FaissVectorStore:
    """Minimal FAISS-backed vector store using an embedding function."""

    def __init__(self, embedding_dim: int = 8, embedding_function: Optional[SimpleEmbeddingFunction] = None) -> None:
        self._dim = embedding_dim
        self._embedding = embedding_function or SimpleEmbeddingFunction()
        self._index = faiss.IndexFlatL2(embedding_dim)
        self._id_map: list[str] = []
        self._docs: dict[str, str] = {}

    def add_texts(self, texts: Sequence[str], ids: Optional[Sequence[str]] = None) -> None:
        ids = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in texts]
        vectors = np.asarray(self._embedding(list(texts)), dtype="float32")
        self._index.add(vectors)
        for _id, text in zip(ids, texts):
            self._id_map.append(_id)
            self._docs[_id] = text

    def query(self, query_texts: Sequence[str], n_results: int = 3):
        vectors = np.asarray(self._embedding(list(query_texts)), dtype="float32")
        _, indices = self._index.search(vectors, n_results)
        docs = [[self._docs[self._id_map[i]] for i in idx if i < len(self._id_map)] for idx in indices]
        return {"documents": docs}
