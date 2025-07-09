"""Memory utilities."""

from ..config import get_settings
from ..graph import create_graph_backend
from .faiss_vector_store import FaissVectorStore
from .hierarchical import HierarchicalMemory
from .tiered import TieredMemory
from .vector_store import (
    ChromaVectorStore,
    FaissVectorStore,
    SimpleEmbeddingFunction,
    VectorStore,
    create_vector_store,
)

__all__ = [
    "HierarchicalMemory",
    "VectorStore",
    "ChromaVectorStore",
    "FaissVectorStore",
    "create_vector_store",
    "SimpleEmbeddingFunction",
    "TieredMemory",
    "create_memory_backend",
]


def create_memory_backend(
    *,
    graph_backend_name: str | None = None,
    collection_name: str = "deepthought",
    persist_directory: str | None = None,
    vector_backend: str | None = None,
    use_gpu: bool | None = None,
    capacity: int = 100,
    top_k: int = 3,
) -> TieredMemory:
    """Return :class:`TieredMemory` configured from environment variables."""

    settings = get_settings()

    store = create_vector_store(
        backend=vector_backend or settings.vector_backend,
        collection_name=collection_name,
        persist_directory=persist_directory,
        use_gpu=use_gpu if use_gpu is not None else settings.vector_use_gpu,
    )

    backend = create_graph_backend(graph_backend_name or settings.graph_backend)

    return TieredMemory(store, backend, capacity=capacity, top_k=top_k)
