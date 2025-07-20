"""Memory utilities."""

from dataclasses import dataclass

from ..config import Settings, get_settings
from ..graph import create_graph_backend
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


@dataclass
class MemorySettings:
    """Subset of configuration relevant for memory backends."""

    vector_backend: str
    vector_use_gpu: bool
    graph_backend: str
    memory_capacity: int
    memory_top_k: int


def load_memory_settings(settings: Settings | None = None) -> MemorySettings:
    """Return memory backend configuration from the given ``Settings``."""

    s = settings or get_settings()
    return MemorySettings(
        vector_backend=s.vector_backend,
        vector_use_gpu=s.vector_use_gpu,
        graph_backend=s.graph_backend,
        memory_capacity=s.memory_capacity,
        memory_top_k=s.memory_top_k,
    )


def create_memory_backend(
    settings: Settings | None = None,
    *,
    graph_backend_name: str | None = None,
    collection_name: str = "deepthought",
    persist_directory: str | None = None,
    vector_backend: str | None = None,
    use_gpu: bool | None = None,
    capacity: int | None = None,
    top_k: int | None = None,
) -> TieredMemory:
    """Return :class:`TieredMemory` configured from the provided ``Settings``."""
    full_settings = settings or get_settings()
    settings = load_memory_settings(full_settings)

    store = create_vector_store(
        backend=vector_backend or settings.vector_backend,
        collection_name=collection_name,
        persist_directory=persist_directory,
        use_gpu=use_gpu if use_gpu is not None else settings.vector_use_gpu,
    )

    backend = create_graph_backend(
        graph_backend_name or settings.graph_backend, settings=full_settings
    )

    return TieredMemory(
        store,
        backend,
        capacity=capacity if capacity is not None else settings.memory_capacity,
        top_k=top_k if top_k is not None else settings.memory_top_k,
    )
