import logging

from deepthought.graph import create_graph_backend
from deepthought.memory.tiered import TieredMemory
from deepthought.memory.vector_store import create_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    store = create_vector_store("faiss", collection_name="demo")
    backend = create_graph_backend("neo4j")
    memory = TieredMemory(store, backend, capacity=2, top_k=3)

    for fact in ["Alice met Bob", "Bob met Carol", "Carol met Dave"]:
        memory.store_interaction(fact)

    ctx = memory.retrieve_context("Alice")
    logger.info("Retrieved context: %s", ctx)


if __name__ == "__main__":
    main()
