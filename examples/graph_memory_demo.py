import logging

from deepthought.graph import create_graph_backend
from deepthought.memory.tiered import TieredMemory
from deepthought.memory.vector_store import create_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    store = create_vector_store("faiss", collection_name="graph_demo")
    backend = create_graph_backend("neo4j")
    memory = TieredMemory(store, backend, capacity=3, top_k=2)

    for fact in [
        "Alice met Bob",
        "Bob chatted with Carol",
        "Carol saw Dave",
    ]:
        memory.store_interaction(fact)

    ctx = memory.retrieve_context("Alice")
    logger.info("Retrieved context: %s", ctx)


if __name__ == "__main__":
    main()
