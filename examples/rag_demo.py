import json
import logging
from pathlib import Path

from deepthought.search import OfflineSearch
from deepthought.services import CognitiveCoreService

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class DummyNATS:
    pass


class DummyJS:
    pass


def main() -> None:
    data_file = Path(__file__).parent / "data" / "sample_docs.json"
    docs = json.loads(data_file.read_text())
    pairs = [(d["title"], d["content"]) for d in docs]

    search = OfflineSearch.create_index("rag_demo.db", pairs)
    service = CognitiveCoreService(DummyNATS(), DummyJS(), search=search)

    for question in ["What is SQLite?", "Where is Python used?"]:
        ctx = service.retrieve_context(question)
        logger.info("Question: %s", question)
        logger.info("Context: %s", ctx)


if __name__ == "__main__":
    main()
