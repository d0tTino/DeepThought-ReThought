import json
import logging
from pathlib import Path

from deepthought.search import OfflineSearch
from deepthought.services import CognitiveCoreService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DummyNATS:
    pass


class DummyJS:
    pass


def main() -> None:
    data_file = Path(__file__).parent / "data" / "sample_docs.json"
    docs = json.loads(data_file.read_text())
    pairs = [(d["title"], d["content"]) for d in docs]

    index_path = Path("offline_demo.db")
    search = OfflineSearch.create_index(str(index_path), pairs)

    service = CognitiveCoreService(DummyNATS(), DummyJS(), search=search)
    results = service.retrieve_context("database")
    logger.info("Results: %s", results)


if __name__ == "__main__":
    main()
