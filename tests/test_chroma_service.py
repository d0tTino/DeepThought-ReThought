import os

import pytest

from tests.helpers import chroma_available

# Skip the test entirely when the optional chromadb package is missing
chromadb = pytest.importorskip("chromadb")

pytestmark = pytest.mark.chroma


def test_chroma_running():
    if not chroma_available():
        pytest.skip("Chroma service not available")
    client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST", "localhost"), port=int(os.getenv("CHROMA_PORT", 8000)))
    collection = client.get_or_create_collection("test")
    collection.add(documents=["hello"], ids=["1"])
    result = collection.query(query_texts=["hello"], n_results=1)
    assert result["ids"][0][0] == "1"
