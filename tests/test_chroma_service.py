import os

import pytest

from tests.helpers import chroma_available

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency
    chromadb = None

pytestmark = pytest.mark.chroma


def test_chroma_running():
    if chromadb is None:
        pytest.skip("chromadb not installed")
    if not chroma_available():
        pytest.skip("Chroma service not available")
    client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST", "localhost"), port=int(os.getenv("CHROMA_PORT", 8000)))
    collection = client.get_or_create_collection("test")
    collection.add(documents=["hello"], ids=["1"])
    result = collection.query(query_texts=["hello"], n_results=1)
    assert result["ids"][0][0] == "1"
