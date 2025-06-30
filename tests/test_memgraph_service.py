import os

import pytest

from tests.helpers import memgraph_available

# Skip this test when ``pymemgraph`` is not installed
pymemgraph = pytest.importorskip("pymemgraph")
Memgraph = pymemgraph.Memgraph

pytestmark = pytest.mark.memgraph


def test_memgraph_running():
    if not memgraph_available():
        pytest.skip("Memgraph not available")
    mg = Memgraph(
        host=os.getenv("MG_HOST", "localhost"),
        port=int(os.getenv("MG_PORT", 7687)),
        username=os.getenv("MG_USER", "memgraph"),
        password=os.getenv("MG_PASSWORD", "memgraph"),
    )
    result = mg.execute("RETURN 1 AS num;")
    mg.close()
    row = result[0] if result else None
    assert row and (row[0] == 1 if isinstance(row, tuple) else row.get("num") == 1)
