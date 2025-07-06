import os
import shutil
import subprocess
import time
import uuid

import pytest

from deepthought.graph import Neo4jConnector, GraphDAL
from tests.helpers import neo4j_available

try:  # optional dependency
    import neo4j  # noqa: F401
except Exception:  # pragma: no cover - dependency missing
    neo4j = None

pytestmark = pytest.mark.neo4j

NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT = int(os.getenv("NEO4J_PORT", 7687))
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")


@pytest.fixture(scope="module")
def neo4j_server():
    """Start a Neo4j Docker container if needed and yield when ready."""
    if neo4j is None:
        pytest.skip("neo4j-driver not installed")
    if shutil.which("docker") is None:
        pytest.skip("Docker not available")

    already_running = neo4j_available(NEO4J_HOST, NEO4J_PORT)
    container_name = None

    if not already_running:
        container_name = f"pytest-neo4j-{uuid.uuid4().hex[:8]}"
        cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            f"{NEO4J_PORT}:7687",
            "-e",
            f"NEO4J_AUTH={NEO4J_USER}/{NEO4J_PASSWORD}",
            "--name",
            container_name,
            "neo4j:5",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip(f"Could not start neo4j container: {proc.stderr}")

        for _ in range(30):
            if neo4j_available(NEO4J_HOST, NEO4J_PORT):
                break
            time.sleep(1)
        else:
            subprocess.run(["docker", "stop", container_name], capture_output=True)
            pytest.skip("Neo4j container did not start")

    try:
        yield
    finally:
        if container_name:
            subprocess.run(["docker", "stop", container_name], capture_output=True)


def test_graphdal_live(neo4j_server):
    """Ensure GraphDAL can interact with a live Neo4j instance."""
    if not neo4j_available(NEO4J_HOST, NEO4J_PORT):
        pytest.skip("Neo4j not reachable")

    connector = Neo4jConnector(
        host=NEO4J_HOST,
        port=NEO4J_PORT,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )
    dal = GraphDAL(connector)

    alice_id = str(uuid.uuid4())
    bob_id = str(uuid.uuid4())
    dal.add_entity("Person", {"id": alice_id, "name": "Alice"})
    dal.add_entity("Person", {"id": bob_id, "name": "Bob"})
    dal.add_relationship(alice_id, bob_id, "KNOWS", {"since": 2024})

    result = dal.query_subgraph(
        "MATCH (a:Person {id: $a_id})-[r:KNOWS]->(b:Person {id: $b_id}) RETURN a.name AS src, b.name AS dst",
        {"a_id": alice_id, "b_id": bob_id},
    )
    assert result
    row = result[0]
    if isinstance(row, tuple):
        assert row[0] == "Alice" and row[1] == "Bob"
    else:
        src = row["src"] if isinstance(row, dict) else row.get("src")
        dst = row["dst"] if isinstance(row, dict) else row.get("dst")
        assert src == "Alice" and dst == "Bob"
