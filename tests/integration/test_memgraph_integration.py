import os
import shutil
import subprocess
import time
import uuid

import pytest

from deepthought.graph import GraphConnector, GraphDAL
from tests.helpers import memgraph_available

try:  # Import optional dependency inside the module
    import pymemgraph  # noqa: F401
except Exception:  # pragma: no cover - dependency missing
    pymemgraph = None

pytestmark = pytest.mark.memgraph

MG_HOST = os.getenv("MG_HOST", "localhost")
MG_PORT = int(os.getenv("MG_PORT", 7687))
MG_USER = os.getenv("MG_USER", "memgraph")
MG_PASSWORD = os.getenv("MG_PASSWORD", "memgraph")


@pytest.fixture(scope="module")
def memgraph_server():
    """Start a Memgraph Docker container if needed and yield when ready."""
    if pymemgraph is None:
        pytest.skip("pymemgraph not installed")
    compose = None
    if shutil.which("docker-compose"):
        compose = ["docker-compose"]
    elif shutil.which("docker"):
        compose = ["docker", "compose"]
    if compose is None:
        pytest.skip("Docker compose not available")

    already_running = memgraph_available(MG_HOST, MG_PORT)

    if not already_running:
        cmd = compose + ["-f", "docker-compose.yml", "up", "-d", "memgraph"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip(f"Could not start memgraph service: {proc.stderr}")

        for _ in range(30):
            if memgraph_available(MG_HOST, MG_PORT):
                break
            time.sleep(1)
        else:
            subprocess.run(compose + ["-f", "docker-compose.yml", "down"], capture_output=True)
            pytest.skip("Memgraph container did not start")

    try:
        yield
    finally:
        if not already_running:
            subprocess.run(compose + ["-f", "docker-compose.yml", "down"], capture_output=True)


def test_graphdal_live(memgraph_server):
    """Ensure GraphDAL can interact with a live Memgraph instance."""
    if not memgraph_available(MG_HOST, MG_PORT):
        pytest.skip("Memgraph not reachable")

    connector = GraphConnector(
        host=MG_HOST,
        port=MG_PORT,
        username=MG_USER,
        password=MG_PASSWORD,
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
        assert row.get("src") == "Alice" and row.get("dst") == "Bob"


def test_graphdal_crud(memgraph_server):
    """Verify basic CRUD operations against Memgraph."""
    if not memgraph_available(MG_HOST, MG_PORT):
        pytest.skip("Memgraph not reachable")

    connector = GraphConnector(
        host=MG_HOST,
        port=MG_PORT,
        username=MG_USER,
        password=MG_PASSWORD,
    )
    dal = GraphDAL(connector)

    entity_id = str(uuid.uuid4())

    # Create
    dal.add_entity("Person", {"id": entity_id, "name": "Charlie"})

    # Read
    rows = dal.query_subgraph(
        "MATCH (n:Person {id: $id}) RETURN n.name AS name",
        {"id": entity_id},
    )
    assert rows
    row = rows[0]
    name = row[0] if isinstance(row, tuple) else row.get("name")
    assert name == "Charlie"

    # Update
    dal.query_subgraph(
        "MATCH (n:Person {id: $id}) SET n.name = $name",
        {"id": entity_id, "name": "Dave"},
    )
    rows = dal.query_subgraph(
        "MATCH (n:Person {id: $id}) RETURN n.name AS name",
        {"id": entity_id},
    )
    row = rows[0]
    name = row[0] if isinstance(row, tuple) else row.get("name")
    assert name == "Dave"

    # Delete
    dal.query_subgraph(
        "MATCH (n:Person {id: $id}) DETACH DELETE n",
        {"id": entity_id},
    )
    rows = dal.query_subgraph(
        "MATCH (n:Person {id: $id}) RETURN n",
        {"id": entity_id},
    )
    assert not rows


def test_connector_requires_host(monkeypatch):
    import types

    ns = types.SimpleNamespace(
        mg_host="",
        mg_port=MG_PORT,
        mg_user=MG_USER,
        mg_password=MG_PASSWORD,
    )
    monkeypatch.setattr("deepthought.config.get_settings", lambda: ns)

    with pytest.raises(ValueError, match="host"):
        GraphConnector()


def test_connector_requires_port(monkeypatch):
    import types

    ns = types.SimpleNamespace(
        mg_host=MG_HOST,
        mg_port=0,
        mg_user=MG_USER,
        mg_password=MG_PASSWORD,
    )
    monkeypatch.setattr("deepthought.config.get_settings", lambda: ns)

    with pytest.raises(ValueError, match="port"):
        GraphConnector()


def test_connector_port_must_be_positive(monkeypatch):
    import types

    ns = types.SimpleNamespace(
        mg_host=MG_HOST,
        mg_port=-1,
        mg_user=MG_USER,
        mg_password=MG_PASSWORD,
    )
    monkeypatch.setattr("deepthought.config.get_settings", lambda: ns)

    with pytest.raises(ValueError, match="positive"):
        GraphConnector()


def test_connector_port_must_be_int(monkeypatch):
    import types

    ns = types.SimpleNamespace(
        mg_host=MG_HOST,
        mg_port="abc",
        mg_user=MG_USER,
        mg_password=MG_PASSWORD,
    )
    monkeypatch.setattr("deepthought.config.get_settings", lambda: ns)

    with pytest.raises(ValueError, match="integer"):
        GraphConnector()
