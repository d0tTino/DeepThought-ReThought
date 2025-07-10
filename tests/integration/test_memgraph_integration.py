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
    if shutil.which("docker") is None:
        pytest.skip("Docker not available")

    already_running = memgraph_available(MG_HOST, MG_PORT)
    container_name = None

    if not already_running:
        container_name = f"pytest-memgraph-{uuid.uuid4().hex[:8]}"
        cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            f"{MG_PORT}:7687",
            "--name",
            container_name,
            "memgraph/memgraph",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip(f"Could not start memgraph container: {proc.stderr}")

        for _ in range(30):
            if memgraph_available(MG_HOST, MG_PORT):
                break
            time.sleep(1)
        else:
            subprocess.run(["docker", "stop", container_name], capture_output=True)
            pytest.skip("Memgraph container did not start")

    try:
        yield
    finally:
        if container_name:
            subprocess.run(["docker", "stop", container_name], capture_output=True)


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
