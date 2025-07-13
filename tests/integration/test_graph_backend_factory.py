import os
import uuid

import pytest

from deepthought.graph import create_graph_backend
from deepthought.graph.backend import GraphDALBackend, Neo4jBackend
from tests.helpers import memgraph_available, neo4j_available
from tests.integration.test_memgraph_integration import memgraph_server
from tests.integration.test_neo4j_integration import neo4j_server

# Environment defaults reused from other integration tests
MG_HOST = os.getenv("MG_HOST", "localhost")
MG_PORT = int(os.getenv("MG_PORT", 7687))
MG_USER = os.getenv("MG_USER", "memgraph")
MG_PASSWORD = os.getenv("MG_PASSWORD", "memgraph")

NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT = int(os.getenv("NEO4J_PORT", 7687))
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")

pytestmark = pytest.mark.integration


@pytest.mark.memgraph
def test_factory_memgraph_live(memgraph_server, monkeypatch):
    if not memgraph_available(MG_HOST, MG_PORT):
        pytest.skip("Memgraph not reachable")

    monkeypatch.setenv("DT_MG_HOST", MG_HOST)
    monkeypatch.setenv("DT_MG_PORT", str(MG_PORT))
    monkeypatch.setenv("DT_MG_USER", MG_USER)
    monkeypatch.setenv("DT_MG_PASSWORD", MG_PASSWORD)

    backend = create_graph_backend("memgraph")
    assert isinstance(backend, GraphDALBackend)

    name = f"Alice-{uuid.uuid4()}"
    backend.merge_entity(name)
    rows = backend.query_subgraph(
        "MATCH (e:Entity {name: $name}) RETURN e.name AS name",
        {"name": name},
    )
    assert rows


@pytest.mark.neo4j
def test_factory_neo4j_live(neo4j_server, monkeypatch):
    if not neo4j_available(NEO4J_HOST, NEO4J_PORT):
        pytest.skip("Neo4j not reachable")

    monkeypatch.setenv("DT_NEO4J_HOST", NEO4J_HOST)
    monkeypatch.setenv("DT_NEO4J_PORT", str(NEO4J_PORT))
    monkeypatch.setenv("DT_NEO4J_USER", NEO4J_USER)
    monkeypatch.setenv("DT_NEO4J_PASSWORD", NEO4J_PASSWORD)

    backend = create_graph_backend("neo4j")
    assert isinstance(backend, Neo4jBackend)

    name = f"Bob-{uuid.uuid4()}"
    backend.merge_entity(name)
    rows = backend.query_subgraph(
        "MATCH (e:Entity {name: $name}) RETURN e.name AS name",
        {"name": name},
    )
    assert rows
