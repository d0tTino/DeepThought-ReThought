import os
import sys
import types
import uuid

import pytest

sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
fake_pyd = types.ModuleType("pydantic")
fake_pyd.AnyUrl = str
fake_pyd.ValidationError = Exception
fake_pyd.Field = lambda default=None, **kwargs: default
sys.modules.setdefault("pydantic", fake_pyd)
fake_ps = types.ModuleType("pydantic_settings")


class DummyBase:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


fake_ps.BaseSettings = DummyBase
fake_ps.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_ps)

from deepthought.graph import GraphConnector, GraphDAL, Neo4jConnector
from deepthought.graph.backend import GraphDALBackend
from deepthought.memory.tiered import TieredMemory
from deepthought.memory.vector_store import create_vector_store
from tests.helpers import memgraph_available, neo4j_available
from tests.integration.test_memgraph_integration import memgraph_server
from tests.integration.test_neo4j_integration import neo4j_server

MG_HOST = os.getenv("MG_HOST", "localhost")
MG_PORT = int(os.getenv("MG_PORT", 7687))
MG_USER = os.getenv("MG_USER", "memgraph")
MG_PASSWORD = os.getenv("MG_PASSWORD", "memgraph")

NEO4J_HOST = os.getenv("NEO4J_HOST", "localhost")
NEO4J_PORT = int(os.getenv("NEO4J_PORT", 7687))
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")


@pytest.mark.memgraph
def test_retrieve_from_memgraph(memgraph_server):
    if not memgraph_available(MG_HOST, MG_PORT):
        pytest.skip("Memgraph not reachable")

    connector = GraphConnector(
        host=MG_HOST,
        port=MG_PORT,
        username=MG_USER,
        password=MG_PASSWORD,
    )
    dal = GraphDAL(connector)
    backend = GraphDALBackend(dal)

    store = create_vector_store("chroma", collection_name=f"kg-{uuid.uuid4()}")
    memory = TieredMemory(store, backend, capacity=0, top_k=2)

    memory.store_interaction("Alice loves Bob")
    memory.store_interaction("Bob loves Carol")

    ctx = memory.retrieve_context("test")
    assert "Alice loves Bob" in ctx and "Bob loves Carol" in ctx


@pytest.mark.neo4j
def test_retrieve_from_neo4j(neo4j_server):
    if not neo4j_available(NEO4J_HOST, NEO4J_PORT):
        pytest.skip("Neo4j not reachable")

    connector = Neo4jConnector(
        host=NEO4J_HOST,
        port=NEO4J_PORT,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )
    dal = GraphDAL(connector)
    backend = GraphDALBackend(dal)

    store = create_vector_store("chroma", collection_name=f"kg-{uuid.uuid4()}")
    memory = TieredMemory(store, backend, capacity=0, top_k=2)

    memory.store_interaction("X meets Y")
    memory.store_interaction("Y meets Z")

    ctx = memory.retrieve_context("test")
    assert "X meets Y" in ctx and "Y meets Z" in ctx
