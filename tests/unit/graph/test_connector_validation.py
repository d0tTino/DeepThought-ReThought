import types

import pytest

from deepthought.graph.connector import GraphConnector, Neo4jConnector


def test_graph_connector_requires_host_and_port(monkeypatch):
    ns = types.SimpleNamespace(mg_host="h", mg_port=7687, mg_user="u", mg_password="p")
    monkeypatch.setattr("deepthought.config.get_settings", lambda: ns)
    with pytest.raises(ValueError):
        GraphConnector(host="", port=0)


def test_neo4j_connector_requires_host_and_port(monkeypatch):
    monkeypatch.delenv("NEO4J_HOST", raising=False)
    monkeypatch.delenv("NEO4J_PORT", raising=False)
    with pytest.raises(ValueError):
        Neo4jConnector(host="", port=0)
