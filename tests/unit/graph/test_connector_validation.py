import types

import pytest

from deepthought.graph.connector import GraphConnector, Neo4jConnector


def test_graph_connector_requires_host_and_port(monkeypatch):
    ns = types.SimpleNamespace(mg_host="h", mg_port=7687, mg_user="u", mg_password="p")
    monkeypatch.setattr("deepthought.config.get_settings", lambda: ns)
    with pytest.raises(ValueError):
        GraphConnector(host="", port=0)


def test_neo4j_connector_requires_host_and_port(monkeypatch):
    monkeypatch.delenv("DT_NEO4J_HOST", raising=False)
    monkeypatch.delenv("DT_NEO4J_PORT", raising=False)
    with pytest.raises(ValueError):
        Neo4jConnector(host="", port=0)


def test_graph_connector_env_overrides(monkeypatch):
    monkeypatch.setenv("DT_MG_HOST", "envhost")
    monkeypatch.setenv("DT_MG_PORT", "9999")
    monkeypatch.setenv("DT_MG_USER", "envuser")
    monkeypatch.setenv("DT_MG_PASSWORD", "envpass")

    import deepthought.config as cfg

    monkeypatch.setattr(cfg, "_settings_cache", None)

    connector = GraphConnector()
    assert connector._params["host"] == "envhost"
    assert connector._params["port"] == 9999
    assert connector._params["username"] == "envuser"
    assert connector._params["password"] == "envpass"


def test_neo4j_connector_env_overrides(monkeypatch):
    monkeypatch.setenv("DT_NEO4J_HOST", "neo4jhost")
    monkeypatch.setenv("DT_NEO4J_PORT", "9998")
    monkeypatch.setenv("DT_NEO4J_USER", "neo4juser")
    monkeypatch.setenv("DT_NEO4J_PASSWORD", "neo4jpass")

    import deepthought.config as cfg

    monkeypatch.setattr(cfg, "_settings_cache", None)

    connector = Neo4jConnector()
    assert connector._uri == "bolt://neo4jhost:9998"
    assert connector._auth == ("neo4juser", "neo4jpass")
