import pytest

import deepthought.config as cfg
from deepthought.graph.connector import Neo4jConnector


def test_env_defaults(monkeypatch):
    monkeypatch.setenv("DT_NEO4J_HOST", "h")
    monkeypatch.setenv("DT_NEO4J_PORT", "7777")
    monkeypatch.setenv("DT_NEO4J_USER", "u")
    monkeypatch.setenv("DT_NEO4J_PASSWORD", "p")
    monkeypatch.setattr(cfg, "_settings_cache", None)
    c = Neo4jConnector()
    assert c._uri == "bolt://h:7777"
    assert c._auth == ("u", "p")


def test_connect_retries(monkeypatch):
    calls = []

    class DummyDB:
        def driver(self, uri, auth):
            calls.append((uri, auth))
            if len(calls) == 1:
                raise RuntimeError("fail")
            return "driver"

    monkeypatch.setattr("deepthought.graph.connector.GraphDatabase", DummyDB())
    monkeypatch.setattr("time.sleep", lambda *_: None)
    connector = Neo4jConnector(max_retries=2, retry_delay=0)
    driver = connector.connect()
    assert driver == "driver"
    assert len(calls) == 2


def test_settings_argument():
    from deepthought.config import Settings

    s = Settings(neo4j_host="h", neo4j_port=11, neo4j_user="u", neo4j_password="p")
    c = Neo4jConnector(settings=s)

    assert c._uri == "bolt://h:11"
    assert c._auth == ("u", "p")
