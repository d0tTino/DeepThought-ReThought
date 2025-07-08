import pytest

from deepthought.graph.connector import Neo4jConnector


def test_env_defaults(monkeypatch):
    monkeypatch.setenv("NEO4J_HOST", "h")
    monkeypatch.setenv("NEO4J_PORT", "7777")
    monkeypatch.setenv("NEO4J_USER", "u")
    monkeypatch.setenv("NEO4J_PASSWORD", "p")
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
