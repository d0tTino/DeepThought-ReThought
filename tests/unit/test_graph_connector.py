import pytest

from deepthought.graph.connector import GraphConnector


class DummyCursor:
    def __init__(self):
        self.executed = []
        self.closed = False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchall(self):
        return [1]

    def close(self):
        self.closed = True


class DummyConnection:
    def __init__(self):
        self.commit_called = False
        self.cursor_obj = DummyCursor()

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commit_called = True


class DummyExecuteConnection:
    def __init__(self):
        self.commit_called = False
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))
        return self

    def fetchall(self):
        return [1]

    def commit(self):
        self.commit_called = True


def test_execute_commits(monkeypatch):
    conn = DummyConnection()
    connector = GraphConnector()
    monkeypatch.setattr(connector, "connect", lambda: conn)

    result = connector.execute("SELECT 1")

    assert result == [1]
    assert conn.commit_called
    assert conn.cursor_obj.closed


def test_execute_direct_execute_commits(monkeypatch):
    conn = DummyExecuteConnection()
    connector = GraphConnector()
    monkeypatch.setattr(connector, "connect", lambda: conn)

    result = connector.execute("SELECT 1")

    assert result == [1]
    assert conn.commit_called
    assert conn.executed == [("SELECT 1", {})]


def test_env_defaults(monkeypatch):
    monkeypatch.setenv("MG_HOST", "envhost")
    monkeypatch.setenv("MG_PORT", "9999")
    monkeypatch.setenv("MG_USER", "user")
    monkeypatch.setenv("MG_PASSWORD", "pass")
    connector = GraphConnector()
    assert connector._params == {
        "host": "envhost",
        "port": 9999,
        "username": "user",
        "password": "pass",
    }


def test_connect_retries(monkeypatch):
    calls = []

    class FailOnce:
        def __init__(self, **_):
            calls.append("called")
            if len(calls) == 1:
                raise RuntimeError("fail")

    monkeypatch.setattr("deepthought.graph.connector.Memgraph", FailOnce)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    connector = GraphConnector(max_retries=2, retry_delay=0)
    conn = connector.connect()
    assert isinstance(conn, FailOnce)
    assert len(calls) == 2
