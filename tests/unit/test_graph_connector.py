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
