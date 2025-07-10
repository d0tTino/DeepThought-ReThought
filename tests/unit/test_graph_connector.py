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
    import types

    fake = types.SimpleNamespace(
        mg_host="envhost",
        mg_port=9999,
        mg_user="user",
        mg_password="pass",
    )
    monkeypatch.setattr("deepthought.config.get_settings", lambda: fake)

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


def test_init_requires_host(monkeypatch):
    import types

    ns = types.SimpleNamespace(mg_host="", mg_port=7687, mg_user="u", mg_password="p")
    monkeypatch.setattr("deepthought.config.get_settings", lambda: ns)

    with pytest.raises(ValueError, match="host"):
        GraphConnector()


def test_init_requires_port(monkeypatch):
    import types

    ns = types.SimpleNamespace(mg_host="h", mg_port=0, mg_user="u", mg_password="p")
    monkeypatch.setattr("deepthought.config.get_settings", lambda: ns)

    with pytest.raises(ValueError, match="port"):
        GraphConnector()


def test_init_port_must_be_positive(monkeypatch):
    import types

    ns = types.SimpleNamespace(mg_host="h", mg_port=-1, mg_user="u", mg_password="p")
    monkeypatch.setattr("deepthought.config.get_settings", lambda: ns)

    with pytest.raises(ValueError, match="positive"):
        GraphConnector()


def test_init_port_must_be_int(monkeypatch):
    import types

    ns = types.SimpleNamespace(mg_host="h", mg_port="abc", mg_user="u", mg_password="p")
    monkeypatch.setattr("deepthought.config.get_settings", lambda: ns)

    with pytest.raises(ValueError, match="integer"):
        GraphConnector()
