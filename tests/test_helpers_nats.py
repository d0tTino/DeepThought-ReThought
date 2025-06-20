import socket

from tests.helpers import nats_server_available


class DummySocket:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


def test_nats_server_available_with_port(monkeypatch):
    captured = {}

    def fake_create_connection(addr, timeout=1):
        captured["addr"] = addr
        return DummySocket()

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)

    assert nats_server_available("nats://example.com:4222") is True
    assert captured["addr"] == ("example.com", 4222)


def test_nats_server_available_without_port(monkeypatch):
    captured = {}

    def fake_create_connection(addr, timeout=1):
        captured["addr"] = addr
        return DummySocket()

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)

    assert nats_server_available("nats://example.com") is True
    # Default NATS port 4222 should be used
    assert captured["addr"] == ("example.com", 4222)
