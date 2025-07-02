import asyncio
from types import SimpleNamespace

import pytest

import setup_jetstream as sj


class DummyStream(SimpleNamespace):
    pass


class DummyJetStream:
    def __init__(self, add_error: bool = False):
        self.add_called = False
        self.update_called = False
        self.add_error = add_error

    async def add_stream(self, config):
        self.add_called = True
        if self.add_error:
            raise Exception("exists")
        return DummyStream(config=config)

    async def update_stream(self, config):
        self.update_called = True
        return DummyStream(config=config)


class DummyNATS:
    def __init__(self, js: DummyJetStream):
        self._js = js
        self.connected = False
        self.drain_called = False

    async def connect(self, servers):
        self.connected = True

    def jetstream(self):
        return self._js

    async def drain(self):
        self.drain_called = True
        self.connected = False

    @property
    def is_connected(self):
        return self.connected


@pytest.mark.asyncio
async def test_setup_creates_stream(monkeypatch):
    js = DummyJetStream()
    nats = DummyNATS(js)
    monkeypatch.setattr(sj, "check_nats_server_running", lambda url=sj.NATS_URL: True)
    monkeypatch.setattr(sj, "NATS", lambda: nats)

    await sj.setup_jetstream()

    assert js.add_called
    assert not js.update_called
    assert nats.drain_called


@pytest.mark.asyncio
async def test_setup_updates_existing_stream(monkeypatch):
    js = DummyJetStream(add_error=True)
    nats = DummyNATS(js)
    monkeypatch.setattr(sj, "check_nats_server_running", lambda url=sj.NATS_URL: True)
    monkeypatch.setattr(sj, "NATS", lambda: nats)

    await sj.setup_jetstream()

    assert js.add_called
    assert js.update_called
    assert nats.drain_called


@pytest.mark.asyncio
async def test_setup_connection_failure(monkeypatch):
    class FailingNATS(DummyNATS):
        async def connect(self, servers):
            raise sj.TimeoutError()

    js = DummyJetStream()
    nats = FailingNATS(js)
    monkeypatch.setattr(sj, "check_nats_server_running", lambda url=sj.NATS_URL: True)
    monkeypatch.setattr(sj, "NATS", lambda: nats)

    with pytest.raises(sj.JetStreamSetupError) as excinfo:
        await sj.setup_jetstream()

    assert isinstance(excinfo.value.__cause__, sj.TimeoutError)


@pytest.mark.asyncio
async def test_setup_stream_creation_failure(monkeypatch):
    class ErrorJetStream(DummyJetStream):
        async def add_stream(self, config):
            raise Exception("boom")

        async def update_stream(self, config):
            raise Exception("kaboom")

    js = ErrorJetStream()
    nats = DummyNATS(js)
    monkeypatch.setattr(sj, "check_nats_server_running", lambda url=sj.NATS_URL: True)
    monkeypatch.setattr(sj, "NATS", lambda: nats)

    with pytest.raises(sj.JetStreamSetupError) as excinfo:
        await sj.setup_jetstream()

    assert isinstance(excinfo.value.__cause__, Exception)
