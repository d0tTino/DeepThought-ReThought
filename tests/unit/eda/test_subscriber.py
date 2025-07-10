import logging

import pytest

pytest.importorskip("nats")
import nats

from deepthought.eda.subscriber import Subscriber


class DummyNATS:
    def __init__(self, error=None):
        self.is_connected = True
        self.error = error

    async def subscribe(self, *args, **kwargs):
        if self.error:
            raise self.error
        return object()


class DummyJS:
    def __init__(self, error=None):
        self.error = error

    async def subscribe(self, *args, **kwargs):
        if self.error:
            raise self.error
        return object()


async def dummy_handler(msg):
    return None


@pytest.mark.asyncio
async def test_subscribe_timeout_logs_and_returns_false(caplog):
    nc = DummyNATS(error=nats.errors.TimeoutError("no sub"))
    sub = Subscriber(nc, DummyJS())

    with caplog.at_level(logging.ERROR):
        result = await sub.subscribe("topic", dummy_handler)

    assert result is False
    assert any("Failed to subscribe" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_js_subscribe_error(caplog):
    js = DummyJS(error=RuntimeError("boom"))
    sub = Subscriber(DummyNATS(), js)

    with caplog.at_level(logging.ERROR):
        result = await sub.subscribe("topic", dummy_handler, use_jetstream=True, durable="d")

    assert result is False
    assert any("Failed to subscribe" in r.getMessage() for r in caplog.records)
