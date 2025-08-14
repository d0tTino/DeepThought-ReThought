"""Tests for the generic JetStream publisher."""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock

import pytest

# Stub out nats modules if not installed
fake_nats = types.ModuleType("nats")
fake_nats.errors = types.SimpleNamespace(TimeoutError=Exception)
aio_mod = types.ModuleType("nats.aio")
client_mod = types.ModuleType("nats.aio.client")
client_mod.Client = object
aio_mod.client = client_mod
js_client_mod = types.ModuleType("nats.js.client")
js_client_mod.JetStreamContext = object
sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", aio_mod)
sys.modules.setdefault("nats.aio.client", client_mod)
sys.modules.setdefault("nats.js.client", js_client_mod)

from deepthought.eda.publisher import Publisher


class StubNATS:
    """Simple stub NATS client."""

    def __init__(self) -> None:
        self.is_connected = True


class StubJS:
    """JetStream stub that fails once before succeeding."""

    def __init__(self) -> None:
        self.attempts = 0

    async def publish(self, subject, data, timeout=10.0):  # pragma: no cover - signature
        self.attempts += 1
        if self.attempts < 2:
            raise RuntimeError("fail")

        class Ack:
            def __init__(self) -> None:
                self.seq = 1
                self.stream = "test"

        return Ack()


@pytest.mark.asyncio
async def test_publish_retries_until_success():
    nc = StubNATS()
    js = StubJS()
    pub = Publisher(nc, js)

    ack = await pub.publish("sub", {"a": 1}, retries=3)

    assert ack == {"seq": 1, "stream": "test"}
    assert js.attempts == 2


@pytest.mark.asyncio
async def test_publish_raises_after_exhausting_retries():
    nc = StubNATS()
    js = StubJS()
    js.publish = AsyncMock(side_effect=RuntimeError("fail"))
    pub = Publisher(nc, js)

    with pytest.raises(RuntimeError):
        await pub.publish("sub", b"data", retries=2)
