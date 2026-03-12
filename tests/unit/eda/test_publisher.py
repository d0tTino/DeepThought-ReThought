"""Tests for the generic JetStream publisher."""

from __future__ import annotations

import sys
import types
import json
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

from deepthought.eda.publisher import Publisher, publish_enveloped  # noqa: E402


class StubNATS:
    """Simple stub NATS client."""

    def __init__(self) -> None:
        self.is_connected = True


class StubJS:
    """JetStream stub that fails once before succeeding."""

    def __init__(self) -> None:
        self.attempts = 0
        self.last_subject = None
        self.last_payload = None

    async def publish(self, subject, data, timeout=10.0):  # pragma: no cover - signature
        self.attempts += 1
        self.last_subject = subject
        if isinstance(data, bytes):
            self.last_payload = json.loads(data.decode())
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


@pytest.mark.asyncio
async def test_publish_enveloped_builds_required_envelope_metadata():
    nc = StubNATS()
    js = StubJS()
    pub = Publisher(nc, js)

    await publish_enveloped(
        pub,
        subject="dtr.social.signals.retrieved",
        payload={"input_id": "in-1"},
        producer="social_graph_service",
    )

    assert js.attempts == 2
    assert js.last_subject == "dtr.social.signals.retrieved"
    assert js.last_payload["trace_id"]
    assert js.last_payload["event_id"]
    assert js.last_payload["causation_id"]
    assert js.last_payload["payload"]["input_id"] == "in-1"
