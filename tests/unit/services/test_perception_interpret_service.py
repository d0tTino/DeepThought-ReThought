import importlib.machinery
import json
import sys
import types

import pytest

if "nats" not in sys.modules:
    nats_stub = types.ModuleType("nats")
    nats_stub.__spec__ = importlib.machinery.ModuleSpec("nats", loader=None)
    nats_stub.aio = types.ModuleType("aio")
    client_mod = types.ModuleType("client")
    client_mod.Client = object
    msg_mod = types.ModuleType("msg")
    msg_mod.Msg = object
    nats_stub.aio.client = client_mod
    nats_stub.aio.msg = msg_mod
    nats_stub.js = types.ModuleType("js")
    js_client_mod = types.ModuleType("client")
    js_client_mod.JetStreamContext = object
    nats_stub.js.client = js_client_mod
    err_mod = types.ModuleType("errors")
    err_mod.Error = Exception
    nats_stub.errors = err_mod
    sys.modules.setdefault("nats", nats_stub)
    sys.modules.setdefault("nats.aio", nats_stub.aio)
    sys.modules.setdefault("nats.aio.client", client_mod)
    sys.modules.setdefault("nats.aio.msg", msg_mod)
    sys.modules.setdefault("nats.js", nats_stub.js)
    sys.modules.setdefault("nats.js.client", js_client_mod)
    sys.modules.setdefault("nats.errors", err_mod)

from deepthought.eda.events import EventSubjects
from deepthought.services.perception_interpret_service import PerceptionInterpretService


class DummyNATS:
    pass


class DummyJS:
    pass


class DummySubscriber:
    def __init__(self, *_args, **_kwargs):
        pass

    async def subscribe(self, **kwargs):
        return True

    async def unsubscribe_all(self):
        return None


class RecordingPublisher:
    def __init__(self, *_args, **_kwargs):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload, use_jetstream, timeout))


class DummyMsg:
    def __init__(self, payload):
        self.data = json.dumps(payload).encode()
        self.acked = False
        self.naked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.naked = True


def test_perception_interpret_service_evicts_after_publish(monkeypatch):
    import deepthought.services.perception_interpret_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", DummySubscriber)

    service = PerceptionInterpretService(DummyNATS(), DummyJS(), cache_max_entries=4, cache_max_age_seconds=60)

    embeddings_msg = DummyMsg(
        {
            "event": EventSubjects.PERCEPTION_EMBEDDINGS,
            "version": 1,
            "payload": {
                "message_id": "m-1",
                "user_id": "u-1",
                "input_id": "input-1",
                "confidence": 0.8,
                "modality_confidence": {"image": 0.7},
                "by_modality": {
                    "image": {
                        "spans": [[0, 1]],
                        "embeddings": [[0.1, 0.2]],
                        "encoders": [],
                    }
                },
            },
        }
    )
    import asyncio

    asyncio.run(service._handle_embeddings(embeddings_msg))

    request_msg = DummyMsg(
        {
            "input_id": "input-1",
            "attachments": [{"url": "https://example.test/image.png", "content_type": "image/png"}],
        }
    )
    asyncio.run(service._handle_interpret_request(request_msg))

    assert embeddings_msg.acked is True
    assert request_msg.acked is True
    assert service.cache_metrics["cache_size"] == 0
    assert service.cache_metrics["evictions"] == 1
    assert service.cache_metrics["hit_rate"] == 1.0


def test_perception_interpret_service_evicts_expired_and_lru(monkeypatch):
    import deepthought.services.perception_interpret_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", DummySubscriber)

    service = PerceptionInterpretService(DummyNATS(), DummyJS(), cache_max_entries=2, cache_max_age_seconds=0.1)

    service._cache_put("a", {"value": "a"})
    service._cache_put("b", {"value": "b"})
    service._cache_put("c", {"value": "c"})

    assert service._cache_get("a") == {}
    assert service._cache_get("b") == {"value": "b"}
    assert service._cache_get("c") == {"value": "c"}

    service._embeddings_by_input_id["b"]["cached_at"] -= 1.0
    assert service._cache_get("missing") == {}
    assert service._cache_get("b") == {}

    metrics = service.cache_metrics
    assert metrics["cache_size"] == 1
    assert metrics["evictions"] == 2
    assert metrics["hit_rate"] == pytest.approx(0.4)
