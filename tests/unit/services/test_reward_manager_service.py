import json
import sys
import types

import pytest

rm_mod = pytest.importorskip("deepthought.motivate.reward_manager")


fake_nats = types.ModuleType("nats")
fake_nats.aio = types.ModuleType("aio")
fake_client_mod = types.ModuleType("client")
setattr(fake_client_mod, "Client", object)
fake_nats.aio.client = fake_client_mod
fake_msg_mod = types.ModuleType("msg")
setattr(fake_msg_mod, "Msg", object)
fake_nats.aio.msg = fake_msg_mod
fake_nats.js = types.ModuleType("js")
fake_js_client_mod = types.ModuleType("client")
setattr(fake_js_client_mod, "JetStreamContext", object)
fake_nats.js.client = fake_js_client_mod
fake_errors_mod = types.ModuleType("errors")
setattr(fake_errors_mod, "Error", Exception)
fake_nats.errors = fake_errors_mod
sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", fake_nats.aio)
sys.modules.setdefault("nats.aio.client", fake_client_mod)
sys.modules.setdefault("nats.aio.msg", fake_msg_mod)
sys.modules.setdefault("nats.js", fake_nats.js)
sys.modules.setdefault("nats.js.client", fake_js_client_mod)
sys.modules.setdefault("nats.errors", fake_errors_mod)


class DummySubscriber:
    async def subscribe(self, *args, **kwargs):
        pass

    async def unsubscribe_all(self):
        pass


class DummyLedger:
    def __init__(self):
        self.events = []

    async def publish(self, prompt, response, reward):
        self.events.append((prompt, response, reward))


class DummyPublisher:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True):
        self.published.append((subject, payload))


class DummyModel:
    def encode(self, text, convert_to_numpy=True):
        return [0.0]


class DummyMsg:
    def __init__(self, payload, *, raw=False):
        if raw:
            self.data = payload.encode()
        else:
            self.data = json.dumps(payload).encode()
        self.acked = False

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_handle_chat_event_invalid_payload():
    mgr = rm_mod.RewardManager(DummySubscriber(), DummyLedger(), DummyPublisher(), "tok", model=DummyModel())
    msg = DummyMsg("{", raw=True)
    await mgr._handle_chat_event(msg)

    assert msg.acked
    assert not mgr._publisher.published
    assert not mgr._ledger.events


@pytest.mark.asyncio
async def test_handle_chat_event_ledger_failure(monkeypatch):
    class FailingLedger(DummyLedger):
        async def publish(self, prompt, response, reward):
            raise RuntimeError("boom")

    ledger = FailingLedger()
    pub = DummyPublisher()
    mgr = rm_mod.RewardManager(DummySubscriber(), ledger, pub, "tok", model=DummyModel())
    monkeypatch.setattr(mgr, "_score_novelty", lambda _t: 1.0)
    monkeypatch.setattr(mgr, "_score_social", lambda *_a: 0)

    msg = DummyMsg({"prompt": "p", "response": "r"})
    await mgr._handle_chat_event(msg)

    assert not msg.acked
    assert not pub.published
    assert not ledger.events


@pytest.mark.asyncio
async def test_handle_chat_event_publish_failure(monkeypatch):
    ledger = DummyLedger()

    class FailingPublisher(DummyPublisher):
        async def publish(self, *args, **kwargs):
            raise RuntimeError("boom")

    pub = FailingPublisher()
    mgr = rm_mod.RewardManager(DummySubscriber(), ledger, pub, "tok", model=DummyModel())
    monkeypatch.setattr(mgr, "_score_novelty", lambda _t: 1.0)
    monkeypatch.setattr(mgr, "_score_social", lambda *_a: 0)

    msg = DummyMsg({"prompt": "p", "response": "r"})
    await mgr._handle_chat_event(msg)

    assert not msg.acked
    assert ledger.events  # ledger succeeded before failure
    assert not pub.published
