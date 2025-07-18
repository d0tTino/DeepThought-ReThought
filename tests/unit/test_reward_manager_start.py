import types
from types import SimpleNamespace

import pytest

from deepthought.eda.publisher import Publisher
from deepthought.eda.subscriber import Subscriber
from deepthought.motivate.ledger import Ledger
from deepthought.motivate.reward_manager import RewardManager


class DummyNATS:
    def __init__(self):
        self.is_connected = True
        self.subscribed = []

    async def publish(self, subject, data):
        return None

    async def subscribe(self, subject, queue="", cb=None):
        sub = SimpleNamespace(unsubscribed=False)
        self.subscribed.append((subject, queue, cb, sub))
        return sub


class DummyJS:
    def __init__(self):
        self.subscribed = []

    async def publish(self, subject, data, timeout=10.0):
        return SimpleNamespace(seq=1, stream="s")

    async def subscribe(self, subject, queue="", durable="", cb=None, manual_ack=True):
        sub = SimpleNamespace(unsubscribed=False)
        self.subscribed.append((subject, queue, durable, cb, manual_ack, sub))
        return sub


class DummyModel:
    def encode(self, text, convert_to_numpy=True):
        return [0.0]


@pytest.mark.asyncio
async def test_start_listening_with_dummy_clients():
    nc = DummyNATS()
    js = DummyJS()
    sub = Subscriber(nc, js)
    ledger = Ledger(nc, js)
    pub = Publisher(nc, js)
    mgr = RewardManager(sub, ledger, pub, "tok", model=DummyModel())

    result = await mgr.start_listening()

    assert result is True
    assert js.subscribed
    subject, queue, durable, cb, manual_ack, _ = js.subscribed[0]
    assert subject == "chat.bot"
    assert durable == "reward_listener"
    assert callable(cb)

