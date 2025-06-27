import json
import sys
import types
from types import SimpleNamespace

import pytest

# Avoid importing heavy optional modules when loading the package
for _name in ["deepthought.harness", "deepthought.learn", "deepthought.modules", "deepthought.motivate"]:
    sys.modules.setdefault(_name, types.ModuleType(_name))

from deepthought.eda.publisher import Publisher
from deepthought.eda.subscriber import Subscriber


class DummyAck(SimpleNamespace):
    pass


class DummyNATS:
    def __init__(self):
        self.is_connected = True
        self.published = []
        self.subscribed = []

    async def publish(self, subject, data):
        self.published.append((subject, data))

    async def subscribe(self, subject, queue="", cb=None):
        sub = SimpleNamespace(unsubscribed=False)

        async def _unsub():
            sub.unsubscribed = True

        sub.unsubscribe = _unsub
        self.subscribed.append((subject, queue, cb, sub))
        return sub


class DummyJS:
    def __init__(self):
        self.published = []
        self.subscribed = []

    async def publish(self, subject, data, timeout=10.0):
        self.published.append((subject, data, timeout))
        return DummyAck(seq=1, stream="s")

    async def subscribe(self, subject, queue="", durable="", cb=None, manual_ack=True):
        sub = SimpleNamespace(unsubscribed=False)

        async def _unsub():
            sub.unsubscribed = True

        sub.unsubscribe = _unsub
        self.subscribed.append((subject, queue, durable, cb, manual_ack, sub))
        return sub


def compute_expected(payload):
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode()
    if hasattr(payload, "to_json"):
        return payload.to_json().encode()
    if isinstance(payload, (dict, list)):
        return json.dumps(payload).encode()
    return str(payload).encode()


class WithToJson:
    def __init__(self, val="x"):
        self.val = val

    def to_json(self):
        return json.dumps({"val": self.val})


class DummyObj:
    def __str__(self):
        return "dummy"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        "text",
        b"bin",
        {"a": 1},
        [1, 2],
        WithToJson("v"),
        DummyObj(),
    ],
)
async def test_publish_serialization(payload):
    nc = DummyNATS()
    js = DummyJS()
    pub = Publisher(nc, js)
    await pub.publish("subj", payload, use_jetstream=True)

    assert js.published
    subject, data, _ = js.published[0]
    assert subject == "subj"
    assert data == compute_expected(payload)


@pytest.mark.asyncio
async def test_publish_basic_nats():
    nc = DummyNATS()
    js = DummyJS()
    pub = Publisher(nc, js)

    ack = await pub.publish("subj", "hi", use_jetstream=False)

    assert ack is None
    assert nc.published == [("subj", b"hi")]
    assert not js.published


@pytest.mark.asyncio
async def test_subscriber_basic_subscribe_and_unsubscribe():
    nc = DummyNATS()
    js = DummyJS()
    sub = Subscriber(nc, js)

    async def handler(msg):
        return None

    await sub.subscribe("topic", handler)
    assert nc.subscribed
    unsub_obj = nc.subscribed[0][3]

    await sub.unsubscribe_all()
    assert unsub_obj.unsubscribed
    assert sub._subscriptions == []


@pytest.mark.asyncio
async def test_subscriber_jetstream_subscribe_and_unsubscribe():
    nc = DummyNATS()
    js = DummyJS()
    sub = Subscriber(nc, js)

    async def handler(msg):
        return None

    await sub.subscribe("topic", handler, use_jetstream=True, durable="d1")
    assert js.subscribed
    entry = js.subscribed[0]
    assert entry[0] == "topic" and entry[2] == "d1"
    unsub_obj = entry[-1]

    await sub.unsubscribe_all()
    assert unsub_obj.unsubscribed
    assert sub._subscriptions == []


@pytest.mark.asyncio
async def test_subscriber_requires_js_context():
    nc = DummyNATS()
    sub = Subscriber(nc, None)

    async def handler(msg):
        return None

    with pytest.raises(ValueError):
        await sub.subscribe("topic", handler, use_jetstream=True, durable="d")


@pytest.mark.asyncio
async def test_subscriber_requires_durable():
    nc = DummyNATS()
    js = DummyJS()
    sub = Subscriber(nc, js)

    async def handler(msg):
        return None

    with pytest.raises(ValueError):
        await sub.subscribe("topic", handler, use_jetstream=True)
