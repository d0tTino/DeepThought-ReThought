import json

import pytest

rm_mod = pytest.importorskip("deepthought.motivate.reward_manager")


class DummySubscriber:
    def __init__(self):
        self.calls = []

    async def subscribe(self, *args, **kwargs):
        self.calls.append((args, kwargs))

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
        return None


class DummyModel:
    def encode(self, text, convert_to_numpy=True):
        return [0.0]


class DummyMsg:
    def __init__(self, payload):
        self.data = json.dumps(payload).encode()
        self.acked = False

    async def ack(self):
        self.acked = True


class DummyResp:
    def __init__(self, status=200, reactions=None):
        self.status = status
        self._reactions = reactions or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return {"reactions": self._reactions}


class DummySession:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, *args, **kwargs):
        return self._resp


@pytest.mark.asyncio
async def test_score_social_counts_reactions(monkeypatch):
    mgr = rm_mod.RewardManager(DummySubscriber(), DummyLedger(), DummyPublisher(), "tok", model=DummyModel())
    resp = DummyResp(reactions=[{"count": 2}, {"count": 1}])
    monkeypatch.setattr(rm_mod.aiohttp, "ClientSession", lambda: DummySession(resp))

    result = await mgr._score_social(123, 456)
    assert result == 3


@pytest.mark.asyncio
async def test_handle_chat_event_publishes_reward(monkeypatch):
    sub = DummySubscriber()
    ledger = DummyLedger()
    pub = DummyPublisher()
    mgr = rm_mod.RewardManager(sub, ledger, pub, "tok", model=DummyModel())

    resp = DummyResp(reactions=[{"count": 3}])
    monkeypatch.setattr(rm_mod.aiohttp, "ClientSession", lambda: DummySession(resp))

    msg = DummyMsg({"prompt": "p", "response": "r", "channel_id": 1, "message_id": 2})
    await mgr._handle_chat_event(msg)

    assert msg.acked
    assert ledger.events
    assert pub.published
    reward = ledger.events[0][2]
    assert pytest.approx(reward) == 2.0
    assert pub.published[0][0] == "agent.reward"
    assert pytest.approx(pub.published[0][1]["reward"]) == 2.0
