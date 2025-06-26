import json

import pytest

from deepthought.motivate import reward_manager as rm_mod


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


@pytest.mark.asyncio
async def test_start_listening_subscribes():
    sub = DummySubscriber()
    mgr = rm_mod.RewardManager(
        sub, DummyLedger(), DummyPublisher(), "tok", model=DummyModel()
    )
    result = await mgr.start_listening()
    assert result is True
    assert sub.calls
    args, kwargs = sub.calls[0]
    assert kwargs["subject"] == "chat.bot"


@pytest.mark.asyncio
async def test_handle_chat_event_publishes(monkeypatch):
    sub = DummySubscriber()
    ledger = DummyLedger()
    pub = DummyPublisher()
    mgr = rm_mod.RewardManager(sub, ledger, pub, "tok", model=DummyModel())

    monkeypatch.setattr(mgr, "_score_novelty", lambda _t: 1.0)

    async def _fake_social(*_a):
        return 2

    monkeypatch.setattr(mgr, "_score_social", _fake_social)

    msg = DummyMsg({"prompt": "p", "response": "r"})
    await mgr._handle_chat_event(msg)

    assert msg.acked
    assert ledger.events[0][2] == pytest.approx(2.0)
    assert pub.published[0][0] == "agent.reward"
    assert pub.published[0][1]["reward"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_score_social_returns_zero_on_exception(monkeypatch):
    """_score_social should return 0 when the HTTP request fails."""
    mgr = rm_mod.RewardManager(
        DummySubscriber(), DummyLedger(), DummyPublisher(), "tok", model=DummyModel()
    )

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(rm_mod.aiohttp, "ClientSession", lambda: DummySession())

    result = await mgr._score_social(1, 2)
    assert result == 0


@pytest.mark.asyncio
async def test_score_social_returns_zero_on_500(monkeypatch):
    """_score_social should return 0 when the HTTP status is not 200."""
    mgr = rm_mod.RewardManager(
        DummySubscriber(), DummyLedger(), DummyPublisher(), "tok", model=DummyModel()
    )

    class DummyResp:
        def __init__(self, status):
            self.status = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def json(self):
            return {}

    class DummySession:
        def __init__(self, resp):
            self.resp = resp

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, *args, **kwargs):
            return self.resp

    monkeypatch.setattr(
        rm_mod.aiohttp, "ClientSession", lambda: DummySession(DummyResp(500))
    )

    result = await mgr._score_social(1, 2)
    assert result == 0
