import json
from types import SimpleNamespace

import pytest

from deepthought.eda.events import EventSubjects, ResponseRankedPayload
from deepthought.services.discord_gateway_service import DiscordGatewayService


class DummyNATS:
    is_connected = True


class DummyJS:
    pass


class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload, use_jetstream, timeout))


class DummySubscriber:
    def __init__(self, *args, **kwargs):
        pass

    async def subscribe(self, **kwargs):
        return True

    async def unsubscribe_all(self):
        return None


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.nacked = True


class DummyChannel:
    def __init__(self):
        self.messages = []

    async def send(self, content: str):
        self.messages.append(content)


class DummyDiscordClient:
    def __init__(self):
        self.channel = DummyChannel()

    def get_channel(self, channel_id: int):
        return self.channel if channel_id == 123 else None


@pytest.fixture
def service(monkeypatch):
    import deepthought.services.base as base_mod

    monkeypatch.setattr(base_mod, "Publisher", DummyPublisher)
    monkeypatch.setattr(base_mod, "Subscriber", DummySubscriber)
    return DiscordGatewayService(DummyNATS(), DummyJS(), discord_client=DummyDiscordClient())


@pytest.mark.asyncio
async def test_handle_discord_message_publishes_input_received(service):
    message = SimpleNamespace(
        content="hello",
        id=99,
        author=SimpleNamespace(id=5, name="alice", display_name="Alice", bot=False),
        channel=SimpleNamespace(id=123),
        guild=SimpleNamespace(id=7),
    )

    input_id = await service.handle_discord_message(message)

    assert input_id is not None
    subject, payload, use_js, _timeout = service._publisher.published[0]
    assert subject == EventSubjects.INPUT_RECEIVED
    assert use_js is True
    assert payload.user_input == "hello"
    assert payload.channel_id == "123"


@pytest.mark.asyncio
async def test_handle_discord_message_ignores_bots(service):
    message = SimpleNamespace(content="ignore", author=SimpleNamespace(bot=True))
    assert await service.handle_discord_message(message) is None
    assert service._publisher.published == []


@pytest.mark.asyncio
async def test_ranked_response_sent_and_acked(service):
    payload = ResponseRankedPayload(final_response="done", input_id="in-1", channel_id="123")
    msg = DummyMsg(payload.to_json())

    await service._handle_ranked_response(msg)

    assert msg.acked
    assert not msg.nacked
    assert service._discord_client.channel.messages == ["done"]


@pytest.mark.asyncio
async def test_ranked_response_invalid_payload_naks(service):
    msg = DummyMsg(json.dumps(["bad"]))

    await service._handle_ranked_response(msg)

    assert msg.nacked
