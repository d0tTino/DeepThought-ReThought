import json
from types import SimpleNamespace

import pytest

from deepthought.eda.contracts import EventEnvelope
from deepthought.eda.events import EventSubjects, ResponseRankedPayload
from deepthought.services.discord_gateway_service import DiscordGatewayService


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    async def sleep(self, seconds: float):
        self.now += max(0.0, seconds)


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
        self.ack_calls = 0
        self.nak_calls = 0

    async def ack(self):
        self.acked = True
        self.ack_calls += 1

    async def nak(self):
        self.nacked = True
        self.nak_calls += 1


class DummyTypingScope:
    def __init__(self, channel):
        self._channel = channel

    async def __aenter__(self):
        self._channel.typing_entries += 1

    async def __aexit__(self, exc_type, exc, tb):
        return None


class DummyChannel:
    def __init__(self):
        self.messages = []
        self.typing_entries = 0

    def typing(self):
        return DummyTypingScope(self)

    async def send(self, content: str, **kwargs):
        self.messages.append((content, kwargs))


class DummyDiscordClient:
    def __init__(self):
        self.channel = DummyChannel()
        self.thread = DummyChannel()

    def get_channel(self, channel_id: int):
        if channel_id == 123:
            return self.channel
        if channel_id == 999:
            return self.thread
        return None


@pytest.fixture

def fake_clock():
    return FakeClock()


@pytest.fixture

def service(monkeypatch, fake_clock):
    import deepthought.services.base as base_mod

    monkeypatch.setattr(base_mod, "Publisher", DummyPublisher)
    monkeypatch.setattr(base_mod, "Subscriber", DummySubscriber)
    return DiscordGatewayService(
        DummyNATS(),
        DummyJS(),
        discord_client=DummyDiscordClient(),
        clock=fake_clock,
        sleeper=fake_clock.sleep,
    )


@pytest.mark.asyncio
async def test_handle_discord_message_publishes_input_received(service):
    message = SimpleNamespace(
        content="hello",
        id=99,
        reference=SimpleNamespace(message_id=66),
        thread=SimpleNamespace(id=999),
        author=SimpleNamespace(id=5, name="alice", display_name="Alice", bot=False),
        channel=SimpleNamespace(id=123),
        guild=SimpleNamespace(id=7),
        attachments=[
            SimpleNamespace(
                url="https://cdn.discordapp.com/file.png",
                content_type="image/png",
                filename="file.png",
                size=512,
            )
        ],
    )

    input_id = await service.handle_discord_message(message)

    assert input_id is not None
    subject, payload, use_js, _timeout = service._publisher.published[0]
    assert subject == EventSubjects.INPUT_RECEIVED
    assert use_js is True
    assert payload["payload"]["user_input"] == "hello"
    assert payload["payload"]["channel_id"] == "123"
    assert payload["payload"]["reference_message_id"] == "66"
    assert payload["payload"]["thread_id"] == "999"
    assert payload["payload"]["attachments"] is not None
    assert payload["payload"]["attachments"][0]["url"] == "https://cdn.discordapp.com/file.png"


@pytest.mark.asyncio
async def test_handle_discord_message_ignores_bot_authored_messages(service):
    message = SimpleNamespace(
        content="beep boop",
        id=100,
        author=SimpleNamespace(id=55, name="robot", display_name="Robot", bot=True),
        channel=SimpleNamespace(id=123),
        guild=SimpleNamespace(id=7),
    )

    input_id = await service.handle_discord_message(message)

    assert input_id is None
    assert service._publisher.published == []
    assert service._pending_routes == {}


@pytest.mark.asyncio
async def test_ranked_response_respects_thread_reply_and_policy_override(service, fake_clock):
    payload = ResponseRankedPayload(
        final_response="done",
        input_id="in-1",
        channel_id="123",
        thread_id="999",
        reply_to_message_id="orig-7",
        interaction_policy={"delay_seconds": 0.4, "typing_seconds": 0.2, "cooldown_seconds": 0.5},
    )
    msg = DummyMsg(payload.to_json())

    await service._handle_ranked_response(msg)

    assert msg.acked
    assert not msg.nacked
    assert fake_clock.now == pytest.approx(100.6, abs=0.001)
    assert service._discord_client.thread.messages == [("done", {"reference": "orig-7"})]


@pytest.mark.asyncio
async def test_ranked_response_applies_cooldown_per_channel(service, fake_clock):
    first = DummyMsg(
        ResponseRankedPayload(
            final_response="first",
            channel_id="123",
            interaction_policy={"delay_seconds": 0.0, "typing_seconds": 0.0, "cooldown_seconds": 1.0},
        ).to_json()
    )
    second = DummyMsg(
        ResponseRankedPayload(
            final_response="second",
            channel_id="123",
            interaction_policy={"delay_seconds": 0.0, "typing_seconds": 0.0, "cooldown_seconds": 1.0},
        ).to_json()
    )

    await service._handle_ranked_response(first)
    before_second = fake_clock.now
    await service._handle_ranked_response(second)

    assert second.acked
    assert fake_clock.now - before_second >= 1.0


@pytest.mark.asyncio
async def test_ranked_response_invalid_payload_naks(service):
    msg = DummyMsg(json.dumps(["bad"]))

    await service._handle_ranked_response(msg)

    assert msg.nacked
    assert msg.nak_calls == 1
    assert msg.ack_calls == 0


@pytest.mark.asyncio
async def test_send_channel_message_invalid_channel_or_thread_id_returns_true(service):
    sent_channel = await service._send_channel_message(
        "not-a-channel-id",
        content="ignored",
        reply_to_message_id=None,
        thread_id=None,
        author_id=None,
        interaction_metadata=None,
    )
    sent_thread = await service._send_channel_message(
        "123",
        content="ignored",
        reply_to_message_id=None,
        thread_id="not-a-thread-id",
        author_id=None,
        interaction_metadata=None,
    )

    assert sent_channel is True
    assert sent_thread is True
    assert service._discord_client.channel.messages == []
    assert service._discord_client.thread.messages == []


@pytest.mark.asyncio
async def test_send_channel_message_missing_channel_lookup_returns_true(service):
    sent = await service._send_channel_message(
        "321",
        content="ignored",
        reply_to_message_id=None,
        thread_id=None,
        author_id=None,
        interaction_metadata=None,
    )

    assert sent is True
    assert service._discord_client.channel.messages == []


@pytest.mark.asyncio
async def test_send_channel_message_applies_typing_delay_and_cooldown(service, fake_clock):
    sent = await service._send_channel_message(
        "123",
        content="hello",
        reply_to_message_id="orig-1",
        thread_id=None,
        author_id="42",
        interaction_metadata={"delay_seconds": 0.4, "typing_seconds": 0.2, "cooldown_seconds": 0.5},
    )

    assert sent is True
    assert fake_clock.now == pytest.approx(100.6, abs=0.001)
    assert service._discord_client.channel.typing_entries == 1
    assert service._discord_client.channel.messages == [("hello", {"reference": "orig-1"})]
    assert service._cooldown_until["channel:123"] == pytest.approx(101.1, abs=0.001)
    assert service._cooldown_until["user:42"] == pytest.approx(101.1, abs=0.001)


@pytest.mark.asyncio
async def test_ranked_response_missing_channel_mapping_acks_once(service):
    msg = DummyMsg(ResponseRankedPayload(final_response="done", input_id="unknown").to_json())

    await service._handle_ranked_response(msg)

    assert msg.acked is True
    assert msg.nacked is False
    assert msg.ack_calls == 1
    assert msg.nak_calls == 0


@pytest.mark.asyncio
async def test_ranked_response_accepts_enveloped_payload(service):
    payload = ResponseRankedPayload(final_response="wrapped", channel_id="123", input_id="in-wrap")
    envelope = EventEnvelope.build(
        subject=EventSubjects.RESPONSE_RANKED,
        payload=json.loads(payload.to_json()),
        producer="selector",
    )
    msg = DummyMsg(json.dumps(envelope.__dict__))

    await service._handle_ranked_response(msg)

    assert msg.acked
    assert service._discord_client.channel.messages[-1][0] == "wrapped"
