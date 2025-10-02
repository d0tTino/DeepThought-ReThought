import asyncio
import json
from types import SimpleNamespace

import pytest

from src.deepthought.edge.discord_gateway import DiscordGateway, DiscordGatewayConfig


class FakePublisher:
    def __init__(self) -> None:
        self.calls = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.calls.append({
            "subject": subject,
            "payload": payload,
            "use_jetstream": use_jetstream,
            "timeout": timeout,
        })


class FakeSubscriber:
    def __init__(self) -> None:
        self.subscriptions = []
        self.unsubscribed = False

    async def subscribe(self, subject, handler, queue="", use_jetstream=False, durable=""):
        self.subscriptions.append({
            "subject": subject,
            "handler": handler,
            "queue": queue,
            "use_jetstream": use_jetstream,
            "durable": durable,
        })
        self.handler = handler

    async def unsubscribe_all(self):
        self.unsubscribed = True


class FakeChannel:
    def __init__(self, channel_id):
        self.id = channel_id
        self.sent_messages = []

    async def send(self, content):
        self.sent_messages.append(content)


class FakeDiscordClient:
    def __init__(self):
        self._events = {}
        self._channels = {}
        self.closed = False

    def event(self, coro):
        self._events[coro.__name__] = coro
        return coro

    def get_channel(self, channel_id):
        return self._channels.get(channel_id)

    def add_channel(self, channel):
        self._channels[channel.id] = channel

    async def start(self, token):  # pragma: no cover - not exercised in unit tests
        self.started_token = token

    async def close(self):
        self.closed = True


class FakeMsg:
    def __init__(self, data):
        self.data = data
        self.acked = False

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_gateway_publishes_discord_messages():
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    client = FakeDiscordClient()
    gateway = DiscordGateway(publisher=publisher, subscriber=subscriber, client=client)

    await gateway.setup()

    message_handler = client._events["on_message"]
    message = SimpleNamespace(
        id="m-1",
        content="Hello DeepThought",
        author=SimpleNamespace(id="user-123", name="Tester", bot=False),
        channel=SimpleNamespace(id=999),
    )

    await message_handler(message)

    assert len(publisher.calls) == 1
    call = publisher.calls[0]
    assert call["subject"] == DiscordGatewayConfig().incoming_subject
    assert call["payload"]["content"] == "Hello DeepThought"
    assert call["payload"]["channel_id"] == 999
    assert call["use_jetstream"] is True

    assert len(subscriber.subscriptions) == 1
    sub = subscriber.subscriptions[0]
    assert sub["durable"] == DiscordGatewayConfig().durable_name
    assert sub["use_jetstream"] is True


@pytest.mark.asyncio
async def test_gateway_ignores_bot_messages():
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    client = FakeDiscordClient()
    gateway = DiscordGateway(publisher=publisher, subscriber=subscriber, client=client)

    await gateway.setup()

    message_handler = client._events["on_message"]
    bot_message = SimpleNamespace(
        id="m-2",
        content="Bot message",
        author=SimpleNamespace(id="bot-1", name="Bot", bot=True),
        channel=SimpleNamespace(id=1),
    )

    await message_handler(bot_message)

    assert publisher.calls == []


@pytest.mark.asyncio
async def test_ranked_responses_post_to_discord():
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    client = FakeDiscordClient()
    channel = FakeChannel(77)
    client.add_channel(channel)
    gateway = DiscordGateway(publisher=publisher, subscriber=subscriber, client=client)

    await gateway.setup()

    sub = subscriber.subscriptions[0]
    handler = sub["handler"]
    payload = {
        "channel_id": 77,
        "content": "Hello from the bus",
    }
    msg = FakeMsg(json.dumps(payload).encode())

    await handler(msg)

    assert channel.sent_messages == ["Hello from the bus"]
    assert msg.acked is True

    await gateway.stop()
    assert subscriber.unsubscribed is True
    assert client.closed is True
