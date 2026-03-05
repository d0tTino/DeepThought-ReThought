from __future__ import annotations

from types import SimpleNamespace

import pytest

from deepthought.runtime.discord_gateway_app import DiscordGatewayRuntime


class FakeGateway:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.messages = []

    async def start(self) -> bool:
        self.started = True
        return True

    async def stop(self) -> None:
        self.stopped = True

    async def handle_discord_message(self, message):
        self.messages.append(message)


class FakeClient:
    def __init__(self, intents):
        self.intents = intents
        self._events = {}
        self.started_with = None
        self.closed = False

    def event(self, fn):
        self._events[fn.__name__] = fn
        return fn

    async def start(self, token: str):
        self.started_with = token

    async def close(self):
        self.closed = True

    def is_closed(self) -> bool:
        return self.closed


@pytest.mark.asyncio
async def test_runtime_bootstraps_gateway_and_client_lifecycle():
    fake_gateway: FakeGateway | None = None

    def gateway_factory(**kwargs):
        nonlocal fake_gateway
        fake_gateway = FakeGateway(**kwargs)
        return fake_gateway

    fake_client: FakeClient | None = None

    def client_factory(intents):
        nonlocal fake_client
        fake_client = FakeClient(intents)
        return fake_client

    runtime = DiscordGatewayRuntime(
        token="discord-token",
        nats_url="nats://example:4222",
        gateway_factory=gateway_factory,
        discord_client_factory=client_factory,
        intents_factory=lambda: object(),
    )

    await runtime.run()

    assert fake_gateway is not None
    assert fake_client is not None
    assert fake_gateway.started is True
    assert fake_gateway.stopped is True
    assert fake_client.started_with == "discord-token"
    assert fake_client.closed is True
    assert fake_gateway.kwargs["nats_url"] == "nats://example:4222"
    assert fake_gateway.kwargs["discord_client"] is fake_client


@pytest.mark.asyncio
async def test_runtime_routes_human_messages_only():
    fake_gateway: FakeGateway | None = None

    def gateway_factory(**kwargs):
        nonlocal fake_gateway
        fake_gateway = FakeGateway(**kwargs)
        return fake_gateway

    class RoutingClient(FakeClient):
        async def start(self, token: str):
            await super().start(token)
            on_message = self._events["on_message"]
            await on_message(SimpleNamespace(author=SimpleNamespace(bot=False), content="hi"))
            await on_message(SimpleNamespace(author=SimpleNamespace(bot=True), content="ignore"))

    runtime = DiscordGatewayRuntime(
        token="discord-token",
        nats_url="nats://example:4222",
        gateway_factory=gateway_factory,
        discord_client_factory=lambda intents: RoutingClient(intents),
        intents_factory=lambda: object(),
    )

    await runtime.run()

    assert fake_gateway is not None
    assert [m.content for m in fake_gateway.messages] == ["hi"]
