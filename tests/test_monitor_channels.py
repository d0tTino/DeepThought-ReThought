import asyncio
import random

import pytest

import examples.social_graph_bot as sg


class DummyChannel:
    def __init__(self):
        self.sent_messages = []

    def history(self, limit=1):
        async def _gen():
            if False:
                yield  # pragma: no cover

        return _gen()

    async def send(self, content):
        self.sent_messages.append(content)

    def typing(self):
        class DummyContext:
            async def __aenter__(self):
                return None

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return DummyContext()


class DummyBot:
    def __init__(self, channel):
        self._closed = False
        self._channel = channel

    async def wait_until_ready(self):
        return None

    def get_channel(self, cid):
        return self._channel

    def is_closed(self):
        if self._closed:
            return True
        self._closed = True
        return False


@pytest.mark.asyncio
async def test_monitor_channels_idle_prompt(monkeypatch):
    channel = DummyChannel()
    bot = DummyBot(channel)

    monkeypatch.setattr(sg, "idle_response_candidates", ["ping"])  # deterministic prompt

    async def fake_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)

    await sg.monitor_channels(bot, 1)

    assert channel.sent_messages == ["ping"]


@pytest.mark.asyncio
async def test_monitor_channels_idle_prompt_old_message(monkeypatch):
    """Send a prompt when the last message is older than the idle timeout."""
    channel = DummyChannel()
    bot = DummyBot(channel)

    from datetime import timedelta

    from discord.utils import utcnow

    class DummyMessage:
        def __init__(self, created_at):
            self.created_at = created_at

    async def history_gen():
        yield DummyMessage(utcnow() - timedelta(minutes=sg.IDLE_TIMEOUT_MINUTES + 1))

    channel.history = lambda limit=1: history_gen()

    monkeypatch.setattr(sg, "idle_response_candidates", ["ping"])  # deterministic prompt

    async def fake_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)

    await sg.monitor_channels(bot, 1)

    assert channel.sent_messages == ["ping"]
