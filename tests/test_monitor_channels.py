import asyncio
import logging
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

    async def send(self, content, *, reference=None, **kwargs):
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

    async def fake_gen():
        return "ping"

    monkeypatch.setattr(sg, "generate_idle_response", fake_gen)
    monkeypatch.setattr(sg, "idle_response_candidates", ["fallback"])  # fallback

    async def fake_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)

    await sg.monitor_channels(bot, 1)

    assert channel.sent_messages == ["ping"]


@pytest.mark.asyncio
async def test_monitor_channels_generator_failure(monkeypatch):
    """Fallback to static list when generation fails."""
    channel = DummyChannel()
    bot = DummyBot(channel)

    async def bad_gen():
        return None

    monkeypatch.setattr(sg, "generate_idle_response", bad_gen)
    monkeypatch.setattr(sg, "idle_response_candidates", ["fallback"])

    async def fake_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)

    await sg.monitor_channels(bot, 1)

    assert channel.sent_messages == ["fallback"]


@pytest.mark.asyncio
async def test_generate_idle_response_env(monkeypatch):
    """``IDLE_GENERATOR_PROMPT`` environment variable is used."""
    captured = {}

    def fake_generator(prompt: str, **kwargs):
        captured["prompt"] = prompt
        return [{"generated_text": "pong"}]

    monkeypatch.setattr(sg, "_get_idle_generator", lambda: fake_generator)
    monkeypatch.setenv("IDLE_GENERATOR_PROMPT", "custom start")

    text = await sg.generate_idle_response()

    assert text == "pong"
    assert captured["prompt"] == "custom start"


@pytest.mark.asyncio
async def test_generate_idle_response_topics(tmp_path, monkeypatch):
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.init_db()

    await sg.store_memory(1, "hi", topic="greet")
    await sg.store_memory(2, "bye", topic="farewell")

    captured = {}

    def fake_generator(prompt: str, **kwargs):
        captured["prompt"] = prompt
        return [{"generated_text": "pong"}]

    monkeypatch.setattr(sg, "_get_idle_generator", lambda: fake_generator)

    text = await sg.generate_idle_response()

    assert text == "pong"
    assert "greet" in captured["prompt"]
    assert "farewell" in captured["prompt"]

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_monitor_channels_only_bots(monkeypatch):
    """Prompt when only bots have chatted for a while."""
    channel = DummyChannel()
    bot = DummyBot(channel)

    from datetime import timedelta
    from types import SimpleNamespace

    from discord.utils import utcnow

    class DummyMessage:
        def __init__(self, created_at, is_bot=True):
            self.created_at = created_at
            self.author = SimpleNamespace(bot=is_bot)

    messages = [
        DummyMessage(utcnow() - timedelta(minutes=1), True),
        DummyMessage(utcnow() - timedelta(minutes=sg.IDLE_TIMEOUT_MINUTES + 1), False),
    ]

    def history_gen(limit=1):
        async def _gen():
            for msg in messages[:limit]:
                yield msg

        return _gen()

    channel.history = history_gen

    f = asyncio.Future()
    f.set_result(({1}, set()))
    monkeypatch.setattr(sg, "who_is_active", lambda channel, limit=20: f)

    monkeypatch.setattr(sg, "BOT_CHAT_ENABLED", True)

    async def fake_gen():
        return "ping"

    monkeypatch.setattr(sg, "generate_idle_response", fake_gen)
    monkeypatch.setattr(sg, "idle_response_candidates", ["fallback"])

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

    async def fake_gen():
        return "ping"

    monkeypatch.setattr(sg, "generate_idle_response", fake_gen)
    monkeypatch.setattr(sg, "idle_response_candidates", ["fallback"])  # fallback

    async def fake_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)

    await sg.monitor_channels(bot, 1)

    assert channel.sent_messages == ["ping"]


class DummyNoChannelBot:
    async def wait_until_ready(self):
        return None

    def get_channel(self, cid):
        return None

    def is_closed(self):
        return True


@pytest.mark.asyncio
async def test_monitor_channels_no_channel(monkeypatch, caplog):
    bot = DummyNoChannelBot()

    with caplog.at_level(logging.ERROR):
        await sg.monitor_channels(bot, 123)

    assert any("does not exist" in r.getMessage() for r in caplog.records)
