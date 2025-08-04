import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import examples.social_graph_bot as sg


class DummyChannel:
    def __init__(self):
        self.sent = []

    async def send(self, content, *, reference=None, **kwargs):
        self.sent.append(content)

    def typing(self):
        class DummyCtx:
            async def __aenter__(self):
                return None

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return DummyCtx()


@pytest.mark.asyncio
async def test_multi_bot_handshake(monkeypatch):
    channel = DummyChannel()

    sg.BOT_CHAT_ENABLED = True
    sg.bot_handshakes.clear()
    sg.bot_reply_times.clear()
    sg.last_bot_reply_time = None

    now = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def fake_now():
        return now

    monkeypatch.setattr(sg.discord.utils, "utcnow", fake_now)

    async def fake_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(sg.random, "uniform", lambda a, b: 0)

    author1 = SimpleNamespace(id=1, bot=True)
    author2 = SimpleNamespace(id=2, bot=True)

    msg = SimpleNamespace(channel=channel)

    msg.author = author1
    msg.content = sg.HANDSHAKE_MESSAGE
    assert await sg.handle_bot_handshake(msg) is False

    msg.author = author2
    msg.content = sg.HANDSHAKE_MESSAGE
    assert await sg.handle_bot_handshake(msg) is False

    assert channel.sent == [sg.HANDSHAKE_MESSAGE, sg.HANDSHAKE_MESSAGE]

    msg.author = author1
    msg.content = "hello"
    assert await sg.handle_bot_handshake(msg) is False

    now += timedelta(seconds=sg.BOT_COOLDOWN_SECONDS + 1)

    assert await sg.handle_bot_handshake(msg) is True
