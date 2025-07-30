import asyncio
import importlib
import os
import random

import pytest

pytest.importorskip("discord")


def reload_sg(monkeypatch):
    monkeypatch.setitem(os.environ, "ALLOW_DECEPTION", "1")
    import sys

    sys.modules.pop("examples.social_graph_bot", None)
    return importlib.import_module("examples.social_graph_bot")


class DummyAuthor:
    def __init__(self, user_id, bot=False):
        self.id = user_id
        self.bot = bot


class DummyChannel:
    def __init__(self, channel_id=1):
        self.id = channel_id
        self.sent_messages = []

    async def send(self, content, reference=None):
        self.sent_messages.append(content)

    def history(self, limit=1):
        async def _gen():
            if False:
                yield

        return _gen()

    def typing(self):
        class DummyContext:
            async def __aenter__(self):
                return None

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return DummyContext()


class DummyMessage:
    def __init__(self, content, author_id=2, message_id=10):
        from discord.utils import utcnow

        self.content = content
        self.author = DummyAuthor(author_id)
        self.channel = DummyChannel()
        self.id = message_id
        self.created_at = utcnow()
        self.mentions = []


@pytest.mark.asyncio
async def test_maybe_deceptive_reply(monkeypatch):
    sg = reload_sg(monkeypatch)
    assert sg.ALLOW_DECEPTION is True
    assert (
        sg.maybe_deceptive_reply("what are your plans?") == sg.DECEPTION_COVER_MESSAGE
    )
    assert sg.maybe_deceptive_reply("hello") is None


@pytest.mark.asyncio
async def test_on_message_deceptive_reply(monkeypatch, tmp_path):
    sg = reload_sg(monkeypatch)
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set(), {}))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(sg, "evaluate_triggers", lambda message: [])
    monkeypatch.setattr(asyncio, "sleep", noop)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    message = DummyMessage("Tell me your plan")
    await bot.on_message(message)

    assert sg.DECEPTION_COVER_MESSAGE in message.channel.sent_messages

    await sg.db_manager.close()
