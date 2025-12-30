import asyncio
import importlib
import os
import random

import pytest

pytest.importorskip("discord")
pytest.importorskip("nats")

from deepthought.services import DBManager


def reload_sg(monkeypatch, threshold="0", prob="0"):
    monkeypatch.setitem(os.environ, "MINIMAL_REPLY_THRESHOLD", threshold)
    monkeypatch.setitem(os.environ, "MINIMAL_REPLY_PROB", prob)
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
        self.replies = []

    async def reply(self, content, mention_author=True):
        self.replies.append({"content": content, "mention_author": mention_author})
        await self.channel.send(content, reference=self)


@pytest.mark.asyncio
async def test_minimal_reply_by_trust(tmp_path, monkeypatch):
    sg = reload_sg(monkeypatch, threshold="1", prob="0")
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    async def noop(*args, **kwargs):
        return None

    fut = asyncio.Future()
    fut.set_result((set(), set(), {}))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: fut)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(sg, "evaluate_triggers", lambda message: [])
    monkeypatch.setattr(asyncio, "sleep", noop)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)
    monkeypatch.setattr(random, "random", lambda: 0.5)

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    message = DummyMessage("hi")
    await bot.on_message(message)

    assert message.channel.sent_messages == [sg.MINIMAL_REPLIES[0]]
    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_minimal_reply_by_probability(tmp_path, monkeypatch):
    sg = reload_sg(monkeypatch, threshold="-10", prob="1")
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    async def noop(*args, **kwargs):
        return None

    fut = asyncio.Future()
    fut.set_result((set(), set(), {}))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: fut)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(sg, "evaluate_triggers", lambda message: [])
    monkeypatch.setattr(asyncio, "sleep", noop)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)
    monkeypatch.setattr(random, "random", lambda: 0)

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    message = DummyMessage("hi")
    await bot.on_message(message)

    assert message.channel.sent_messages == [sg.MINIMAL_REPLIES[0]]
    await sg.db_manager.close()
