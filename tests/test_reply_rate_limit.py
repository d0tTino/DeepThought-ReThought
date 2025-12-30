import asyncio
import importlib
import os
import random

import pytest

pytest.importorskip("discord")
pytest.importorskip("nats")


def reload_sg(monkeypatch):
    monkeypatch.setitem(os.environ, "USER_REPLY_RATE_SECONDS", "1")
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
        self._state = None

    async def reply(self, content, mention_author=True):
        self.replies.append({"content": content, "mention_author": mention_author})
        await self.channel.send(content, reference=self)


@pytest.mark.asyncio
async def test_on_message_rate_limit(monkeypatch, tmp_path):
    sg = reload_sg(monkeypatch)
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(sg, "_ensure_nats", noop)

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
    msg1 = DummyMessage("hi")
    await bot.on_message(msg1)

    msg2 = DummyMessage("hi again", message_id=11)
    await bot.on_message(msg2)

    assert len(msg1.channel.sent_messages) == 1
    assert msg1.replies[0]["mention_author"] is True
    assert msg2.channel.sent_messages == []

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_on_message_no_mention_channels(monkeypatch, tmp_path):
    monkeypatch.setenv("NO_MENTION_CHANNEL_IDS", "1")
    sg = reload_sg(monkeypatch)
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(sg, "_ensure_nats", noop)

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
    msg = DummyMessage("hello")
    await bot.on_message(msg)

    assert len(msg.channel.sent_messages) == 1
    assert msg.channel.sent_messages[0] == msg.replies[0]["content"]
    assert msg.replies[0]["mention_author"] is False

    await sg.db_manager.close()
