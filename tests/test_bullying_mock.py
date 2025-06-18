import asyncio
import random

import pytest

import examples.social_graph_bot as sg


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
async def test_bullying_triggers_sarcasm(tmp_path, monkeypatch):
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.init_db()

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set()))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(sg, "evaluate_triggers", lambda message: [])
    monkeypatch.setattr(asyncio, "sleep", noop)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)

    async def allow_mock(user_id):
        return False

    monkeypatch.setattr(sg, "is_do_not_mock", allow_mock)

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    assert bot.intents.members
    assert bot.intents.presences

    message = DummyMessage("You are an idiot")
    await bot.on_message(message)

    assert "Oh, how original." in message.channel.sent_messages
    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_do_not_mock_blocks_sarcasm(tmp_path, monkeypatch):
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.init_db()

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set()))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(sg, "evaluate_triggers", lambda message: [])
    monkeypatch.setattr(asyncio, "sleep", noop)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)

    async def prevent_mock(user_id):
        return True

    monkeypatch.setattr(sg, "is_do_not_mock", prevent_mock)

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    assert bot.intents.members
    assert bot.intents.presences

    message = DummyMessage("You are an idiot")
    await bot.on_message(message)

    assert "Oh, how original." not in message.channel.sent_messages
    await sg.db_manager.close()
