import asyncio
import random

import pytest

pytest.importorskip("discord")
pytest.importorskip("nats")

sg = pytest.importorskip("examples.social_graph_bot")
if not hasattr(sg, "TrustService"):
    pytest.skip("social_graph_bot optional dependencies not installed", allow_module_level=True)
from deepthought.services import DBManager


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
async def test_refuses_low_trust(tmp_path, monkeypatch, input_events):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    sg.trust_service = sg.TrustService(sg.db_manager)
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

    await sg.trust_service.adjust_trust(2, -10)

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    message = DummyMessage("hi")
    await bot.on_message(message)

    assert message.channel.sent_messages == []
    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_minimal_reply_low_trust(tmp_path, monkeypatch, input_events):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    sg.trust_service = sg.TrustService(sg.db_manager)
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

    sg.MINIMAL_REPLY_THRESHOLD = 1
    sg.MINIMAL_REPLY_PROB = 0

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    message = DummyMessage("hello")
    await bot.on_message(message)

    assert message.channel.sent_messages == [sg.MINIMAL_REPLIES[0]]
    await sg.db_manager.close()
