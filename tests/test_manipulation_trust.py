import asyncio
import random

import pytest

pytest.importorskip("discord")
pytest.importorskip("aiosqlite")
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


@pytest.mark.asyncio
async def test_manipulative_message_adjusts_trust(tmp_path, monkeypatch, input_events):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    sg.trust_service = sg.TrustService(sg.db_manager)
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
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(sg, "detect_emotions", lambda _t: {})
    monkeypatch.setattr(
        sg,
        "analyze_social",
        lambda _t: {"flirtation": 0, "avoidance": 0, "manipulation": 0},
    )

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    message = DummyMessage("After all I've done for you")
    await bot.on_message(message)

    trust = await sg.db_manager.get_trust(message.author.id)
    assert trust < 0
    assert message.channel.sent_messages == []
    await sg.db_manager.close()
