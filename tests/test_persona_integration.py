import asyncio
import random

import pytest

pytest.importorskip("discord")

sg = pytest.importorskip("examples.social_graph_bot")
if not hasattr(sg, "TrustService"):
    pytest.skip("social_graph_bot optional dependencies not installed", allow_module_level=True)

pytest.importorskip("nats")
from deepthought.services import DBManager, PersonaManager


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
                yield  # pragma: no cover

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
        # Avoid triggering time-based theories by using a fixed hour
        self.created_at = utcnow().replace(hour=1)
        self.mentions = []


@pytest.mark.asyncio
async def test_on_message_persona_changes_with_affinity(tmp_path, monkeypatch, input_events):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    sg.trust_service = sg.TrustService(sg.db_manager)

    # Use lower thresholds for easier testing
    sg.persona_manager = PersonaManager(sg.db_manager, friendly=3, playful=1)

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set(), {}))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(asyncio, "sleep", noop)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)
    monkeypatch.setattr(sg.reply_limiter, "allow", lambda _id: True)
    monkeypatch.setattr(sg, "detect_emotions", lambda _t: {})
    monkeypatch.setattr(
        sg,
        "analyze_social",
        lambda _t: {"flirtation": 0, "avoidance": 0, "manipulation": 0},
    )

    bot = sg.SocialGraphBot(monitor_channel_id=1)

    msg1 = DummyMessage("hi")
    await bot.on_message(msg1)
    assert msg1.channel.sent_messages[-1] == sg.PERSONA_REPLIES["snarky"][0]

    await sg.adjust_affinity(msg1.author.id, 1)
    msg2 = DummyMessage("hi again", author_id=msg1.author.id, message_id=11)
    await bot.on_message(msg2)
    assert msg2.channel.sent_messages[-1] == sg.PERSONA_REPLIES["playful"][0]

    await sg.db_manager.adjust_trust(msg1.author.id, 2)
    msg3 = DummyMessage("hello friend", author_id=msg1.author.id, message_id=12)
    await bot.on_message(msg3)
    assert msg3.channel.sent_messages[-1] == sg.PERSONA_REPLIES["friendly"][0]

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_social_analysis_adjusts_tone(tmp_path, monkeypatch, input_events):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    sg.trust_service = sg.TrustService(sg.db_manager)

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set(), {}))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(asyncio, "sleep", noop)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)
    monkeypatch.setattr(sg.reply_limiter, "allow", lambda _id: True)
    monkeypatch.setattr(sg, "detect_emotions", lambda _t: {})

    bot = sg.SocialGraphBot(monitor_channel_id=1)

    monkeypatch.setattr(
        sg,
        "analyze_social",
        lambda _t: {"flirtation": 0.9, "avoidance": 0.05, "manipulation": 0.05},
    )
    msg1 = DummyMessage("hey there")
    await bot.on_message(msg1)
    assert msg1.channel.sent_messages[-1] == sg.PERSONA_REPLIES["playful"][0]

    monkeypatch.setattr(
        sg,
        "analyze_social",
        lambda _t: {"flirtation": 0.1, "avoidance": 0.8, "manipulation": 0.1},
    )
    msg2 = DummyMessage("leave me alone", author_id=msg1.author.id, message_id=11)
    await bot.on_message(msg2)
    assert msg2.channel.sent_messages[-1] == sg.AVOIDANCE_REPLY

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_on_message_updates_affinity_from_sentiment(tmp_path, monkeypatch, input_events):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    sg.trust_service = sg.TrustService(sg.db_manager)
    sg.persona_manager = PersonaManager(sg.db_manager)

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set(), {}))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(sg, "log_interaction", noop)
    monkeypatch.setattr(asyncio, "sleep", noop)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)
    monkeypatch.setattr(sg.reply_limiter, "allow", lambda _id: True)
    monkeypatch.setattr(sg, "detect_emotions", lambda _t: {})
    monkeypatch.setattr(
        sg,
        "analyze_social",
        lambda _t: {"flirtation": 0, "avoidance": 0, "manipulation": 0},
    )

    bot = sg.SocialGraphBot(monitor_channel_id=1)

    monkeypatch.setattr(sg, "analyze_sentiment", lambda _t: 0.8)
    msg1 = DummyMessage("you are great")
    await bot.on_message(msg1)
    assert await sg.get_affinity(msg1.author.id) == sg.AFFINITY_POS_DELTA

    monkeypatch.setattr(sg, "analyze_sentiment", lambda _t: -0.8)
    msg2 = DummyMessage("you are terrible", author_id=msg1.author.id, message_id=11)
    await bot.on_message(msg2)
    assert await sg.get_affinity(msg2.author.id) == sg.AFFINITY_POS_DELTA + sg.AFFINITY_NEG_DELTA

    await sg.db_manager.close()
