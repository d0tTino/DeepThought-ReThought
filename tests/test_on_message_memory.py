import asyncio
import json
import logging
import random

import pytest

pytest.importorskip("discord")
pytest.importorskip("aiosqlite")
pytest.importorskip("nats")
import aiosqlite

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
        self.created_at = utcnow()
        self.mentions = []
        self.replies = []

    async def reply(self, content, mention_author=True):
        self.replies.append({"content": content, "mention_author": mention_author})
        await self.channel.send(content, reference=self)


@pytest.mark.asyncio
async def test_on_message_stores_memory(tmp_path, monkeypatch, input_events):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()
    sg.reply_limiter.clear("2")

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set(), {}))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(asyncio, "sleep", noop)

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    assert bot.intents.members
    assert bot.intents.presences

    message = DummyMessage("hello world")
    await bot.on_message(message)

    async with aiosqlite.connect(str(tmp_path / "sg.db")) as db:
        async with db.execute(
            "SELECT memory, sentiment_score FROM memories WHERE user_id=?",
            (str(message.author.id),),
        ) as cur:
            rows = await cur.fetchall()

    assert rows, "Memory row should be inserted"
    assert len(rows) == 1, "Only one memory row should be created"
    stored_memory, score = rows[0]
    assert stored_memory == message.content
    assert isinstance(score, float)
    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_on_message_calls_send_to_prism(tmp_path, monkeypatch, input_events):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()
    sg.reply_limiter.clear("2")

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set(), {}))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(asyncio, "sleep", noop)

    prism_calls = []

    async def fake_send(data):
        prism_calls.append(data)

    monkeypatch.setattr(sg, "send_to_prism", fake_send)

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    assert bot.intents.members
    assert bot.intents.presences

    message = DummyMessage("send prism")
    await bot.on_message(message)

    assert len(prism_calls) == 1
    assert prism_calls[0]["content"] == "send prism"
    assert input_events == ["send prism"]
    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_update_sentiment_trend(tmp_path):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    await sg.update_sentiment_trend("u1", "c1", 0.2)
    await sg.update_sentiment_trend("u1", "c1", -0.1)

    row = await sg.get_sentiment_trend("u1", "c1")
    assert row == (0.1, 2)
    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_update_sentiment_trend_validation(tmp_path):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    with pytest.raises(ValueError):
        await sg.update_sentiment_trend("u1", "c1", "bad")

    with pytest.raises(ValueError):
        await sg.update_sentiment_trend("u1", "c1", 1.5)

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_on_message_updates_sentiment_trend(tmp_path, monkeypatch, input_events):

    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
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
    monkeypatch.setattr(asyncio, "sleep", noop)

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    assert bot.intents.members
    assert bot.intents.presences

    message = DummyMessage("hello again")
    await bot.on_message(message)

    trend = await sg.get_sentiment_trend(message.author.id, message.channel.id)
    expected = sg.analyze_sentiment(message.content)
    assert trend == (expected, 1)
    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_on_message_waits_for_other_bot(tmp_path, monkeypatch):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    async def noop(*args, **kwargs):
        return None

    from datetime import timedelta

    from discord.utils import utcnow

    f = asyncio.Future()
    f.set_result(({123}, set(), {123: utcnow() - timedelta(minutes=1)}))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(asyncio, "sleep", noop)

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    message = DummyMessage("hi there")
    await bot.on_message(message)

    assert message.channel.sent_messages == []
    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_publish_input_received_warns_when_no_publisher(monkeypatch, caplog):
    """publish_input_received should warn if NATS is unavailable."""
    sg._input_publisher = None

    async def fake_ensure():
        sg._input_publisher = None

    monkeypatch.setattr(sg, "_ensure_nats", fake_ensure)

    with caplog.at_level(logging.WARNING):
        await sg.publish_input_received("hello")

    assert any(
        "Dropping INPUT_RECEIVED event because NATS publisher is unavailable" in r.getMessage() for r in caplog.records
    )


class DummyPublisher:
    def __init__(self):
        self.published = []

    async def publish(self, *args, **kwargs):
        self.published.append(args)


@pytest.mark.asyncio
async def test_publish_input_received_filters_banned(monkeypatch):
    pub = DummyPublisher()
    sg._input_publisher = pub

    async def fake_ensure():
        return None

    monkeypatch.setattr(sg, "_ensure_nats", fake_ensure)

    await sg.publish_input_received("this contains banned text")
    assert pub.published == []

    await sg.publish_input_received("hello")
    assert pub.published and pub.published[0][0] == sg.EventSubjects.INPUT_RECEIVED


@pytest.mark.asyncio
async def test_on_message_ignores_other_bot_mentions(tmp_path, monkeypatch):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()
    sg.reply_limiter.clear("2")

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set(), {}))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(sg, "publish_input_received", noop)
    monkeypatch.setattr(asyncio, "sleep", noop)

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    dummy = DummyAuthor(5, bot=True)
    monkeypatch.setattr(
        sg.SocialGraphBot,
        "user",
        property(lambda self: dummy),
        raising=False,
    )

    message = DummyMessage("hi otherbot")
    message.mentions = [DummyAuthor(9, bot=True)]
    await bot.on_message(message)
    assert message.channel.sent_messages == []

    message2 = DummyMessage(f"hi both <@{bot.user.id}>", message_id=11)
    message2.mentions = [DummyAuthor(9, bot=True), bot.user]
    await bot.on_message(message2)
    assert message2.channel.sent_messages

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_on_message_includes_memory_hint(tmp_path, monkeypatch, input_events):

    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()
    sg.reply_limiter.clear("2")

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set(), {}))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(asyncio, "sleep", noop)
    monkeypatch.setattr(sg, "publish_input_received", noop)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])

    async def fake_recall_user(user_id):
        return [("greeting", "that you enjoy coding challenges")]

    monkeypatch.setattr(sg, "recall_user", fake_recall_user)

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    message = DummyMessage("hello with history")
    await bot.on_message(message)

    assert message.channel.sent_messages
    reply = message.channel.sent_messages[0]
    assert "You mentioned" in reply
    assert "coding challenges" in reply

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_on_message_manipulation_trust(tmp_path, monkeypatch, input_events):
    """Trust should decrease when manipulation is detected."""
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
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
    message = DummyMessage("After all I've done for you")
    await bot.on_message(message)

    trust = await sg.db_manager.get_trust(message.author.id)
    assert trust < 0
    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_on_message_classifier_manipulation(tmp_path, monkeypatch, input_events):
    """Trust decreases when classifier flags manipulation."""
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
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
    monkeypatch.setattr(sg, "manipulation_score", lambda text: "threat")

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    message = DummyMessage("hello")
    await bot.on_message(message)

    trust = await sg.db_manager.get_trust(message.author.id)
    assert trust < 0
    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_store_emotion(tmp_path):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    await sg.db_manager.store_emotion(1, {"happy": 0.8})

    async with aiosqlite.connect(str(tmp_path / "sg.db")) as db:
        async with db.execute(
            "SELECT emotion_json FROM emotions WHERE user_id=?",
            ("1",),
        ) as cur:
            row = await cur.fetchone()
    assert json.loads(row[0]) == {"happy": 0.8}
    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_log_manipulation(tmp_path):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    await sg.db_manager.log_manipulation(2, "coercion")

    async with aiosqlite.connect(str(tmp_path / "sg.db")) as db:
        async with db.execute(
            "SELECT manipulation_type FROM manipulations WHERE user_id=?",
            ("2",),
        ) as cur:
            row = await cur.fetchone()
    assert row[0] == "coercion"
    await sg.db_manager.close()
