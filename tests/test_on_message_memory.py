import asyncio

import aiosqlite
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
async def test_on_message_stores_memory(tmp_path, monkeypatch):
    sg.DB_PATH = str(tmp_path / "sg.db")
    await sg.init_db()

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set()))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(asyncio, "sleep", noop)

    bot = sg.SocialGraphBot(monitor_channel_id=1)

    message = DummyMessage("hello world")
    await bot.on_message(message)

    async with aiosqlite.connect(sg.DB_PATH) as db:
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


@pytest.mark.asyncio
async def test_on_message_calls_send_to_prism(tmp_path, monkeypatch, prism_calls):
    sg.DB_PATH = str(tmp_path / "sg.db")
    await sg.init_db()

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set()))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(asyncio, "sleep", noop)

    bot = sg.SocialGraphBot(monitor_channel_id=1)

    message = DummyMessage("send prism")
    await bot.on_message(message)

    assert len(prism_calls) == 1
    assert prism_calls[0]["content"] == "send prism"


@pytest.mark.asyncio
async def test_on_message_multiple_messages(tmp_path, monkeypatch):
    """Each message should result in exactly one memory record."""
    sg.DB_PATH = str(tmp_path / "sg.db")
    await sg.init_db()

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set()))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(asyncio, "sleep", noop)

    bot = sg.SocialGraphBot(monitor_channel_id=1)

    m1 = DummyMessage("first")
    m2 = DummyMessage("second", message_id=11)
    await bot.on_message(m1)
    await bot.on_message(m2)

    async with aiosqlite.connect(sg.DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id=?",
            (str(m1.author.id),),
        ) as cur:
            count_row = await cur.fetchone()

    assert count_row[0] == 2
