import asyncio

import aiosqlite
import pytest

import examples.social_graph_bot as sg


@pytest.mark.asyncio
async def test_store_theory(tmp_path):
    db_file = tmp_path / "db.sqlite"
    sg.DB_PATH = str(db_file)
    await sg.init_db()
    await sg.store_theory("u1", "insomniac", 0.5)
    async with aiosqlite.connect(sg.DB_PATH) as db:
        async with db.execute("SELECT theory FROM theories WHERE subject_id=?", ("u1",)) as cur:
            row = await cur.fetchone()
    assert row[0] == "insomniac"


@pytest.mark.asyncio
async def test_store_theory_update(tmp_path):
    db_file = tmp_path / "db.sqlite"
    sg.DB_PATH = str(db_file)
    await sg.init_db()
    await sg.store_theory("u1", "insomniac", 0.5)
    await sg.store_theory("u1", "insomniac", 0.8)
    async with aiosqlite.connect(sg.DB_PATH) as db:
        async with db.execute(
            "SELECT confidence FROM theories WHERE subject_id=? AND theory=?",
            ("u1", "insomniac"),
        ) as cur:
            row = await cur.fetchone()
    assert row[0] == 0.8


@pytest.mark.asyncio
async def test_store_memory(tmp_path):
    db_file = tmp_path / "db.sqlite"
    sg.DB_PATH = str(db_file)
    await sg.init_db()
    await sg.store_memory("u1", "hello", sentiment_score=0.3)
    async with aiosqlite.connect(sg.DB_PATH) as db:
        async with db.execute(
            "SELECT memory, sentiment_score FROM memories WHERE user_id=?",
            ("u1",),
        ) as cur:
            row = await cur.fetchone()
    assert row == ("hello", 0.3)


@pytest.mark.asyncio
async def test_queue_deep_reflection(tmp_path):
    db_file = tmp_path / "db.sqlite"
    sg.DB_PATH = str(db_file)
    await sg.init_db()
    task_id = await sg.queue_deep_reflection("u2", {"channel_id": 1}, "hello")
    async with aiosqlite.connect(sg.DB_PATH) as db:
        async with db.execute("SELECT status FROM queued_tasks WHERE task_id=?", (task_id,)) as cur:
            row = await cur.fetchone()
    assert row[0] == "pending"


def test_evaluate_triggers():
    class Dummy:
        def __init__(self, content, hour):
            from datetime import datetime, timezone

            self.content = content
            self.created_at = datetime(2025, 1, 1, hour, tzinfo=timezone.utc)

    m1 = Dummy("I agree with you", 1)
    assert ("social chameleon", 0.6) in sg.evaluate_triggers(m1)
    m2 = Dummy("hello", 2)
    assert ("insomniac", 0.7) in sg.evaluate_triggers(m2)


def test_generate_reflection_positive_negative():
    pos = sg.generate_reflection("I love this!")
    neg = sg.generate_reflection("I hate this!")
    assert pos == "Your message felt positive."
    assert neg == "Your message felt negative."


def test_generate_reflection_neutral():
    neutral = sg.generate_reflection("Meh")
    assert neutral == "Your message felt neutral."


class DummyChannel:
    def __init__(self):
        self.sent_messages = []

    async def send(self, content, reference=None):
        self.sent_messages.append(content)

    async def fetch_message(self, message_id):
        return None


class DummyBot:
    def __init__(self):
        self.channel = DummyChannel()
        self.closed = False

    async def wait_until_ready(self):
        return None

    def is_closed(self):
        was_closed = self.closed
        self.closed = True
        return was_closed

    def get_channel(self, channel_id):
        return self.channel if channel_id == 1 else None


@pytest.mark.asyncio
async def test_process_deep_reflections_posts(tmp_path, monkeypatch):
    sg.DB_PATH = str(tmp_path / "db.sqlite")
    await sg.init_db()

    bot = DummyBot()

    await sg.queue_deep_reflection("u1", {"channel_id": 1, "message_id": 1}, "I love bots")

    async def noop(*_, **__):
        return None

    monkeypatch.setattr(asyncio, "sleep", noop)
    monkeypatch.setattr(sg, "REFLECTION_CHECK_SECONDS", 0)

    await sg.process_deep_reflections(bot)

    async with aiosqlite.connect(sg.DB_PATH) as db:
        async with db.execute("SELECT status FROM queued_tasks WHERE task_id=1") as cur:
            row = await cur.fetchone()
    assert row[0] == "done"
    assert bot.channel.sent_messages == ["After some thought... Your message felt positive."]


@pytest.mark.asyncio
async def test_process_deep_reflections_negative(tmp_path, monkeypatch):
    sg.DB_PATH = str(tmp_path / "db.sqlite")
    await sg.init_db()

    bot = DummyBot()

    await sg.queue_deep_reflection("u1", {"channel_id": 1, "message_id": 1}, "I hate bots")

    async def noop(*_, **__):
        return None

    monkeypatch.setattr(asyncio, "sleep", noop)
    monkeypatch.setattr(sg, "REFLECTION_CHECK_SECONDS", 0)

    await sg.process_deep_reflections(bot)

    async with aiosqlite.connect(sg.DB_PATH) as db:
        async with db.execute("SELECT status FROM queued_tasks WHERE task_id=1") as cur:
            row = await cur.fetchone()
    assert row[0] == "done"
    assert bot.channel.sent_messages == ["After some thought... Your message felt negative."]
