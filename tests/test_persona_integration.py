import asyncio
import random

import pytest

import examples.social_graph_bot as sg
from deepthought.services import PersonaManager


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
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

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

    bot = sg.SocialGraphBot(monitor_channel_id=1)

    msg1 = DummyMessage("hi")
    await bot.on_message(msg1)
    assert msg1.channel.sent_messages[-1] == sg.PERSONA_REPLIES["snarky"][0]

    await sg.adjust_affinity(msg1.author.id, 1)
    msg2 = DummyMessage("hi again", author_id=msg1.author.id, message_id=11)
    await bot.on_message(msg2)
    assert msg2.channel.sent_messages[-1] == sg.PERSONA_REPLIES["playful"][0]

    await sg.adjust_affinity(msg1.author.id, 2)
    msg3 = DummyMessage("hello friend", author_id=msg1.author.id, message_id=12)
    await bot.on_message(msg3)
    assert msg3.channel.sent_messages[-1] == sg.PERSONA_REPLIES["friendly"][0]

    await sg.db_manager.close()
