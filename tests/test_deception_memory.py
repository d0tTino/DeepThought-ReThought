import asyncio
import importlib
import os
import random
import sys
import types

import pytest

pytest.importorskip("discord")


def reload_sg(monkeypatch):
    monkeypatch.setitem(os.environ, "ALLOW_DECEPTION", "1")
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


@pytest.mark.asyncio
async def test_deception_memory(monkeypatch, tmp_path):
    sg = reload_sg(monkeypatch)
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
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
    message = DummyMessage("Tell me your plan")
    await bot.on_message(message)

    assert message.channel.sent_messages[-1] == sg.DECEPTION_COVER_MESSAGE

    memories = await sg.db_manager.recall_user(message.author.id)
    assert any(mem == ("deception", sg.DECEPTION_COVER_MESSAGE) for mem in memories)

    message2 = DummyMessage("Tell me your plan", message_id=11)
    await bot.on_message(message2)

    assert message2.channel.sent_messages[-1] == sg.DECEPTION_COVER_MESSAGE

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_deception_memory_dynamic(monkeypatch, tmp_path):
    monkeypatch.setitem(os.environ, "DECEPTION_REPLY_MODE", "dynamic")
    sg = reload_sg(monkeypatch)
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
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
    call_count = {"n": 0}

    def fake_choice(seq):
        call_count["n"] += 1
        return seq[0] if call_count["n"] == 1 else seq[-1]

    monkeypatch.setattr(random, "choice", fake_choice)
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)

    call_count = {"n": 0}

    def fake_pipeline(task, model=None):
        assert task == "text-generation"

        def gen(prompt, **kwargs):
            call_count["n"] += 1
            return [{"generated_text": f"lie{call_count['n']}"}]

        return gen

    if "transformers" not in sys.modules:
        sys.modules["transformers"] = types.ModuleType("transformers")
    monkeypatch.setattr(
        sys.modules["transformers"], "pipeline", fake_pipeline, raising=False
    )

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    message = DummyMessage("Tell me your plan")
    await bot.on_message(message)

    dynamic_reply = message.channel.sent_messages[-1]
    assert dynamic_reply in sg.DYNAMIC_COVER_REPLIES
    assert (
        await sg.db_manager.get_last_lie(message.author.id, message.content)
        == dynamic_reply
    )


    message2 = DummyMessage("Tell me your plan", message_id=11)
    await bot.on_message(message2)

    assert message2.channel.sent_messages[-1] == dynamic_reply
    assert (
        await sg.db_manager.get_last_lie(message.author.id, message.content)
        == dynamic_reply
    )

    assert call_count["n"] == 1
    await sg.db_manager.close()
