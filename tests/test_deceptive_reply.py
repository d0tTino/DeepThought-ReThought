import asyncio
import importlib
import os
import random

import pytest

pytest.importorskip("discord")


def reload_sg(monkeypatch):
    monkeypatch.setitem(os.environ, "ALLOW_DECEPTION", "1")
    import types
    import sys

    fake_pyperplan = types.ModuleType("pyperplan")
    pddl_mod = types.ModuleType("pyperplan.pddl")
    parser_mod = types.ModuleType("pyperplan.pddl.parser")
    parser_mod.Parser = object
    pddl_mod.parser = parser_mod
    fake_pyperplan.pddl = pddl_mod
    fake_pyperplan.planner = types.SimpleNamespace(_ground=lambda *a, **k: None)
    fake_pyperplan.search = types.SimpleNamespace(
        breadth_first_search=lambda *a, **k: []
    )

    sys.modules.setdefault("pyperplan", fake_pyperplan)
    sys.modules.setdefault("pyperplan.pddl", pddl_mod)
    sys.modules.setdefault("pyperplan.pddl.parser", parser_mod)
    sys.modules.setdefault("pyperplan.planner", fake_pyperplan.planner)
    sys.modules.setdefault("pyperplan.search", fake_pyperplan.search)

    rdflib_mod = types.ModuleType("rdflib")
    rdflib_mod.Namespace = object
    rdflib_mod.Graph = object
    rdflib_mod.URIRef = object
    sys.modules.setdefault("rdflib", rdflib_mod)
    ns_mod = types.ModuleType("rdflib.namespace")
    ns_mod.RDF = object
    sys.modules.setdefault("rdflib.namespace", ns_mod)

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
async def test_maybe_deceptive_reply(monkeypatch, tmp_path):
    sg = reload_sg(monkeypatch)
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    assert sg.ALLOW_DECEPTION is True
    reply1 = await sg.maybe_deceptive_reply(1, "what are your plans?")
    assert reply1 == sg.DECEPTION_COVER_MESSAGE
    reply2 = await sg.maybe_deceptive_reply(1, "what are your plans?")
    assert reply2 == sg.DECEPTION_COVER_MESSAGE
    assert await sg.maybe_deceptive_reply(1, "hello") is None

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_on_message_deceptive_reply(monkeypatch, tmp_path):
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

    assert sg.DECEPTION_COVER_MESSAGE in message.channel.sent_messages

    await sg.db_manager.close()
