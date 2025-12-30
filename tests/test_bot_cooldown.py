import asyncio
import importlib
import os
import random
import sys
import types

import pytest

pytest.importorskip("discord")
pytest.importorskip("nats")


def reload_sg(monkeypatch):
    monkeypatch.setitem(os.environ, "BOT_COOLDOWN_SECONDS", "1")
    sys.modules.setdefault("pydantic", types.ModuleType("pydantic"))
    sys.modules.setdefault("pydantic_settings", types.ModuleType("pydantic_settings"))
    sys.modules.setdefault("prometheus_client", types.ModuleType("prometheus_client"))
    tb_mod = types.ModuleType("textblob")
    tb_mod.TextBlob = lambda text: types.SimpleNamespace(
        sentiment=types.SimpleNamespace(polarity=0.0)
    )
    sys.modules.setdefault("textblob", tb_mod)
    rdf_mod = types.ModuleType("rdflib")
    rdf_mod.Namespace = lambda *a, **k: None
    rdf_mod.Graph = object
    rdf_mod.URIRef = str
    ns_mod = types.ModuleType("rdflib.namespace")
    ns_mod.RDF = types.SimpleNamespace(type="rdf:type")
    sys.modules.setdefault("rdflib", rdf_mod)
    sys.modules.setdefault("rdflib.namespace", ns_mod)
    py_mod = types.ModuleType("pyperplan")
    py_mod.pddl = types.ModuleType("pyperplan.pddl")
    parser_mod = types.ModuleType("pyperplan.pddl.parser")
    parser_mod.Parser = object
    py_mod.planner = types.ModuleType("pyperplan.planner")
    py_mod.planner._ground = lambda p: p
    py_mod.search = types.ModuleType("pyperplan.search")
    py_mod.search.breadth_first_search = lambda t: []
    py_mod.pddl.parser = parser_mod
    sys.modules.setdefault("pyperplan", py_mod)
    sys.modules.setdefault("pyperplan.pddl", py_mod.pddl)
    sys.modules.setdefault("pyperplan.pddl.parser", parser_mod)
    sys.modules.setdefault("pyperplan.planner", py_mod.planner)
    sys.modules.setdefault("pyperplan.search", py_mod.search)
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
    def __init__(self, content, author_id=2, message_id=10, is_bot=True):
        from discord.utils import utcnow

        self.content = content
        self.author = DummyAuthor(author_id, bot=is_bot)
        self.channel = DummyChannel()
        self.id = message_id
        self.created_at = utcnow()
        self.mentions = []
        self.replies = []

    async def reply(self, content, mention_author=True):
        self.replies.append({"content": content, "mention_author": mention_author})
        await self.channel.send(content, reference=self)


@pytest.mark.asyncio
async def test_on_message_bot_cooldown(monkeypatch, tmp_path):
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

    msg1 = DummyMessage("hi", author_id=8, message_id=10, is_bot=True)
    await bot.on_message(msg1)

    msg2 = DummyMessage("hi", author_id=8, message_id=11, is_bot=True)
    await bot.on_message(msg2)

    assert len(msg1.channel.sent_messages) == 1
    assert msg2.channel.sent_messages == []
    await sg.db_manager.close()
