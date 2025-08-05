import asyncio
import importlib
import logging
import os
import sys
import types

import pytest


def reload_sg(monkeypatch):
    monkeypatch.setitem(os.environ, "BOT_COOLDOWN_SECONDS", "1")
    tb_mod = types.ModuleType("textblob")
    tb_mod.TextBlob = lambda text: types.SimpleNamespace(sentiment=types.SimpleNamespace(polarity=0.0))
    sys.modules.setdefault("textblob", tb_mod)
    sys.modules.setdefault("torch", types.ModuleType("torch"))
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


sg = reload_sg(pytest.MonkeyPatch())


class DummyChannel:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(text)


class DummyBot:
    def __init__(self, channel=None):
        self.channel = channel or DummyChannel()

    def get_channel(self, cid):
        return self.channel


@pytest.mark.asyncio
async def test_log_thought_sends_message(monkeypatch):
    channel = DummyChannel()
    bot = DummyBot(channel)
    monkeypatch.setattr(sg, "THOUGHT_CHANNEL_ID", 123)

    sg.log_thought(bot, "hello")
    await asyncio.sleep(0)

    assert channel.sent == ["hello"]


@pytest.mark.asyncio
async def test_send_thought_missing_channel(monkeypatch, caplog):
    monkeypatch.setattr(sg, "THOUGHT_CHANNEL_ID", 999)
    bot = DummyBot(channel=None)
    await sg._send_thought(bot, "hi")


@pytest.mark.asyncio
async def test_process_thought_commands_goal(monkeypatch):
    monkeypatch.setattr(sg, "THOUGHT_CHANNEL_ID", 42)

    queued: list[tuple[str, int]] = []

    class Bot(DummyBot):
        def __init__(self):
            super().__init__()
            self.goal_scheduler = types.SimpleNamespace(
                queue_intention=lambda goal, priority: queued.append((goal, priority))
            )
            self.user = types.SimpleNamespace(id=1)
            self._closed = False

        async def wait_until_ready(self):
            return None

        def is_closed(self):
            return self._closed

        async def wait_for(self, event, check):
            msg = types.SimpleNamespace(
                content="/goal test",
                channel=types.SimpleNamespace(id=42),
                author=types.SimpleNamespace(bot=False, id=2),
            )
            assert check(msg)
            self._closed = True
            return msg

    bot = Bot()
    await sg.process_thought_commands(bot)
    assert queued == [("test", 1)]


@pytest.mark.asyncio
async def test_process_thought_commands_memory(monkeypatch):
    monkeypatch.setattr(sg, "THOUGHT_CHANNEL_ID", 99)

    stored: list[tuple[int, str]] = []

    async def fake_store(uid, mem, *_, **__):
        stored.append((uid, mem))

    monkeypatch.setattr(sg, "store_memory", fake_store)

    class Bot(DummyBot):
        def __init__(self):
            super().__init__()
            self.goal_scheduler = types.SimpleNamespace(queue_intention=lambda *a, **k: None)
            self.user = types.SimpleNamespace(id=1)
            self._closed = False

        async def wait_until_ready(self):
            return None

        def is_closed(self):
            return self._closed

        async def wait_for(self, event, check):
            msg = types.SimpleNamespace(
                content="/memory remember this",
                channel=types.SimpleNamespace(id=99),
                author=types.SimpleNamespace(bot=False, id=5),
            )
            assert check(msg)
            self._closed = True
            return msg

    bot = Bot()
    await sg.process_thought_commands(bot)
    assert stored == [(5, "remember this")]
