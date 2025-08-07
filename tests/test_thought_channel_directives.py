import importlib
import os
import sys
import types

import pytest


def reload_sg(monkeypatch: pytest.MonkeyPatch):
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


class DummyChannel:
    def __init__(self):
        self.sent: list[str] = []

    async def send(self, text: str) -> None:
        self.sent.append(text)


@pytest.mark.asyncio
async def test_goal_directive_internal_only(monkeypatch: pytest.MonkeyPatch):
    sg = reload_sg(monkeypatch)
    monkeypatch.setattr(sg, "THOUGHT_CHANNEL_ID", 7)

    queued: list[tuple[str, int]] = []

    class Bot:
        def __init__(self):
            self.channel = DummyChannel()
            self.goal_scheduler = types.SimpleNamespace(
                queue_intention=lambda goal, priority: queued.append((goal, priority))
            )
            self.user = types.SimpleNamespace(id=1)
            self._closed = False

        async def wait_until_ready(self):
            return None

        def is_closed(self) -> bool:
            return self._closed

        async def wait_for(self, event: str, check):
            msg = types.SimpleNamespace(
                content="/goal study",
                channel=types.SimpleNamespace(id=7),
                author=types.SimpleNamespace(bot=False, id=2),
            )
            assert check(msg)
            self._closed = True
            return msg

    bot = Bot()
    await sg.process_thought_commands(bot)

    assert queued == [("study", 1)]
    assert bot.channel.sent == []
