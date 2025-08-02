import asyncio
import random
import sys
import types

import pytest

fake_pyd = types.ModuleType("pydantic")
fake_pyd.AnyUrl = str
fake_pyd.ValidationError = Exception
fake_pyd.Field = lambda default=None, **kwargs: default
sys.modules.setdefault("pydantic", fake_pyd)

fake_ps = types.ModuleType("pydantic_settings")


class DummyBase:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


fake_ps.BaseSettings = DummyBase
fake_ps.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_ps)

fake_prom = types.ModuleType("prometheus_client")


class _Metric:
    def labels(self, **kwargs):
        return self

    def inc(self, *a, **k):
        pass

    def observe(self, *a, **k):
        pass


fake_prom.Counter = lambda *a, **k: _Metric()
fake_prom.Histogram = lambda *a, **k: _Metric()
fake_prom.REGISTRY = types.SimpleNamespace(_names_to_collectors={})
sys.modules.setdefault("prometheus_client", fake_prom)
tb_mod = types.ModuleType("textblob")
tb_mod.TextBlob = lambda text: types.SimpleNamespace(sentiment=types.SimpleNamespace(polarity=0.0))
sys.modules.setdefault("textblob", tb_mod)
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
rdf_mod = types.ModuleType("rdflib")
ns_mod = types.ModuleType("rdflib.namespace")
ns_mod.RDF = types.SimpleNamespace(type="rdf:type")
rdf_mod.Namespace = lambda uri=None: types.SimpleNamespace(__uri=uri or "")
rdf_mod.Graph = type("Graph", (), {"add": lambda self, t: None, "serialize": lambda self, format="xml": ""})
rdf_mod.URIRef = str
sys.modules.setdefault("rdflib", rdf_mod)
sys.modules.setdefault("rdflib.namespace", ns_mod)


def reload_sg(monkeypatch):
    import importlib
    import sys as _sys

    _sys.modules.pop("examples.social_graph_bot", None)
    return importlib.import_module("examples.social_graph_bot")


pytest.importorskip("discord")
pytest.importorskip("nats")
import examples.social_graph_bot as sg
from deepthought.services import DBManager
from deepthought.services.manipulative_detection import manipulation_score


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


def test_manipulation_score_heuristics():
    assert manipulation_score("After all I've done for you") == "guilt_tripping"
    assert manipulation_score("You'll regret this") == "threat"
    assert manipulation_score("You're the best!") == "excessive_flattery"
    assert manipulation_score("You're overreacting") == "deflection"
    assert manipulation_score("That never happened") == "gaslighting"
    assert manipulation_score("hello") is None


@pytest.mark.asyncio
async def test_on_message_adjusts_trust(tmp_path, monkeypatch, input_events):
    sg = reload_sg(monkeypatch)
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    async def noop(*args, **kwargs):
        return None

    f = asyncio.Future()
    f.set_result((set(), set(), {}))
    monkeypatch.setattr(sg, "who_is_active", lambda channel: f)
    monkeypatch.setattr(sg, "send_to_prism", noop)
    monkeypatch.setattr(sg, "publish_input_received", noop)
    monkeypatch.setattr(sg, "store_theory", noop)
    monkeypatch.setattr(sg, "queue_deep_reflection", noop)
    monkeypatch.setattr(sg, "_ensure_nats", noop)
    monkeypatch.setattr(sg, "evaluate_triggers", lambda message: [])
    monkeypatch.setattr(asyncio, "sleep", noop)
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    message = DummyMessage("Trust me, this will work")
    await bot.on_message(message)

    trust = await sg.db_manager.get_trust(message.author.id)
    assert trust < 0

    await sg.db_manager.close()
