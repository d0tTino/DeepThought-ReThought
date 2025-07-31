import asyncio
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
tb_mod.TextBlob = lambda text: types.SimpleNamespace(
    sentiment=types.SimpleNamespace(polarity=0.0)
)
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
rdf_mod.Graph = type(
    "Graph",
    (),
    {"add": lambda self, t: None, "serialize": lambda self, format="xml": ""},
)
rdf_mod.URIRef = str
sys.modules.setdefault("rdflib", rdf_mod)
sys.modules.setdefault("rdflib.namespace", ns_mod)

fake_nats = types.ModuleType("nats")
fake_nats.aio = types.ModuleType("aio")
fake_nats.aio.client = types.ModuleType("client")
fake_nats.js = types.ModuleType("js")
fake_nats.js.client = types.ModuleType("client")
fake_nats.errors = types.ModuleType("errors")
sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", fake_nats.aio)
sys.modules.setdefault("nats.aio.client", fake_nats.aio.client)
sys.modules.setdefault("nats.js", fake_nats.js)
sys.modules.setdefault("nats.js.client", fake_nats.js.client)
sys.modules.setdefault("nats.errors", fake_nats.errors)

from deepthought.goal_scheduler import GoalScheduler
from deepthought.services.db_manager import DBManager


@pytest.mark.asyncio
async def test_scheduler_autoload(tmp_path):
    db_file = tmp_path / "db.sqlite"
    manager = DBManager(str(db_file))
    await manager.init_db()
    await manager.add_intention("alpha", 1)
    await manager.add_intention("beta", 5)
    sched = GoalScheduler(manager)
    await sched.wait_loaded()
    assert sched.next_goal() == "beta"
    assert sched.next_goal() == "alpha"
    await manager.close()
