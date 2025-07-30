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



pytest.importorskip("aiosqlite")

from deepthought.services.db_manager import DBManager


@pytest.mark.asyncio
async def test_intention_persistence(tmp_path):
    db_file = tmp_path / "db.sqlite"
    manager = DBManager(str(db_file))
    await manager.init_db()

    first = await manager.add_intention("goal one", 1)
    second = await manager.add_intention("goal two", 5)

    rows = await manager.list_pending_intentions()
    assert rows == [(second, "goal two", 5), (first, "goal one", 1)]

    await manager.mark_intention_done(first)
    rows = await manager.list_pending_intentions()
    assert (first, "goal one", 1) not in rows

    await manager.close()


@pytest.mark.asyncio
async def test_manipulation_trust_adjustment(tmp_path):
    db_file = tmp_path / "db.sqlite"
    manager = DBManager(str(db_file))
    await manager.init_db()

    await manager.adjust_trust("u1", -0.6)
    await manager.adjust_trust("u1", -0.2)

    trust = await manager.get_trust("u1")
    assert pytest.approx(trust) == -0.8

    await manager.close()
