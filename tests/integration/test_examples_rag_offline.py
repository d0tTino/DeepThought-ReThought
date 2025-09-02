import importlib
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("aiosqlite")


@pytest.fixture(autouse=True)
def _stub_dependencies(monkeypatch):
    fake_nats = types.ModuleType("nats")
    fake_nats.__spec__ = importlib.machinery.ModuleSpec("nats", loader=None)
    fake_nats.connect = lambda *a, **k: None
    fake_nats.errors = types.SimpleNamespace(Error=Exception, TimeoutError=Exception)
    fake_nats.aio = types.ModuleType("aio")
    fake_nats.aio.client = types.ModuleType("client")
    fake_nats.aio.client.Client = object
    fake_nats.aio.msg = types.ModuleType("msg")
    fake_nats.aio.msg.Msg = object
    fake_nats.js = types.ModuleType("js")
    fake_nats.js.client = types.ModuleType("client")
    fake_nats.js.client.JetStreamContext = object
    fake_nats.js.api = types.ModuleType("api")
    sys.modules.setdefault("nats", fake_nats)
    sys.modules.setdefault("nats.aio", fake_nats.aio)
    sys.modules.setdefault("nats.aio.client", fake_nats.aio.client)
    sys.modules.setdefault("nats.aio.msg", fake_nats.aio.msg)
    sys.modules.setdefault("nats.js", fake_nats.js)
    sys.modules.setdefault("nats.js.client", fake_nats.js.client)
    sys.modules.setdefault("nats.js.api", fake_nats.js.api)
    sys.modules.setdefault("nats.errors", fake_nats.errors)

    fake_prom = types.ModuleType("prometheus_client")

    class _Metric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    fake_prom.Counter = lambda *a, **k: _Metric()
    fake_prom.Histogram = lambda *a, **k: _Metric()
    fake_prom.start_http_server = lambda *a, **k: None
    fake_prom.REGISTRY = types.SimpleNamespace(_names_to_collectors={})
    sys.modules.setdefault("prometheus_client", fake_prom)

    fake_pyd = types.ModuleType("pydantic")
    fake_pyd.AnyUrl = str
    fake_pyd.ValidationError = Exception
    fake_pyd.Field = lambda default=None, **kwargs: default
    sys.modules.setdefault("pydantic", fake_pyd)
    fake_ps = types.ModuleType("pydantic_settings")
    fake_ps.BaseSettings = object
    fake_ps.SettingsConfigDict = dict
    sys.modules.setdefault("pydantic_settings", fake_ps)

    tb_mod = types.ModuleType("textblob")
    tb_mod.TextBlob = lambda text: types.SimpleNamespace(sentiment=types.SimpleNamespace(polarity=0.0))
    sys.modules.setdefault("textblob", tb_mod)
    vader_pkg = types.ModuleType("vaderSentiment")
    vader_mod = types.ModuleType("vaderSentiment.vaderSentiment")
    vader_mod.SentimentIntensityAnalyzer = object
    sys.modules.setdefault("vaderSentiment", vader_pkg)
    sys.modules.setdefault("vaderSentiment.vaderSentiment", vader_mod)

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
    ns_mod.RDF = object()
    rdf_mod.Namespace = lambda uri=None: types.SimpleNamespace(__uri=uri or "")
    rdf_mod.Graph = object
    rdf_mod.URIRef = str
    sys.modules.setdefault("rdflib", rdf_mod)
    sys.modules.setdefault("rdflib.namespace", ns_mod)

    # Provide lightweight deepthought.services module exposing only CognitiveCoreService
    services = types.ModuleType("deepthought.services")
    cc_path = Path(__file__).resolve().parents[2] / "src" / "deepthought" / "services" / "cognitive_core_service.py"
    cc_spec = importlib.util.spec_from_file_location("deepthought.services.cognitive_core_service", cc_path.resolve())
    cc_mod = importlib.util.module_from_spec(cc_spec)
    assert cc_spec.loader is not None
    cc_spec.loader.exec_module(cc_mod)
    services.CognitiveCoreService = cc_mod.CognitiveCoreService
    sys.modules.setdefault("deepthought.services", services)


def test_rag_demo_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.find_spec("examples.rag_demo")
    demo = importlib.util.module_from_spec(spec)
    sys.modules["examples.rag_demo"] = demo
    spec.loader.exec_module(demo)
    demo.main()
    assert Path("rag_demo.db").exists()


def test_offline_search_demo_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.find_spec("examples.offline_search_demo")
    demo = importlib.util.module_from_spec(spec)
    sys.modules["examples.offline_search_demo"] = demo
    spec.loader.exec_module(demo)
    demo.main()
    assert Path("offline_demo.db").exists()
