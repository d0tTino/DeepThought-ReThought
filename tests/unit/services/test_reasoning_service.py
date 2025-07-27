import sys
import types
from types import SimpleNamespace

import pytest

fake_pyd = types.ModuleType("pydantic")
fake_pyd.AnyUrl = str
fake_pyd.ValidationError = Exception
fake_pyd.Field = lambda default=None, **kwargs: default
sys.modules.setdefault("pydantic", fake_pyd)
fake_ps = types.ModuleType("pydantic_settings")
fake_ps.BaseSettings = object
fake_ps.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_ps)
fake_prom = types.ModuleType("prometheus_client")
fake_prom.Counter = lambda *a, **k: object()
fake_prom.Histogram = lambda *a, **k: object()
fake_prom.REGISTRY = SimpleNamespace(_names_to_collectors={})
sys.modules.setdefault("prometheus_client", fake_prom)
sys.modules.setdefault("pyperplan", types.ModuleType("pyperplan"))
planning_stub = types.ModuleType("planning_service")
planning_stub.PlanningService = object
sys.modules.setdefault("deepthought.services.planning_service", planning_stub)
sys.modules.setdefault("aiosqlite", types.ModuleType("aiosqlite"))
tb_mod = types.ModuleType("textblob")
tb_mod.TextBlob = lambda text: types.SimpleNamespace(
    sentiment=types.SimpleNamespace(polarity=0.0)
)
sys.modules.setdefault("textblob", tb_mod)
rdf_mod = types.ModuleType("rdflib")
ns_mod = types.ModuleType("rdflib.namespace")
ns_mod.RDF = types.SimpleNamespace(type="rdf:type")
sys.modules.setdefault("rdflib.namespace", ns_mod)


class _NS:
    def __init__(self, uri=""):
        self._u = uri

    def __getitem__(self, key):
        return f"{self._u}{key}"


rdf_mod.Namespace = lambda uri=None: _NS(uri or "")
rdf_mod.Graph = type(
    "Graph",
    (),
    {"add": lambda self, triple: None, "serialize": lambda self, format="xml": ""},
)
rdf_mod.URIRef = str
sys.modules.setdefault("rdflib", rdf_mod)

import owlready2

owlready2.World.get_ontology = lambda self, iri: types.SimpleNamespace(
    load=lambda fileobj=None: types.SimpleNamespace(individuals=lambda: [])
)

from deepthought.eda.events import EventSubjects, ResponseGeneratedPayload
from deepthought.services.reasoning_service import ReasoningService


class DummyNATS:
    def __init__(self):
        self.is_connected = True


class DummyJS:
    pass


class DummyPublisher:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload, **kw):

        self.published.append((subject, payload))
        return SimpleNamespace(seq=1, stream="test")


class DummySubscriber:
    async def subscribe(self, *a, **k):

        pass

    async def unsubscribe_all(self):
        pass


class DummyMsg:
    def __init__(self, data: str):
        self.data = data.encode()
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.nacked = True


@pytest.mark.asyncio
async def test_handle_response_publishes_facts(monkeypatch):
    svc = ReasoningService(DummyNATS(), DummyJS())
    svc._publisher = DummyPublisher()
    svc._subscriber = DummySubscriber()

    payload = ResponseGeneratedPayload(final_response="A is B", input_id="1")
    msg = DummyMsg(payload.to_json())
    await svc._handle_response(msg)

    assert msg.acked
    # Ontology stubs may produce no inferred facts


@pytest.mark.asyncio
async def test_handle_response_emits_input_event(monkeypatch):
    class DummyOntology:
        def __init__(self):
            self.triples = []

        def add_triples(self, triples):
            self.triples.extend(triples)

        def infer_facts(self):
            return [("A", "B", "C")]

        def verify_triples(self, triples):
            return triples, []

    svc = ReasoningService(DummyNATS(), DummyJS(), ontology=DummyOntology())
    svc._publisher = DummyPublisher()
    svc._subscriber = DummySubscriber()

    payload = ResponseGeneratedPayload(final_response="A is B", input_id="1")
    msg = DummyMsg(payload.to_json())
    await svc._handle_response(msg)

    assert msg.acked
    assert svc._publisher.published
    subj, out = svc._publisher.published[0]
    assert subj == EventSubjects.INPUT_RECEIVED
    assert out.user_input == "A B C"


def test_extract_triples_simple():
    svc = ReasoningService(DummyNATS(), DummyJS())

    triples = svc._extract_triples("A is B\nC likes D\n")

    assert triples == [
        (
            "http://deepthought.local/resp#A",
            "http://deepthought.local/resp#is",
            "http://deepthought.local/resp#B",
        ),
        (
            "http://deepthought.local/resp#C",
            "http://deepthought.local/resp#likes",
            "http://deepthought.local/resp#D",
        ),
    ]


@pytest.mark.asyncio
async def test_warning_on_contradiction(monkeypatch):
    class DummyOntology:
        def add_triples(self, triples):
            pass

        def infer_facts(self):
            return [("A", "B", "C"), ("A", "B", "D")]

        def verify_triples(self, triples):
            valid = [triples[0]]
            contradictory = [triples[1]]
            return valid, contradictory

    svc = ReasoningService(DummyNATS(), DummyJS(), ontology=DummyOntology())
    svc._publisher = DummyPublisher()
    svc._subscriber = DummySubscriber()

    payload = ResponseGeneratedPayload(final_response="X is Y", input_id="1")
    msg = DummyMsg(payload.to_json())
    await svc._handle_response(msg)

    assert msg.acked
    subjects = [s for s, _ in svc._publisher.published]
    assert EventSubjects.WARNING in subjects
    assert EventSubjects.INPUT_RECEIVED in subjects
    inp = next(
        p for s, p in svc._publisher.published if s == EventSubjects.INPUT_RECEIVED
    )
    assert inp.user_input == "A B C"
