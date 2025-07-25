import json
import importlib.machinery
import sys
import types
from types import SimpleNamespace

import pytest

# Provide minimal stubs for rdflib and owlready2 so the service can be imported
fake_rdflib = types.ModuleType("rdflib")
fake_rdflib.Graph = type("Graph", (), {"serialize": lambda self, format=None: b""})
fake_rdflib.URIRef = lambda s: s
ns_mod = types.ModuleType("namespace")
ns_mod.RDF = types.SimpleNamespace(type="rdf:type")
fake_rdflib.namespace = ns_mod
sys.modules.setdefault("rdflib", fake_rdflib)
sys.modules.setdefault("rdflib.namespace", ns_mod)

fake_owl = types.ModuleType("owlready2")
fake_owl.ThingClass = type("ThingClass", (), {})
fake_owl.World = type("World", (), {"get_ontology": lambda self, uri: types.SimpleNamespace(load=lambda fileobj=None: types.SimpleNamespace(individuals=lambda: []))})
fake_owl.sync_reasoner_hermit = lambda world: None
sys.modules.setdefault("owlready2", fake_owl)

from deepthought.eda.events import EventSubjects
from deepthought.services.reasoning_service import ReasoningService, TRIPLE_SUBJECT


class DummyNATS:
    def __init__(self):
        self.is_connected = True


class DummyJS:
    pass


class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload))
        return SimpleNamespace(seq=1, stream="test")


class DummySubscriber:
    async def subscribe(self, *args, **kwargs):
        pass

    async def unsubscribe_all(self):
        pass


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_infers_and_publishes(monkeypatch):
    added = []

    class DummyOntology:
        def add_triple(self, s, p, o):
            added.append((s, p, o))

        def infer_facts(self):
            return [("a", "b", "c")]

    service = ReasoningService(DummyNATS(), DummyJS(), ontology=DummyOntology())
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = {"triples": [["s", "p", "o"]], "input_id": "x"}
    msg = DummyMsg(json.dumps(payload))
    await service._handle_triples(msg)

    assert msg.acked
    assert added == [("s", "p", "o")]
    assert service._publisher.published
    subject, sent = service._publisher.published[0]
    assert subject == EventSubjects.MEMORY_RETRIEVED
    assert sent.retrieved_knowledge["facts"] == [("a", "b", "c")]
    assert sent.input_id == "x"
