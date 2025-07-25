import importlib.machinery
import json
import sys
import types

pyperplan_mod = types.ModuleType("pyperplan")
pyperplan_mod.pddl = types.ModuleType("pyperplan.pddl")
pyperplan_parser = types.ModuleType("pyperplan.pddl.parser")


class DummyParser:
    def __init__(self, domain, problem):
        self.domInput = domain
        self.probInput = problem

    def parse_domain(self, read_from_file=False):
        return "domain"

    def parse_problem(self, domain, read_from_file=False):
        return "problem"


def dummy_ground(problem):
    return ["task"]


def dummy_search(task):
    return [types.SimpleNamespace(name="action")]


pyperplan_parser.Parser = DummyParser
pyperplan_mod.planner = types.ModuleType("pyperplan.planner")
pyperplan_mod.planner._ground = dummy_ground
pyperplan_mod.search = types.ModuleType("pyperplan.search")
pyperplan_mod.search.breadth_first_search = dummy_search
pyperplan_mod.pddl.parser = pyperplan_parser
sys.modules.setdefault("pyperplan", pyperplan_mod)
sys.modules.setdefault("pyperplan.pddl", pyperplan_mod.pddl)
sys.modules.setdefault("pyperplan.pddl.parser", pyperplan_parser)
sys.modules.setdefault("pyperplan.planner", pyperplan_mod.planner)
sys.modules.setdefault("pyperplan.search", pyperplan_mod.search)
rdflib_mod = types.ModuleType("rdflib")
rdflib_mod.Namespace = object
rdflib_mod.Graph = object
rdflib_mod.URIRef = str
rdflib_mod.namespace = types.SimpleNamespace(RDF=object())
sys.modules.setdefault("rdflib", rdflib_mod)
sys.modules.setdefault("rdflib.namespace", rdflib_mod.namespace)

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
fake_prom.REGISTRY = types.SimpleNamespace(_names_to_collectors={})
sys.modules.setdefault("prometheus_client", fake_prom)
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("aiosqlite", types.ModuleType("aiosqlite"))
torch_mod = types.ModuleType("torch")
torch_mod.no_grad = lambda: types.SimpleNamespace(
    __enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: None
)()
torch_mod.softmax = lambda t, dim=-1: t
sys.modules.setdefault("torch", torch_mod)
tb_mod = types.ModuleType("textblob")
tb_mod.TextBlob = lambda text: types.SimpleNamespace(
    sentiment=types.SimpleNamespace(polarity=0.0)
)
sys.modules.setdefault("textblob", tb_mod)
tf_mod = types.ModuleType("transformers")
tf_mod.AutoTokenizer = type(
    "AutoTokenizer", (), {"from_pretrained": classmethod(lambda cls, p: cls())}
)
tf_mod.AutoModelForSequenceClassification = type(
    "AutoModelForSequenceClassification",
    (),
    {
        "from_pretrained": classmethod(lambda cls, p: cls()),
        "__call__": lambda self, **k: types.SimpleNamespace(logits=[[0, 0, 0]]),
    },
)
sys.modules.setdefault("transformers", tf_mod)

fake_nats = types.ModuleType("nats")
fake_nats.__spec__ = importlib.machinery.ModuleSpec("nats", loader=None)
fake_nats.aio = types.ModuleType("aio")
client_mod = types.ModuleType("client")
setattr(client_mod, "Client", object)
fake_nats.aio.client = client_mod
msg_mod = types.ModuleType("msg")
setattr(msg_mod, "Msg", object)
fake_nats.aio.msg = msg_mod
fake_nats.js = types.ModuleType("js")
js_client_mod = types.ModuleType("client")
setattr(js_client_mod, "JetStreamContext", object)
fake_nats.js.client = js_client_mod
fake_nats.errors = types.ModuleType("errors")
setattr(fake_nats.errors, "Error", Exception)
sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", fake_nats.aio)
sys.modules.setdefault("nats.aio.client", client_mod)
sys.modules.setdefault("nats.aio.msg", msg_mod)
sys.modules.setdefault("nats.js", fake_nats.js)
sys.modules.setdefault("nats.js.client", js_client_mod)
sys.modules.setdefault("nats.errors", fake_nats.errors)

import deepthought.services.planning_service as planning_service
from deepthought.eda.events import (
    EventSubjects,
    MemoryRetrievedPayload,
    PlanGeneratedPayload,
    PlanRequestedPayload,
)
from deepthought.services.planning_service import PlanningService


class DummyPublisher:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload, **kw):
        self.published.append((subject, payload))


class DummySubscriber:
    async def subscribe(self, *a, **k):
        return True

    async def unsubscribe_all(self):
        pass


class DummyMsg:
    def __init__(self, payload):
        self.data = payload.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


import pytest


@pytest.mark.asyncio
async def test_memory_triggers_plan(tmp_path):
    cfg = tmp_path / "d.json"
    cfg.write_text(json.dumps({"desires": ["do it"]}), encoding="utf-8")

    service = PlanningService(None, None, desires_file=str(cfg))
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = MemoryRetrievedPayload(
        retrieved_knowledge={"facts": ["hi"]}, input_id="x"
    )
    msg = DummyMsg(payload.to_json())
    await service._handle_memory(msg)

    assert msg.acked
    assert service._publisher.published
    subj, out = service._publisher.published[0]
    assert subj == EventSubjects.PLAN_REQUESTED
    assert out.goal == "do it"
    assert out.input_id == "x"


@pytest.mark.asyncio
async def test_execute_plan():
    service = PlanningService(None, None)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = PlanGeneratedPayload(plan=["step1", "step2"], input_id="y")
    msg = DummyMsg(payload.to_json())
    await service._handle_plan(msg)

    assert msg.acked
    subjects = [s for s, _ in service._publisher.published]
    assert subjects == [EventSubjects.CHAT_RAW, EventSubjects.CHAT_RAW]


@pytest.mark.asyncio
async def test_plan_request_generates_plan(monkeypatch):
    class DummyTranslator:
        def translate(self, goal):
            return "d", "p"

    def dummy_plan(d, p):
        return ["ok"]

    monkeypatch.setattr(planning_service, "L2PTranslator", DummyTranslator)
    monkeypatch.setattr(planning_service, "plan", dummy_plan)

    service = PlanningService(None, None)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = PlanRequestedPayload(goal="do", input_id="z")
    msg = DummyMsg(payload.to_json())
    await service._handle_plan_request(msg)

    assert msg.acked
    assert service._publisher.published
    subj, out = service._publisher.published[0]
    assert subj == EventSubjects.PLAN_GENERATED
    assert out.plan == ["ok"]
    assert out.input_id == "z"


@pytest.mark.asyncio
async def test_planning_service_event_flow(monkeypatch, tmp_path):
    class DummyTranslator:
        def translate(self, goal):
            return "d", "p"

    monkeypatch.setattr(planning_service, "L2PTranslator", DummyTranslator)
    monkeypatch.setattr(planning_service, "plan", lambda d, p: ["a1", "a2"])

    cfg = tmp_path / "d.json"
    cfg.write_text(json.dumps({"desires": ["go"]}), encoding="utf-8")

    service = PlanningService(None, None, desires_file=str(cfg))
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    mem_payload = MemoryRetrievedPayload(
        retrieved_knowledge={"facts": ["f"]}, input_id="1"
    )
    await service._handle_memory(DummyMsg(mem_payload.to_json()))
    assert service._publisher.published
    subj, plan_req = service._publisher.published[-1]
    assert subj == EventSubjects.PLAN_REQUESTED

    await service._handle_plan_request(DummyMsg(plan_req.to_json()))
    subj2, plan_gen = service._publisher.published[-1]
    assert subj2 == EventSubjects.PLAN_GENERATED

    await service._handle_plan(DummyMsg(plan_gen.to_json()))
    subjects = [s for s, _ in service._publisher.published[-2:]]
    assert subjects == [EventSubjects.CHAT_RAW, EventSubjects.CHAT_RAW]
