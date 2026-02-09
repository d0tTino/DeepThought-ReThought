import importlib.machinery
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("aiosqlite")

fake_pyd = types.ModuleType("pydantic")
fake_pyd.AnyUrl = str
fake_pyd.ValidationError = Exception
fake_pyd.Field = lambda default=None, **kwargs: default
fake_pyd.__spec__ = importlib.machinery.ModuleSpec("pydantic", loader=None)
sys.modules.setdefault("pydantic", fake_pyd)
fake_ps = types.ModuleType("pydantic_settings")


fake_prom = types.ModuleType("prometheus_client")


class _Metric:
    def labels(self, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass


fake_prom.Counter = lambda *a, **k: _Metric()
fake_prom.Histogram = lambda *a, **k: _Metric()
fake_prom.REGISTRY = SimpleNamespace(_names_to_collectors={})
sys.modules.setdefault("prometheus_client", fake_prom)
sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
torch_mod = types.ModuleType("torch")


class _NoGrad:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc, tb):
        pass


torch_mod.no_grad = lambda: _NoGrad()
torch_mod.softmax = lambda t, dim=-1: t
sys.modules.setdefault("torch", torch_mod)
tb_mod = types.ModuleType("textblob")
tb_mod.TextBlob = lambda text: types.SimpleNamespace(sentiment=types.SimpleNamespace(polarity=0.0))
sys.modules.setdefault("textblob", tb_mod)
tf_mod = types.ModuleType("transformers")
tf_mod.AutoTokenizer = type("AutoTokenizer", (), {"from_pretrained": classmethod(lambda cls, p: cls())})
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
err_mod = types.ModuleType("errors")
setattr(err_mod, "Error", Exception)
setattr(err_mod, "TimeoutError", Exception)
fake_nats.errors = err_mod
sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", fake_nats.aio)
sys.modules.setdefault("nats.aio.client", client_mod)
sys.modules.setdefault("nats.aio.msg", msg_mod)
sys.modules.setdefault("nats.js", fake_nats.js)
sys.modules.setdefault("nats.js.client", js_client_mod)
sys.modules.setdefault("nats.errors", err_mod)

pp_mod = types.ModuleType("pyperplan")
pp_pddl = types.ModuleType("pddl")
pp_parser = types.ModuleType("parser")
setattr(pp_parser, "Parser", object)
pp_pddl.parser = pp_parser
pp_mod.pddl = pp_pddl
pp_planner = types.ModuleType("planner")
setattr(pp_planner, "_ground", lambda *a, **k: None)
pp_mod.planner = pp_planner
pp_search = types.ModuleType("search")
setattr(pp_search, "breadth_first_search", lambda *a, **k: None)
pp_mod.search = pp_search
sys.modules.setdefault("pyperplan", pp_mod)
sys.modules.setdefault("pyperplan.pddl", pp_pddl)
sys.modules.setdefault("pyperplan.pddl.parser", pp_parser)
sys.modules.setdefault("pyperplan.planner", pp_planner)
sys.modules.setdefault("pyperplan.search", pp_search)

planning_stub = types.ModuleType("planning_service")
setattr(planning_stub, "PlanningService", object)
reasoning_stub = types.ModuleType("reasoning_service")
setattr(reasoning_stub, "ReasoningService", object)
sys.modules.setdefault("deepthought.services.planning_service", planning_stub)
sys.modules.setdefault("deepthought.services.reasoning_service", reasoning_stub)


class DummyBase:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


fake_ps.BaseSettings = DummyBase
fake_ps.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_ps)

import importlib.util
import json
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[3] / "src"
deep_pkg = types.ModuleType("deepthought")
deep_pkg.__path__ = [str(SRC / "deepthought")]
deep_pkg.__spec__ = importlib.machinery.ModuleSpec("deepthought", loader=None, is_package=True)
services_pkg = types.ModuleType("deepthought.services")
services_pkg.__path__ = [str(SRC / "deepthought" / "services")]
services_pkg.__spec__ = importlib.machinery.ModuleSpec("deepthought.services", loader=None, is_package=True)
sys.modules.setdefault("deepthought", deep_pkg)
sys.modules.setdefault("deepthought.services", services_pkg)
spec = importlib.util.spec_from_file_location(
    "deepthought.services.cognitive_core_service", SRC / "deepthought/services/cognitive_core_service.py"
)
cognitive_core_service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cognitive_core_service)
sys.modules.setdefault("deepthought.services.cognitive_core_service", cognitive_core_service)
from deepthought.config import Settings
from deepthought.eda.events import (
    EventSubjects,
    InputReceivedPayload,
    ModalityEmbeddings,
    PerceptionEmbeddingsEvent,
    PerceptionEmbeddingsPayload,
)
from deepthought.services.cognitive_core_service import CognitiveCoreService


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


class DummyMemory:
    def __init__(self):
        self.interactions = []

    def store_interaction(self, text):
        self.interactions.append(text)

    def retrieve_context(self, prompt: str):
        return self.interactions[-3:]


class DummyDB:
    def __init__(self):
        self.memories = []
        self.memory_user_ids = []
        self.interactions = 0
        self.interaction_user_ids = []
        self.affinity = 0.0
        self.affinity_user_ids = []
        self.perceptions = []

    async def store_memory(self, user_id, memory, topic="", sentiment_score=None):
        self.memory_user_ids.append(user_id)
        if topic == "social_perception":
            self.perceptions.append(json.loads(memory))
        else:
            self.memories.append(memory)

    async def log_interaction(self, user_id, target_id=None, sentiment_score=None):
        self.interaction_user_ids.append(user_id)
        self.interactions += 1

    async def adjust_affinity(self, user_id, delta):
        self.affinity_user_ids.append(user_id)
        self.affinity += delta

    async def recall_user(self, user_id, limit=None):
        return [("", m) for m in self.memories]

    async def close(self):
        pass


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_handle_input_stores_and_publishes(monkeypatch):
    memory = DummyMemory()
    db = DummyDB()
    monkeypatch.setattr(
        cognitive_core_service,
        "analyze_social",
        lambda text: {"flirtation": 0.2, "avoidance": 0.1, "manipulation": 0.0},
    )
    monkeypatch.setattr(
        CognitiveCoreService,
        "_publisher",
        DummyPublisher(DummyNATS(), DummyJS()),
        raising=False,
    )
    monkeypatch.setattr(CognitiveCoreService, "_subscriber", DummySubscriber(), raising=False)
    settings = Settings()
    service = CognitiveCoreService(DummyNATS(), DummyJS(), settings, memory=memory, db=db)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = InputReceivedPayload(user_input="hello", input_id="x")
    msg = DummyMsg(payload.to_json())
    await service._handle_input(msg)

    assert msg.acked
    assert memory.interactions == ["hello"]
    assert db.memories == ["hello"]
    assert db.perceptions
    perception = db.perceptions[0]
    assert perception.get("flirtation") == pytest.approx(0.2)
    assert "avoidance" in perception
    assert "manipulation" in perception
    expected_delta = perception.get("flirtation", 0.0) - (
        perception.get("avoidance", 0.0) + perception.get("manipulation", 0.0)
    )
    assert db.affinity == pytest.approx(expected_delta)
    subject, sent_payload = service._publisher.published[0]
    assert subject == EventSubjects.MEMORY_RETRIEVED
    assert sent_payload.input_id == "x"
    assert "hello" in sent_payload.retrieved_knowledge["facts"]


@pytest.mark.asyncio
async def test_handle_input_prefers_payload_user_id_over_header(monkeypatch):
    memory = DummyMemory()
    db = DummyDB()
    monkeypatch.setattr(
        cognitive_core_service,
        "analyze_social",
        lambda text: {"flirtation": 0.0, "avoidance": 0.0, "manipulation": 0.0},
    )
    service = CognitiveCoreService(DummyNATS(), DummyJS(), Settings(), memory=memory, db=db)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    msg = DummyMsg(json.dumps({"user_input": "hello", "input_id": "x", "user_id": "payload-user"}))
    msg.headers = {"user_id": "header-user"}

    await service._handle_input(msg)

    assert msg.acked
    assert db.memory_user_ids == ["payload-user", "payload-user"]
    assert db.interaction_user_ids == ["payload-user"]
    assert db.affinity_user_ids == ["payload-user"]


@pytest.mark.asyncio
async def test_handle_input_uses_header_user_id_then_anonymous(monkeypatch):
    memory = DummyMemory()
    db = DummyDB()
    monkeypatch.setattr(
        cognitive_core_service,
        "analyze_social",
        lambda text: {"flirtation": 0.0, "avoidance": 0.0, "manipulation": 0.0},
    )
    service = CognitiveCoreService(DummyNATS(), DummyJS(), Settings(), memory=memory, db=db)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload_with_header = InputReceivedPayload(user_input="hello", input_id="x")
    msg_with_header = DummyMsg(payload_with_header.to_json())
    msg_with_header.headers = {"user_id": "header-user"}
    await service._handle_input(msg_with_header)

    payload_without_user = InputReceivedPayload(user_input="hi", input_id="y")
    msg_without_user = DummyMsg(payload_without_user.to_json())
    await service._handle_input(msg_without_user)

    assert msg_with_header.acked
    assert msg_without_user.acked
    assert db.interaction_user_ids == ["header-user", "anonymous"]
    assert db.affinity_user_ids == ["header-user", "anonymous"]


class DummyStore:
    def __init__(self) -> None:
        self.upserts: list = []

    def upsert_vectors(self, vectors, ids, metadatas=None):
        self.upserts.append((list(vectors), list(ids)))


class DummyGraph:
    def __init__(self) -> None:
        self.queries: list = []

    def query_subgraph(self, query, params):
        self.queries.append((query, params))

    def merge_entity(self, name):
        pass


class DummyMem2:
    def __init__(self):
        self._store = DummyStore()
        self.graph_backend = DummyGraph()


@pytest.mark.asyncio
async def test_handle_embeddings_upserts(monkeypatch):
    memory = DummyMem2()
    db = DummyDB()
    service = CognitiveCoreService(DummyNATS(), DummyJS(), Settings(), memory=memory, db=db)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()
    payload = PerceptionEmbeddingsPayload(
        message_id="m1",
        user_id="u",
        fused=[[0.1, 0.2]],
        spans=[],
        modality_mask={},
        by_modality={},

    )
    msg = DummyMsg(payload.to_json())
    await service._handle_embeddings(msg)
    assert msg.acked
    assert memory._store.upserts == [([[0.1, 0.2]], ["m1:0"])]
    assert len(memory.graph_backend.queries) == 0
