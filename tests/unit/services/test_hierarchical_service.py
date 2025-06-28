import sys
import types
import json
from datetime import datetime, timezone
from types import SimpleNamespace

sys.modules.setdefault("deepthought.harness", types.ModuleType("harness"))
sys.modules.setdefault("deepthought.learn", types.ModuleType("learn"))
sys.modules.setdefault("deepthought.modules", types.ModuleType("modules"))
sys.modules.setdefault("deepthought.motivate", types.ModuleType("motivate"))

import pytest

from deepthought.eda.events import EventSubjects, InputReceivedPayload
from deepthought.memory.tiered import TieredMemory
from deepthought.services.hierarchical_service import HierarchicalService


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


class DummyVector:
    def query(self, query_texts, n_results=3):
        return {"documents": [["vec1"], ["vec2"]]}


class DummyDAL:
    def query_subgraph(self, query, params):
        return [{"fact": "graph1"}]


class FailingVector:
    def query(self, *args, **kwargs):
        raise RuntimeError("boom")


class FailingDAL:
    def query_subgraph(self, *args, **kwargs):
        raise RuntimeError("boom")


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_handle_input_publishes_combined_context(monkeypatch):
    vec = DummyVector()
    dal = DummyDAL()
    memory = TieredMemory(vec, dal, top_k=3)
    service = HierarchicalService(DummyNATS(), DummyJS(), memory)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = InputReceivedPayload(user_input="hello", input_id="x")
    msg = DummyMsg(payload.to_json())
    await service._handle_input(msg)

    assert msg.acked
    assert service._publisher.published
    subject, sent_payload = service._publisher.published[0]
    assert subject == EventSubjects.MEMORY_RETRIEVED
    assert sent_payload.input_id == "x"
    facts = sent_payload.retrieved_knowledge["retrieved_knowledge"]["facts"]
    assert "vec1" in facts and "graph1" in facts
    ts = sent_payload.timestamp
    assert datetime.fromisoformat(ts).tzinfo == timezone.utc


def test_retrieve_context_merges():
    vec = DummyVector()
    dal = DummyDAL()
    memory = TieredMemory(vec, dal, top_k=3)
    service = HierarchicalService(DummyNATS(), DummyJS(), memory)
    ctx = service.retrieve_context("hi")
    assert ctx == ["vec1", "vec2", "graph1"]


def test_retrieve_context_failures():
    memory = TieredMemory(FailingVector(), FailingDAL(), top_k=3)
    service = HierarchicalService(DummyNATS(), DummyJS(), memory)
    ctx = service.retrieve_context("x")
    assert ctx == []


class DummyGraphDAL:
    def query_subgraph(self, query, params):
        return [
            {"src_id": 1, "src": "A", "rel": "KNOWS", "dst_id": 2, "dst": "B"},
            {"src_id": 2, "src": "B", "rel": "LIKES", "dst_id": 3, "dst": "C"},
        ]


class DummyMemory:
    def __init__(self, dal):
        self._dal = dal


def test_dump_graph(tmp_path):
    dal = DummyGraphDAL()
    memory = DummyMemory(dal)
    service = HierarchicalService(DummyNATS(), DummyJS(), memory)

    dot_file = service.dump_graph(str(tmp_path))

    assert dot_file == str(tmp_path / "graph.dot")
    contents = (tmp_path / "graph.dot").read_text()
    assert '"A" -> "B" [label="KNOWS"]' in contents
    assert '"B" -> "C" [label="LIKES"]' in contents
