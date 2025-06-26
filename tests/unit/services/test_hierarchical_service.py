import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from deepthought.eda.events import EventSubjects, InputReceivedPayload
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


class DummyDALDump(DummyDAL):
    def query_subgraph(self, query, params):
        return [
            {"src": "a", "dst": "b", "rel": "NEXT", "src_id": 1, "dst_id": 2},
        ]


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
    service = HierarchicalService(DummyNATS(), DummyJS(), vec, dal)
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


def test_dump_graph_writes_dot(tmp_path):
    service = HierarchicalService(DummyNATS(), DummyJS(), None, DummyDALDump())
    out = service.dump_graph(tmp_path)
    with open(out, "r", encoding="utf-8") as f:
        data = f.read()
    assert "digraph" in data
    assert '"a" -> "b"' in data
