import pytest
pytest.importorskip("nats")

from deepthought.eda.events import EventSubjects, InputReceivedPayload
from deepthought.services.social_graph_service import SocialGraphService


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
        return type("Ack", (), {"seq": 1, "stream": "test"})()


class DummySubscriber:
    async def subscribe(self, *args, **kwargs):
        pass

    async def unsubscribe_all(self):
        pass


class DummyDB:
    def __init__(self):
        self.memories = []
        self.interactions = 0

    async def store_memory(self, user_id, memory, topic="", sentiment_score=None):
        self.memories.append(memory)

    async def log_interaction(self, user_id, target_id=None, sentiment_score=None):
        self.interactions += 1

    async def recall_user(self, user_id):
        return [("", m) for m in self.memories]


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_handle_input_publishes_social_context(monkeypatch):
    db = DummyDB()
    monkeypatch.setattr(
        SocialGraphService,
        "_publisher",
        DummyPublisher(DummyNATS(), DummyJS()),
        raising=False,
    )
    monkeypatch.setattr(SocialGraphService, "_subscriber", DummySubscriber(), raising=False)
    service = SocialGraphService(DummyNATS(), DummyJS(), db)
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = InputReceivedPayload(user_input="hello", input_id="x")
    msg = DummyMsg(payload.to_json())
    await service._handle_input(msg)

    assert msg.acked
    assert db.memories == ["hello"]
    subject, sent_payload = service._publisher.published[0]
    assert subject == EventSubjects.MEMORY_RETRIEVED
    assert sent_payload.input_id == "x"
    assert "hello" in sent_payload.retrieved_knowledge["facts"]
