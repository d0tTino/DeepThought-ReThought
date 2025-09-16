import pytest

pytest.importorskip("nats")

from nats.aio.client import Client as NATS
from nats.js.api import (
    ConsumerConfig,
    DeliverPolicy,
    DiscardPolicy,
    RetentionPolicy,
    StorageType,
    StreamConfig,
)

from deepthought.eda.events import (
    EventSubjects,
    ModalityEmbeddings,
    PerceptionEmbeddingsEvent,
    PerceptionEmbeddingsPayload,
)
from deepthought.services.cognitive_core_service import CognitiveCoreService
from tests.helpers import nats_server_available
from types import SimpleNamespace

pytest_plugins = ["tests.helpers"]


pytestmark = pytest.mark.nats


class DummyStore:
    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}

    def upsert_vectors(self, vecs, ids):
        for vec, vid in zip(vecs, ids):
            self.vectors[vid] = vec


class DummyGraph:
    def __init__(self) -> None:
        self.nodes: set[str] = set()

    def query_subgraph(self, query: str, params: dict):
        self.nodes.add(params["sid"])
        return []


class DummyMemory:
    def __init__(self) -> None:
        self._store = DummyStore()
        self.graph_backend = DummyGraph()


@pytest.mark.asyncio
async def test_perception_replay_idempotent(nats_server):
    if not nats_server_available(nats_server):
        pytest.skip("NATS server not available")

    nc = await NATS().connect(servers=[nats_server])
    js = nc.jetstream()

    cfg = StreamConfig(
        name="deepthought_events",
        subjects=["dtr.>"],
        retention=RetentionPolicy.LIMITS,
        storage=StorageType.MEMORY,
        max_msgs_per_subject=100,
        discard=DiscardPolicy.OLD,
    )
    try:
        await js.add_stream(cfg)
    except Exception:
        pass

    settings = SimpleNamespace(
        search_db=None,
        memory_top_k=1,
        wandb_enabled=False,
        wandb_project=None,
        wandb_sweep_id=None,
    )
    service = CognitiveCoreService(
        nc,
        js,
        settings=settings,
        memory=DummyMemory(),
        db=SimpleNamespace(),
    )

    payload = PerceptionEmbeddingsPayload(
        message_id="m1",
        user_id="u1",
        fused=[[0.1, 0.2]],
        by_modality={
            "text": ModalityEmbeddings(spans=[[0, 1]], embeddings=[[0.1, 0.2]], encoders=[])
        },
    )
    event = PerceptionEmbeddingsEvent(payload=payload)
    await js.publish(EventSubjects.PERCEPTION_EMBEDDINGS, event.to_json().encode())

    sub = await js.pull_subscribe(
        EventSubjects.PERCEPTION_EMBEDDINGS,
        durable="perception-listener",
        stream="deepthought_events",
        config=ConsumerConfig(deliver_policy=DeliverPolicy.ALL),
    )
    msgs = await sub.fetch(1, timeout=1)
    await service._handle_embeddings(msgs[0])
    await sub.unsubscribe()

    assert len(service._memory._store.vectors) == 1
    assert len(service._memory.graph_backend.nodes) == 1

    sub2 = await js.pull_subscribe(
        EventSubjects.PERCEPTION_EMBEDDINGS,
        durable="perception-replay",
        stream="deepthought_events",
        config=ConsumerConfig(deliver_policy=DeliverPolicy.ALL),
    )
    msgs = await sub2.fetch(1, timeout=1)
    await service._handle_embeddings(msgs[0])
    await sub2.unsubscribe()

    assert len(service._memory._store.vectors) == 1
    assert len(service._memory.graph_backend.nodes) == 1

    await nc.drain()
