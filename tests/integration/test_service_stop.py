import asyncio
import os
import sys
import types

sys.modules.setdefault("faiss", types.ModuleType("faiss"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("aiosqlite", types.ModuleType("aiosqlite"))
fake_nx = types.ModuleType("networkx")
setattr(fake_nx, "DiGraph", object)
sys.modules.setdefault("networkx", fake_nx)
fake_pyd = types.ModuleType("pydantic")
fake_pyd.AnyUrl = str
fake_pyd.ValidationError = Exception
sys.modules.setdefault("pydantic", fake_pyd)
fake_ps = types.ModuleType("pydantic_settings")
fake_ps.BaseSettings = object
fake_ps.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_ps)
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
fake_prom.REGISTRY = types.SimpleNamespace(_names_to_collectors={})
sys.modules.setdefault("prometheus_client", fake_prom)

import pytest

pytest.importorskip("nats")
from nats.aio.client import Client as NATS
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from deepthought.modules.memory_graph import GraphMemory
from deepthought.services.hierarchical_service import HierarchicalService
from deepthought.services.memory_service import MemoryService
from tests.helpers import nats_server_available

pytestmark = pytest.mark.nats

STREAM_NAME = "deepthought_events"


def get_nats_url() -> str:
    return os.getenv("NATS_URL", "nats://localhost:4222")


async def ensure_stream_exists(js):
    try:
        await js.stream_info(STREAM_NAME)
    except Exception:
        cfg = StreamConfig(
            name=STREAM_NAME,
            subjects=["dtr.>"],
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.MEMORY,
            max_msgs_per_subject=100,
            discard=DiscardPolicy.OLD,
        )
        await js.add_stream(cfg)


class DummyMemory:
    def store_interaction(self, text):
        pass

    def retrieve_context(self, prompt):
        return []

    def vector_matches(self, prompt):
        return []

    def graph_facts(self):
        return []


@pytest.mark.asyncio
async def test_memory_service_stop_closes_nats():
    if not nats_server_available(get_nats_url()):
        pytest.skip("NATS server not available")
    nc = await NATS().connect(servers=[get_nats_url()], connect_timeout=10)
    js = nc.jetstream(timeout=10.0)
    await ensure_stream_exists(js)

    service = MemoryService(nc, js, memory=DummyMemory())
    await service.start(durable_name="stop_mem_listener")
    await service.stop()

    assert not nc.is_connected


@pytest.mark.asyncio
async def test_hierarchical_service_stop_closes_nats():
    if not nats_server_available(get_nats_url()):
        pytest.skip("NATS server not available")
    nc = await NATS().connect(servers=[get_nats_url()], connect_timeout=10)
    js = nc.jetstream(timeout=10.0)
    await ensure_stream_exists(js)

    service = HierarchicalService(nc, js, DummyMemory())
    await service.start(durable_name="stop_hier_listener")
    await service.stop()

    assert not nc.is_connected


@pytest.mark.asyncio
async def test_graph_memory_stop_closes_nats(tmp_path):
    if not nats_server_available(get_nats_url()):
        pytest.skip("NATS server not available")
    nc = await NATS().connect(servers=[get_nats_url()], connect_timeout=10)
    js = nc.jetstream(timeout=10.0)
    await ensure_stream_exists(js)

    graph_file = tmp_path / "graph.json"
    service = GraphMemory(nc, js, memory=DummyMemory(), graph_file=str(graph_file))
    await service.start_listening(durable_name="stop_graph_listener")
    await service.stop_listening()

    assert not nc.is_connected
