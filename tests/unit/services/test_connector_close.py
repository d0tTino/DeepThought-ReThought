import types

import pytest

# Import memory_service tests for dummy modules and classes
import tests.unit.services.test_memory_service as tm
from deepthought.config import Settings
from deepthought.services.hierarchical_service import HierarchicalService
from deepthought.services.knowledge_graph_service import KnowledgeGraphService
from deepthought.services.memory_service import MemoryService


class DummyConnector:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class DummyMemory:
    def __init__(self, connector):
        backend = types.SimpleNamespace(_dal=types.SimpleNamespace(_connector=connector))
        self.graph_backend = backend

    def store_interaction(self, text):
        pass

    def retrieve_context(self, prompt):
        return []


class DummyNATS(tm.DummyNATS):
    async def drain(self):
        pass


class DummyJS(tm.DummyJS):
    pass


class DummySubscriber(tm.DummySubscriber):
    pass


class DummyPublisher(tm.DummyPublisher):
    pass


@pytest.mark.asyncio
async def test_knowledge_graph_service_stop_closes_connector():
    connector = DummyConnector()
    memory = DummyMemory(connector)
    service = KnowledgeGraphService(DummyNATS(), DummyJS(), Settings(), memory)
    service._subscriber = DummySubscriber()
    service._publisher = DummyPublisher()
    await service.stop()
    assert connector.closed


@pytest.mark.asyncio
async def test_memory_service_stop_closes_connector():
    connector = DummyConnector()
    memory = DummyMemory(connector)
    service = MemoryService(DummyNATS(), DummyJS(), Settings(), memory)
    service._subscriber = DummySubscriber()
    service._publisher = DummyPublisher()
    await service.stop()
    assert connector.closed


@pytest.mark.asyncio
async def test_hierarchical_service_stop_closes_connector():
    connector = DummyConnector()
    memory = DummyMemory(connector)
    service = HierarchicalService(DummyNATS(), DummyJS(), Settings(), memory)
    service._subscriber = DummySubscriber()
    service._publisher = DummyPublisher()
    await service.stop()
    assert connector.closed
