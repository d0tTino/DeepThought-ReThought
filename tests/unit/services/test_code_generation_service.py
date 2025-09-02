import json
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("aiosqlite")
pytest.importorskip("nats")
import importlib.machinery

fake_nx = types.ModuleType("networkx")
setattr(fake_nx, "DiGraph", object)
fake_nx.__spec__ = importlib.machinery.ModuleSpec("networkx", loader=None)
sys.modules.setdefault("networkx", fake_nx)
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

from deepthought.eda.events import CodeTemplatePayload, EventSubjects
from deepthought.services.code_generation_service import CodeGenerationService


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
async def test_handle_template_request(monkeypatch):
    service = CodeGenerationService(DummyNATS(), DummyJS())
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = CodeTemplatePayload(template="result = ${x} + ${y}", variables={"x": 1, "y": 2}, input_id="a")
    msg = DummyMsg(payload.to_json())
    await service._handle_template_request(msg)

    assert msg.acked
    pub = service._publisher
    assert pub.published
    subject, sent_payload = pub.published[0]
    assert subject == EventSubjects.CODE_GENERATED
    assert sent_payload.input_id == "a"
    assert sent_payload.result == "3"


@pytest.mark.asyncio
async def test_rejects_unsafe_code(monkeypatch):
    service = CodeGenerationService(DummyNATS(), DummyJS())
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = CodeTemplatePayload(template="result = __import__('os').system('echo hi')", variables={}, input_id="b")
    msg = DummyMsg(payload.to_json())
    await service._handle_template_request(msg)

    assert msg.acked
    pub = service._publisher
    assert not pub.published


@pytest.mark.asyncio
async def test_rejects_loops():
    service = CodeGenerationService(DummyNATS(), DummyJS())
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    template = "result = 0\nfor i in range(5):\n    result += i"
    payload = CodeTemplatePayload(template=template, variables={}, input_id="c")
    msg = DummyMsg(payload.to_json())
    await service._handle_template_request(msg)

    assert msg.acked
    assert not service._publisher.published


@pytest.mark.asyncio
async def test_rejects_large_constant():
    service = CodeGenerationService(DummyNATS(), DummyJS())
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    payload = CodeTemplatePayload(template="result = 1000001", variables={}, input_id="d")
    msg = DummyMsg(payload.to_json())
    await service._handle_template_request(msg)

    assert msg.acked
    assert not service._publisher.published


@pytest.mark.asyncio
async def test_execution_timeout(monkeypatch):
    service = CodeGenerationService(DummyNATS(), DummyJS())
    service._publisher = DummyPublisher()
    service._subscriber = DummySubscriber()

    def slow_eval(self, node, variables):
        import time

        time.sleep(0.2)
        return 1

    monkeypatch.setattr(CodeGenerationService, "_eval_expr", slow_eval)

    payload = CodeTemplatePayload(template="result = 1", variables={}, input_id="e")
    msg = DummyMsg(payload.to_json())
    await service._handle_template_request(msg)

    assert msg.acked
    assert not service._publisher.published
