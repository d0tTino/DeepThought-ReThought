import importlib
import logging
import sys
import types
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from deepthought.eda.events import EventSubjects, MemoryRetrievedPayload


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
    def __init__(self, *args, **kwargs):
        pass

    async def subscribe(self, *args, **kwargs):
        pass

    async def unsubscribe_all(self):
        pass


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.nacked = True


class DummyTensor:
    def __init__(self, length):
        self._shape = (1, length)

    @property
    def shape(self):
        return self._shape


class DummyTokenizer:
    def __init__(self):
        self.prompt = ""

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def __call__(self, prompt, return_tensors=None):
        self.prompt = prompt
        return {"input_ids": DummyTensor(len(prompt.split()))}

    def decode(self, _data, skip_special_tokens=True):
        return self.prompt + " generated"


class DummyModel:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def generate(self, **kwargs):
        return [DummyTensor(1)]


class DummyPeftModel:
    @classmethod
    def from_pretrained(cls, base_model, adapter_dir):
        return cls()

    def merge_and_unload(self):
        return DummyModel()


def create_llm(monkeypatch):
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = DummyTokenizer
    tf.AutoModelForCausalLM = DummyModel
    torch_mod = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc, tb):
            pass

    def no_grad():
        return _NoGrad()

    torch_mod.no_grad = no_grad

    class SymBool:
        pass

    torch_mod.SymBool = SymBool

    autograd_mod = types.ModuleType("autograd")
    grad_mode_mod = types.ModuleType("grad_mode")

    def set_grad_enabled(_val: bool) -> None:  # noqa: D401 - simple placeholder
        """Dummy setter."""
        return None

    grad_mode_mod.set_grad_enabled = set_grad_enabled
    autograd_mod.grad_mode = grad_mode_mod
    torch_mod.autograd = autograd_mod

    peft_mod = types.ModuleType("peft")
    peft_mod.PeftModel = DummyPeftModel

    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "peft", peft_mod)

    import deepthought.modules.llm_base as llm_base
    import deepthought.modules.llm_prod as llm_prod

    importlib.reload(llm_base)
    importlib.reload(llm_prod)
    monkeypatch.setattr(llm_prod, "Publisher", DummyPublisher)
    monkeypatch.setattr(llm_prod, "Subscriber", DummySubscriber)
    monkeypatch.setattr(llm_prod.os.path, "isdir", lambda path: False)
    return llm_prod.ProductionLLM(DummyNATS(), DummyJS(), model_name="dummy", adapter_dir="dummy")


@pytest.mark.asyncio
async def test_handle_memory_event_non_dict(monkeypatch, caplog):
    llm = create_llm(monkeypatch)
    payload = MemoryRetrievedPayload(retrieved_knowledge=["x"], input_id="abc")
    msg = DummyMsg(payload.to_json())
    with caplog.at_level(logging.WARNING):
        await llm._handle_memory_event(msg)

    assert msg.nacked
    pub = llm._publisher
    assert not pub.published
    assert any("not a dict" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_handle_memory_event_missing_facts(monkeypatch, caplog):
    llm = create_llm(monkeypatch)
    payload = MemoryRetrievedPayload(retrieved_knowledge={}, input_id="p1")
    msg = DummyMsg(payload.to_json())
    with caplog.at_level(logging.ERROR):
        await llm._handle_memory_event(msg)

    assert msg.nacked
    pub = llm._publisher
    assert not pub.published
    assert any("missing facts" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_handle_memory_event_facts_not_list(monkeypatch, caplog):
    llm = create_llm(monkeypatch)
    payload = MemoryRetrievedPayload(retrieved_knowledge={"facts": "bad"}, input_id="p2")
    msg = DummyMsg(payload.to_json())
    with caplog.at_level(logging.ERROR):
        await llm._handle_memory_event(msg)

    assert msg.nacked
    pub = llm._publisher
    assert not pub.published
    assert any("missing facts" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_handle_memory_event_missing_input_id(monkeypatch):
    llm = create_llm(monkeypatch)
    payload = MemoryRetrievedPayload(retrieved_knowledge={"facts": ["x"]})
    msg = DummyMsg(payload.to_json())
    await llm._handle_memory_event(msg)

    assert msg.nacked
    pub = llm._publisher
    assert not pub.published


@pytest.mark.asyncio
async def test_start_listening_no_subscriber(monkeypatch, caplog):
    llm = create_llm(monkeypatch)
    llm._subscriber = None
    with caplog.at_level(logging.ERROR):
        result = await llm.start_listening()

    assert result is False
    assert any("Subscriber not initialized for ProductionLLM." in r.getMessage() for r in caplog.records)
