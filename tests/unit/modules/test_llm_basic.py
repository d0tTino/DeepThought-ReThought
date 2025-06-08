import importlib
import json
import sys
import types
from datetime import datetime, timezone
from types import SimpleNamespace
import pytest


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

    async def ack(self):
        self.acked = True


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
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    import deepthought.modules.llm_basic as llm_basic
    importlib.reload(llm_basic)
    monkeypatch.setattr(llm_basic, "Publisher", DummyPublisher)
    monkeypatch.setattr(llm_basic, "Subscriber", DummySubscriber)
    return llm_basic.BasicLLM(DummyNATS(), DummyJS(), model_name="dummy")


from deepthought.eda.events import EventSubjects, MemoryRetrievedPayload


@pytest.mark.asyncio
async def test_handle_memory_event(monkeypatch):
    llm = create_llm(monkeypatch)
    payload = MemoryRetrievedPayload(retrieved_knowledge={"facts": ["f1"]}, input_id="abc")
    msg = DummyMsg(payload.to_json())
    await llm._handle_memory_event(msg)

    assert msg.acked
    pub = llm._publisher
    assert pub.published
    subject, sent_payload = pub.published[0]
    assert subject == EventSubjects.RESPONSE_GENERATED
    assert sent_payload.input_id == "abc"
    assert sent_payload.final_response == "generated"
    ts = sent_payload.timestamp
    assert datetime.fromisoformat(ts).tzinfo == timezone.utc
