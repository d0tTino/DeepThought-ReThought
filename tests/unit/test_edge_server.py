import importlib
import sys
import types
from pathlib import Path

import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


class DummyTokenizer:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def __call__(self, _text: str, return_tensors: str = "pt"):
        class DummyInputs(dict):
            def to(self, _device: str):
                return self

        return DummyInputs({"input_ids": [0]})

    def decode(self, _ids, skip_special_tokens: bool = True) -> str:
        return "dummy generated"


class DummyModel:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    device = "cpu"

    def generate(self, **_kwargs):
        return [[1, 2, 3]]


def test_generate_endpoint(monkeypatch):
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = DummyTokenizer
    tf.AutoModelForCausalLM = DummyModel

    torch_mod = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc, tb):
            pass

    torch_mod.no_grad = lambda: _NoGrad()

    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import edge_server

    importlib.reload(edge_server)

    client = TestClient(edge_server.app)
    response = client.post("/generate", json={"text": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["text"], str)
    assert data["text"]
