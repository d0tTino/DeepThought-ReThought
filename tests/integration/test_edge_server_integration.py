import importlib
import sys
import types
from pathlib import Path

import pytest

from tests.unit.test_edge_server import DummyModel, DummyTokenizer

pytest.importorskip("fastapi")
import requests
from fastapi.testclient import TestClient


@pytest.mark.slow
def test_generate_with_distilgpt2(monkeypatch):
    pytest.importorskip("torch")
    _tf = pytest.importorskip("transformers")

    try:  # pragma: no cover - network check
        requests.head("https://huggingface.co", timeout=5)
    except Exception:
        pytest.skip("huggingface.co not reachable")

    real_load = _tf.AutoModelForCausalLM.from_pretrained

    def load_model(model_name, *args, **kwargs):
        kwargs.pop("load_in_4bit", None)
        return real_load(model_name, *args, **kwargs)

    monkeypatch.setattr(_tf.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setenv("MODEL_PATH", "distilgpt2")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import edge_server

    importlib.reload(edge_server)

    with TestClient(edge_server.app) as client:
        resp = client.post("/generate", json={"text": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data.get("text"), str)
        assert data["text"]


def test_generate_unreachable(monkeypatch):
    """Edge server returns 500 when model inference fails."""
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

    def fail_generate(**_kwargs):
        raise requests.ConnectionError

    monkeypatch.setattr(edge_server.model, "generate", fail_generate)

    with TestClient(edge_server.app, raise_server_exceptions=False) as client:
        resp = client.post("/generate", json={"text": "Hello"})
        assert resp.status_code == 500
