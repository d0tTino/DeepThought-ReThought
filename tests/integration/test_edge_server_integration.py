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


@pytest.mark.asyncio
async def test_llm_session_closed(monkeypatch):
    pytest.importorskip("aiohttp")
    from deepthought.modules import llm_remote as llm_mod

    class DummyNATS:
        pass

    class DummyJS:
        pass

    class DummyPublisher:
        async def publish(self, *a, **k):
            pass

    class DummySubscriber:
        async def unsubscribe_all(self):
            pass

    class DummyResp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            pass

        async def json(self):
            return {"text": "ok"}

    class DummySession:
        def __init__(self):
            self.closed = False

        def post(self, *_args, **_kwargs):
            return DummyResp()

        async def close(self):
            self.closed = True

    session = DummySession()
    monkeypatch.setattr(llm_mod, "Publisher", lambda *a, **k: DummyPublisher())
    monkeypatch.setattr(llm_mod, "Subscriber", lambda *a, **k: DummySubscriber())
    monkeypatch.setattr(llm_mod.aiohttp, "ClientSession", lambda: session)

    llm = llm_mod.RemoteLLM(DummyNATS(), DummyJS(), endpoint="http://api")

    await llm._generate("hi")
    await llm.stop_listening()

    assert session.closed


@pytest.mark.asyncio
async def test_llm_requires_endpoint(monkeypatch):
    pytest.importorskip("aiohttp")
    from deepthought.modules import llm_remote as llm_mod

    class DummyNATS:
        pass

    class DummyJS:
        pass

    monkeypatch.setenv("LLM_ENDPOINT", "")
    with pytest.raises(ValueError):
        llm_mod.RemoteLLM(DummyNATS(), DummyJS())

    with pytest.raises(ValueError):
        llm_mod.RemoteLLM(DummyNATS(), DummyJS(), endpoint="")
