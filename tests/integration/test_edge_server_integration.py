import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.mark.slow
def test_generate_with_distilgpt2(monkeypatch):
    pytest.importorskip("torch")
    _tf = pytest.importorskip("transformers")


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
