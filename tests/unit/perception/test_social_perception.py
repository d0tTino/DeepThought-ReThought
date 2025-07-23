import importlib
import sys
import types

import pytest


@pytest.mark.usefixtures("monkeypatch")
def test_analyze_returns_probs(monkeypatch):
    tf_mod = types.ModuleType("transformers")

    class DummyTokenizer:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def __call__(self, text, return_tensors=None):
            return {"text": text}

    class DummyModel:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def __call__(self, **kwargs):
            return types.SimpleNamespace(logits=[[1.0, 2.0, 3.0]])

    tf_mod.AutoTokenizer = DummyTokenizer
    tf_mod.AutoModelForSequenceClassification = DummyModel
    monkeypatch.setitem(sys.modules, "transformers", tf_mod)

    torch_mod = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc, tb):
            pass

    torch_mod.no_grad = lambda: _NoGrad()

    class DummyTensor(list):
        def tolist(self):
            return list(self)

    torch_mod.softmax = lambda t, dim=0: [DummyTensor([0.1, 0.2, 0.7])]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    sp = importlib.import_module("deepthought.perception.social_perception")
    importlib.reload(sp)

    result = sp.analyze("hi")
    assert result == {
        "flirtation": 0.1,
        "avoidance": 0.2,
        "manipulation": 0.7,
    }
