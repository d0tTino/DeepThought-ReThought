import importlib
import logging
import os
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

    monkeypatch.setenv("SOCIAL_PERCEPTION_MODEL", "/tmp/model")
    monkeypatch.setattr("os.path.exists", lambda p: True)

    sp = importlib.import_module("deepthought.perception.social_perception")
    importlib.reload(sp)

    result = sp.analyze("hi")
    assert result == {
        "flirtation": 0.1,
        "avoidance": 0.2,
        "manipulation": 0.7,
    }


def test_env_model_path(monkeypatch):
    paths: dict[str, str] = {}

    tf_mod = types.ModuleType("transformers")

    class DummyTokenizer:
        @classmethod
        def from_pretrained(cls, path):
            paths["tokenizer"] = path
            return cls()

        def __call__(self, text, return_tensors=None):
            return {"text": text}

    class DummyModel:
        @classmethod
        def from_pretrained(cls, path):
            paths["model"] = path
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

    monkeypatch.setenv("SOCIAL_PERCEPTION_MODEL", "/tmp/custom-model")
    monkeypatch.setattr("os.path.exists", lambda p: True)

    sp = importlib.import_module("deepthought.perception.social_perception")
    importlib.reload(sp)

    sp.analyze("hello")

    assert paths["tokenizer"] == "/tmp/custom-model"
    assert paths["model"] == "/tmp/custom-model"


def test_missing_env_var(monkeypatch, caplog):
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

    monkeypatch.delenv("SOCIAL_PERCEPTION_MODEL", raising=False)
    monkeypatch.setattr("os.path.exists", lambda p: False)

    sp = importlib.import_module("deepthought.perception.social_perception")
    importlib.reload(sp)

    with caplog.at_level(logging.WARNING):
        result = sp.analyze("hello")

    neutral = 1.0 / len(sp.LABELS)
    assert result == {label: pytest.approx(neutral) for label in sp.LABELS}
    assert any("SOCIAL_PERCEPTION_MODEL" in r.getMessage() for r in caplog.records)


def test_missing_model_path(monkeypatch, caplog):
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

    monkeypatch.setenv("SOCIAL_PERCEPTION_MODEL", "/tmp/missing")
    monkeypatch.setattr("os.path.exists", lambda p: False)

    sp = importlib.import_module("deepthought.perception.social_perception")
    importlib.reload(sp)

    with caplog.at_level(logging.WARNING):
        result = sp.analyze("hello")

    neutral = 1.0 / len(sp.LABELS)
    assert result == {label: pytest.approx(neutral) for label in sp.LABELS}
    assert any("path not found" in r.getMessage() for r in caplog.records)
