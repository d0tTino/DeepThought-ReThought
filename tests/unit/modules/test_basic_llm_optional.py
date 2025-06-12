import importlib
import sys
import types

import pytest
from deepthought import modules


def test_basic_llm_instantiation(monkeypatch):
    """Instantiate BasicLLM without requiring network access."""
    tf = types.ModuleType("transformers")

    class DummyTokenizer:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class DummyModel:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def generate(self, **kwargs):
            return []

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

    try:
        instance = llm_basic.BasicLLM(None, None, model_name="dummy")
    except ImportError:
        pytest.skip("BasicLLM dependencies not installed")
    assert instance is not None
