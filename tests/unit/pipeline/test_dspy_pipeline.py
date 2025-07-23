import importlib.util
import sys
from types import SimpleNamespace

import types
import pytest

spec = importlib.util.spec_from_file_location(
    "deepthought.pipeline.dspy_pipeline",
    "src/deepthought/pipeline/dspy_pipeline.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None


def _setup_dummy_dspy(monkeypatch):
    class DummyLMFunction:
        def __init__(self, sig):
            self.sig = sig
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(answer="ok")

    dummy = types.SimpleNamespace(
        Signature=type("Signature", (), {}),
        InputField=lambda **k: None,
        OutputField=lambda **k: None,
        LMFunction=lambda sig: DummyLMFunction(sig),
    )
    monkeypatch.setitem(sys.modules, "dspy", dummy)
    return dummy


@pytest.mark.usefixtures("monkeypatch")
def test_build_qa_pipeline(monkeypatch):
    _setup_dummy_dspy(monkeypatch)
    spec.loader.exec_module(module)
    pipeline = module.build_qa_pipeline()
    assert pipeline("hi") == "ok"

