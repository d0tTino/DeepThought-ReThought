import importlib
import sys
from importlib.metadata import EntryPoint, EntryPoints
from types import ModuleType

import deepthought.train as train


def test_custom_model_loader(monkeypatch):
    calls = {}

    def loader(mp, bits):
        calls["args"] = (mp, bits)
        return "m", "t"

    mod = ModuleType("dummy_mod")
    mod.loader = loader
    sys.modules["dummy_mod"] = mod
    ep = EntryPoint(name="dummy", value="dummy_mod:loader", group="dtrt.model_loaders")
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: EntryPoints([ep]))

    model, tok = train.load_model("foo", 4, loader="dummy")
    assert (model, tok) == ("m", "t")
    assert calls["args"] == ("foo", 4)
