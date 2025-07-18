import importlib
import sys
from importlib.metadata import EntryPoint, EntryPoints
from types import ModuleType


def _load_train_module():
    sys.modules.setdefault("torch", ModuleType("torch"))

    datasets_mod = ModuleType("datasets")
    datasets_mod.Dataset = object
    datasets_mod.load_dataset = lambda *a, **k: None
    sys.modules["datasets"] = datasets_mod

    peft_mod = ModuleType("peft")
    peft_mod.LoraConfig = object
    peft_mod.get_peft_model = lambda *a, **k: None
    peft_mod.prepare_model_for_kbit_training = lambda *a, **k: None
    sys.modules["peft"] = peft_mod

    transformers_mod = ModuleType("transformers")
    for cls in [
        "AutoModelForCausalLM",
        "AutoTokenizer",
        "BitsAndBytesConfig",
        "DataCollatorForLanguageModeling",
        "Trainer",
        "TrainingArguments",
    ]:
        setattr(transformers_mod, cls, type(cls, (), {}))
    sys.modules["transformers"] = transformers_mod

    return importlib.import_module("deepthought.train")


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

    train = _load_train_module()
    model, tok = train.load_model("foo", 4, loader="dummy")
    assert (model, tok) == ("m", "t")
    assert calls["args"] == ("foo", 4)


def test_plugin_fallback(monkeypatch):
    ep = EntryPoint(name="dummy", value="dummy_mod:loader", group="dtrt.model_loaders")

    def dummy_entry_points():
        return {"dtrt.model_loaders": [ep]}

    monkeypatch.setattr(importlib.metadata, "entry_points", dummy_entry_points)
    mod = ModuleType("dummy_mod")
    mod.loader = lambda *a, **k: ("m", "t")
    sys.modules["dummy_mod"] = mod

    train = _load_train_module()
    model, tok = train.load_model("foo", 4, loader="dummy")
    assert (model, tok) == ("m", "t")
