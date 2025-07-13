import importlib
import logging
import sys
import types
from unittest import mock

import pytest

# Provide minimal stubs when optional deps are missing
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = "bf16"
    sys.modules["torch"] = torch_stub

if "datasets" not in sys.modules:
    ds_stub = types.ModuleType("datasets")
    ds_stub.Dataset = object
    ds_stub.load_dataset = lambda *a, **k: None
    sys.modules["datasets"] = ds_stub

if "peft" not in sys.modules:
    peft_stub = types.ModuleType("peft")
    peft_stub.LoraConfig = object
    peft_stub.get_peft_model = lambda *a, **k: None
    peft_stub.prepare_model_for_kbit_training = lambda m: m
    sys.modules["peft"] = peft_stub

if "transformers" not in sys.modules:
    tf_stub = types.ModuleType("transformers")

    class DummyAutoModel:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise RuntimeError("boom")

        @classmethod
        def from_config(cls, cfg):
            return cls()

    class DummyTokenizer:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    class DummyBitsAndBytesConfig:
        def __init__(self, **_kwargs):
            pass

    tf_stub.AutoModelForCausalLM = DummyAutoModel
    tf_stub.AutoTokenizer = DummyTokenizer
    tf_stub.BitsAndBytesConfig = DummyBitsAndBytesConfig
    tf_stub.GPT2Config = lambda **kwargs: types.SimpleNamespace(**kwargs)
    tf_stub.DataCollatorForLanguageModeling = object
    tf_stub.Trainer = object
    tf_stub.TrainingArguments = object
    sys.modules["transformers"] = tf_stub

train = importlib.import_module("deepthought.train")


def test_load_model_raises(monkeypatch, caplog):
    exc = RuntimeError("failed")
    monkeypatch.setattr(train.AutoModelForCausalLM, "from_pretrained", mock.Mock(side_effect=exc))
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as info:
            train.load_model("foo", 4, loader="hf")
    assert info.value is exc
    assert any("Failed to load model" in r.getMessage() for r in caplog.records)
