import importlib
import sys
import types
from unittest import mock

import pytest


def _install_stubs():
    datasets_mod = types.ModuleType("datasets")
    datasets_mod.Dataset = object
    datasets_mod.load_dataset = lambda *a, **k: None
    sys.modules["datasets"] = datasets_mod

    peft_mod = types.ModuleType("peft")
    peft_mod.LoraConfig = object
    peft_mod.get_peft_model = lambda *a, **k: None
    peft_mod.prepare_model_for_kbit_training = lambda *a, **k: None
    sys.modules["peft"] = peft_mod

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoModelForCausalLM = object
    transformers_mod.AutoTokenizer = object
    transformers_mod.BitsAndBytesConfig = object
    transformers_mod.DataCollatorForLanguageModeling = object
    transformers_mod.Trainer = object
    transformers_mod.TrainingArguments = object
    sys.modules["transformers"] = transformers_mod


_install_stubs()
train = importlib.import_module("deepthought.train")


def test_main_delegates_to_run_training(monkeypatch):
    dummy_run = mock.Mock(return_value=42)
    monkeypatch.setattr(train, "run_training", dummy_run)
    result = train.main(["--model-path", "mp", "--dataset-path", "ds"])
    dummy_run.assert_called_once()
    assert result == 42
