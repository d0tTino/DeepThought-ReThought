import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import pytest


def test_train_script_run(monkeypatch, tmp_path):
    # provide lightweight stubs for optional heavy dependencies
    tf = types.ModuleType("transformers")

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def train(self, *args, **kwargs):
            return None

        def save_model(self):
            pass

        def save_state(self):
            pass

    class DummyTrainingArguments:
        def __init__(self, output_dir, **kwargs):
            self.output_dir = output_dir

    class DummyModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class DummyTokenizer:
        pad_token = "<pad>"
        eos_token = "</s>"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, text, truncation=True, max_length=2048, padding="max_length"):
            return {"input_ids": [0] * max_length}

    tf.AutoModelForCausalLM = DummyModel
    tf.AutoTokenizer = DummyTokenizer
    tf.BitsAndBytesConfig = lambda **kwargs: None
    tf.DataCollatorForLanguageModeling = lambda tokenizer, mlm=False: None
    tf.TrainingArguments = DummyTrainingArguments
    tf.Trainer = DummyTrainer
    monkeypatch.setitem(sys.modules, "transformers", tf)

    peft = types.ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: None
    peft.get_peft_model = lambda model, config: model
    peft.prepare_model_for_kbit_training = lambda model: model
    monkeypatch.setitem(sys.modules, "peft", peft)

    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = "bf16"
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    datasets_mod = types.ModuleType("datasets")
    datasets_mod.load_dataset = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    # import target modules after stubbing dependencies
    train = importlib.import_module("deepthought.train")
    ts = importlib.import_module("deepthought.train_script")
    monkeypatch.setattr(ts, "train_utils", train)

    def dummy_load_model(model_path, bits):
        return DummyModel(), DummyTokenizer()

    def dummy_load_dataset(path, tokenizer, max_seq_length=2048):
        with open(path, "r", encoding="utf-8") as fh:
            data = [json.loads(line) for line in fh]
        return data, data

    monkeypatch.setattr(train, "load_model", dummy_load_model)
    monkeypatch.setattr(train, "load_dataset", dummy_load_dataset)
    monkeypatch.setattr(tf.Trainer, "train", lambda self, **kw: None)

    dataset_path = str(Path(__file__).parent.parent / "data" / "dummy.json")
    args = argparse.Namespace(
        model_path=None,
        dataset_path=dataset_path,
        bits=4,
        output_dir=str(tmp_path),
        resume=False,
    )
    result = ts.run(args)
    assert result == 0
