import importlib
import sys
import types

import pytest
import torch

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

train = importlib.import_module("deepthought.train")
compute_metrics = train.compute_metrics
_prepare_metrics = train._prepare_metrics
_get_eval_metric = train._get_eval_metric


def test_compute_metrics_returns_neutral_keys():
    logits = torch.zeros((1, 3, 5))
    labels = torch.tensor([[1, 2, 3]])

    metrics = compute_metrics((logits, labels))

    assert set(metrics) == {"loss", "perplexity"}
    assert metrics["perplexity"] == pytest.approx(torch.exp(torch.tensor(metrics["loss"])).item())


def test_prepare_metrics_serializes_tensors_and_nested_objects():
    metrics = {
        "eval_loss": torch.tensor(1.23),
        "eval_perplexity": torch.tensor(3.0),
        "nested": {"a": 1},
    }

    prepared = _prepare_metrics(metrics)

    assert prepared["eval_loss"] == pytest.approx(1.23)
    assert prepared["eval_perplexity"] == pytest.approx(3.0)
    assert isinstance(prepared["nested"], str)


def test_get_eval_metric_prefers_prefixed_keys():
    metrics = {"eval_loss": 0.5, "loss": 1.0}

    assert _get_eval_metric(metrics, "loss") == 0.5
    assert _get_eval_metric({"loss": 1.0}, "loss") == 1.0
