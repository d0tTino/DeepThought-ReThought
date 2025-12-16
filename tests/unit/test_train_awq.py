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


class _DummyResult:
    def __init__(self, metrics):
        self.metrics = metrics


def test_run_training_invokes_awq(monkeypatch, tmp_path):
    trainer = mock.Mock()
    trainer.train.return_value = _DummyResult({"train_loss": 1.0})
    trainer.evaluate.return_value = {"eval_loss": 0.5, "eval_perplexity": 2.0}
    trainer.save_metrics = mock.Mock()
    trainer.save_model = mock.Mock()
    trainer.save_state = mock.Mock()

    monkeypatch.setattr(train, "load_model", mock.Mock(return_value=("model", "tokenizer")))
    monkeypatch.setattr(train, "load_dataset", mock.Mock(return_value=("train", "eval")))
    monkeypatch.setattr(train, "create_trainer", mock.Mock(return_value=(trainer, None)))

    awq_result = {"awq_output_dir": str(tmp_path / "awq"), "awq_quant_config": {"w_bit": 4}}
    monkeypatch.setattr(train, "_perform_awq_quantization", mock.Mock(return_value=awq_result))

    cfg = train.TrainingConfig(output_dir=str(tmp_path), awq=train.AWQConfig(enabled=True))

    exit_code = train.run_training(cfg)

    assert exit_code == 0
    train._perform_awq_quantization.assert_called_once_with(cfg, "tokenizer")
    summary_path = tmp_path / "metrics_summary.json"
    assert summary_path.exists()
    summary = summary_path.read_text()
    assert "awq_output_dir" in summary


def test_perform_awq_quantization_writes_results(monkeypatch, tmp_path):
    class DummyDataset:
        def __init__(self, size: int = 5):
            self.size = size
            self.selected = None

        def select(self, items):
            self.selected = list(items)
            return self

        def __len__(self):
            return self.size

    dummy_dataset = DummyDataset(size=10)

    class DummyAWQModel:
        last_instance = None

        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):
            inst = cls()
            inst.loaded = (model_dir, kwargs)
            cls.last_instance = inst
            return inst

        def quantize(self, tokenizer, quant_config, calib_data):
            self.quantize_call = (tokenizer, quant_config, calib_data)

        def save_quantized(self, output_dir):
            self.saved_dir = output_dir

    awq_mod = types.SimpleNamespace(AutoAWQForCausalLM=DummyAWQModel)
    monkeypatch.setitem(sys.modules, "awq", awq_mod)
    monkeypatch.setattr(train, "hf_load_dataset", lambda *a, **k: dummy_dataset)

    cfg = train.TrainingConfig(
        output_dir=str(tmp_path),
        dataset_path="dummy/calibration",
        awq=train.AWQConfig(
            enabled=True,
            calibration_samples=3,
            q_group_size=64,
            w_bit=3,
            zero_point=False,
            version="TEST",
        ),
    )

    tokenizer = object()
    result = train._perform_awq_quantization(cfg, tokenizer)

    assert result["awq_output_dir"] == str(tmp_path / "awq")
    assert result["awq_quant_config"] == {
        "zero_point": False,
        "q_group_size": 64,
        "w_bit": 3,
        "version": "TEST",
    }
    assert result["awq_calibration_samples"] == 3

    awq_results_path = tmp_path / "awq_results.json"
    assert awq_results_path.exists()

    instance = DummyAWQModel.last_instance
    assert instance.saved_dir == str(tmp_path / "awq")
    assert instance.quantize_call[0] is tokenizer
    assert isinstance(instance.quantize_call[2], DummyDataset)
    assert instance.quantize_call[2].selected == [0, 1, 2]
