import builtins
import importlib
import json
import sys
import types
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict


class _TorchStub(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.bfloat16 = "bfloat16"
        self.float16 = "float16"
        self.float32 = "float32"
        self.Tensor = type("Tensor", (), {})
        self.Generator = type("Generator", (), {})
        self.nn = SimpleNamespace(Module=type("Module", (), {}))

    def from_numpy(self, array):
        return array


class _TransformersStub(types.ModuleType):
    class PreTrainedTokenizerBase:
        pass

    class TrainingArguments:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.args = kwargs.get("args")

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DataCollatorForLanguageModeling:
        def __init__(self, tokenizer=None, mlm=False):
            self.tokenizer = tokenizer
            self.mlm = mlm

    class _Tokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "<eos>"

        def __call__(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            return {"input_ids": [[0] * len(text) for text in texts]}

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _TransformersStub._Tokenizer()

    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(save_pretrained=lambda *_args, **_kwargs: None)


class _PeftStub(types.ModuleType):
    class LoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class PeftModel:
        @classmethod
        def from_pretrained(cls, model, output_dir):
            merged = SimpleNamespace(save_pretrained=lambda *_args, **_kwargs: None)
            return SimpleNamespace(merge_and_unload=lambda: merged)

    @staticmethod
    def get_peft_model(model, config):
        return model

    @staticmethod
    def prepare_model_for_kbit_training(model):
        return model


class FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.seen_texts = []

    def __call__(self, texts, truncation=None, max_length=None, padding=None):
        if isinstance(texts, str):
            texts = [texts]
        self.seen_texts.extend(texts)
        ids = [list(range(min(len(text.split()), max_length or 0))) for text in texts]
        masks = [[1] * len(entry) for entry in ids]
        if padding == "max_length" and max_length:
            for entry, mask in zip(ids, masks):
                padding_len = max_length - len(entry)
                entry.extend([0] * padding_len)
                mask.extend([0] * padding_len)
        return {"input_ids": ids, "attention_mask": masks}


@pytest.fixture()
def train_module(monkeypatch):
    torch_stub = _TorchStub()
    transformers_stub = _TransformersStub("transformers")
    transformers_stub.TrainingArguments = _TransformersStub.TrainingArguments
    transformers_stub.Trainer = _TransformersStub.Trainer
    transformers_stub.BitsAndBytesConfig = _TransformersStub.BitsAndBytesConfig
    transformers_stub.DataCollatorForLanguageModeling = _TransformersStub.DataCollatorForLanguageModeling
    transformers_stub.AutoTokenizer = _TransformersStub.AutoTokenizer
    transformers_stub.AutoModelForCausalLM = _TransformersStub.AutoModelForCausalLM
    peft_stub = _PeftStub("peft")
    peft_stub.LoraConfig = _PeftStub.LoraConfig
    peft_stub.PeftModel = _PeftStub.PeftModel
    peft_stub.get_peft_model = _PeftStub.get_peft_model
    peft_stub.prepare_model_for_kbit_training = _PeftStub.prepare_model_for_kbit_training

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_stub.nn)
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    monkeypatch.setitem(sys.modules, "peft", peft_stub)

    module = importlib.import_module("src.deepthought.train")
    return importlib.reload(module)


@pytest.fixture()
def instruction_dataset():
    train_ds = Dataset.from_dict(
        {
            "instruction": ["Do the thing", "Summarize"],
            "context": ["Some context", ""],
            "response": ["Result one", "Result two"],
        }
    )
    return DatasetDict({"train": train_ds})


@pytest.fixture()
def fake_tokenizer():
    return FakeTokenizer()


def test_parse_args_defaults(train_module):
    args = train_module.parse_args([])
    assert args.dataset_path == "databricks/databricks-dolly-15k"
    assert args.batch_size == 2
    assert args.pack_sequences == "off"
    assert args.use_provided_splits is True
    assert args.enable_awq is False
    assert args.awq_group_size == 128
    assert args.awq_zero_point is True


def test_dataset_loader_formats_prompts(monkeypatch, train_module, instruction_dataset, fake_tokenizer):
    def _fake_load_dataset(path, split=None):
        if split:
            return instruction_dataset["train"]
        return instruction_dataset

    monkeypatch.setattr(train_module, "hf_load_dataset", _fake_load_dataset)
    train_ds, eval_ds = train_module._hf_dataset_loader(
        "unused",
        fake_tokenizer,
        max_seq_length=8,
        pack_sequences="off",
    )

    assert eval_ds is not None
    assert any("### Instruction:" in text for text in fake_tokenizer.seen_texts)
    assert any("### Response:" in text for text in fake_tokenizer.seen_texts)
    example_with_context = next(text for text in fake_tokenizer.seen_texts if "Some context" in text)
    assert "### Input:\nSome context" in example_with_context
    example_without_context = next(text for text in fake_tokenizer.seen_texts if "Summarize" in text)
    assert "### Input:" not in example_without_context
    assert train_ds["input_ids"]


def test_create_trainer_parameters(monkeypatch, train_module, fake_tokenizer):
    model = object()
    train_ds = [1]
    eval_ds = [2]
    lora_kwargs = {}
    trainer_kwargs = {}

    def _lora_config(**kwargs):
        lora_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    def _peft_model(model_arg, config):
        return (model_arg, config)

    class DummyTrainer:
        def __init__(self, **kwargs):
            trainer_kwargs.update(kwargs)
            self.args = kwargs["args"]

    monkeypatch.setattr(train_module, "LoraConfig", _lora_config)
    monkeypatch.setattr(train_module, "get_peft_model", _peft_model)
    monkeypatch.setattr(train_module, "Trainer", DummyTrainer)

    trainer, training_args = train_module.create_trainer(
        model,
        fake_tokenizer,
        train_ds,
        eval_ds,
        "./out",
        epochs=2,
        batch_size=3,
        lr=1e-4,
        evaluation_strategy="epoch",
        eval_steps=25,
        logging_steps=9,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.2,
        lora_target_modules=("a", "b"),
    )

    assert lora_kwargs == {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.2,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["a", "b"],
    }
    assert training_args.output_dir.endswith("out")
    assert training_args.num_train_epochs == 2
    assert training_args.per_device_train_batch_size == 3
    assert training_args.learning_rate == 1e-4
    assert training_args.evaluation_strategy == "epoch"
    assert training_args.eval_steps == 25
    assert training_args.logging_steps == 9
    assert trainer_kwargs["train_dataset"] == train_ds
    assert trainer_kwargs["eval_dataset"] == eval_ds
    assert trainer_kwargs["compute_metrics"] == train_module.compute_metrics
    assert trainer.args == training_args


def test_run_training_writes_summary(monkeypatch, train_module, fake_tokenizer, tmp_path):
    output_dir = tmp_path / "results"

    def _fake_load_model(*_args, **_kwargs):
        return object(), fake_tokenizer

    def _fake_load_dataset(*_args, **_kwargs):
        return [1], [2]

    class DummyTrainer:
        def __init__(self):
            self.saved = []

        def train(self, resume_from_checkpoint=False):
            return SimpleNamespace(metrics={"train_loss": 0.42})

        def evaluate(self):
            return {"eval_loss": 0.2, "eval_perplexity": 1.5}

        def save_metrics(self, split, metrics):
            self.saved.append((split, metrics))

        def save_model(self):
            self.saved.append("model")

        def save_state(self):
            self.saved.append("state")

    monkeypatch.setattr(train_module, "load_model", _fake_load_model)
    monkeypatch.setattr(train_module, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(train_module, "create_trainer", lambda *_a, **_k: (DummyTrainer(), None))

    cfg = train_module.TrainingConfig(output_dir=str(output_dir), evaluation_strategy="steps")
    result = train_module.run_training(cfg)

    assert result == 0
    summary_path = output_dir / "metrics_summary.json"
    data = json.loads(summary_path.read_text())
    assert data["output_dir"] == str(output_dir)
    assert data["train_loss"] == 0.42
    assert data["eval_loss"] == 0.2
    assert data["eval_perplexity"] == 1.5


def test_awq_quantization_skip_without_model_path(train_module, fake_tokenizer):
    cfg = train_module.TrainingConfig(output_dir="./out", awq=train_module.AWQConfig(enabled=True))
    assert train_module._perform_awq_quantization(cfg, fake_tokenizer) is None


def test_awq_quantization_error_when_awq_missing(monkeypatch, train_module, fake_tokenizer):
    real_import = builtins.__import__

    def _raise_import(name, *args, **kwargs):
        if name == "awq":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise_import)
    cfg = train_module.TrainingConfig(
        model_path="model-id",
        output_dir="./out",
        awq=train_module.AWQConfig(enabled=True),
    )
    with pytest.raises(RuntimeError):
        train_module._perform_awq_quantization(cfg, fake_tokenizer)


def test_awq_quantization_success(monkeypatch, train_module, instruction_dataset, fake_tokenizer, tmp_path):
    class DummyAWQModel:
        def __init__(self):
            self.saved_to = None
            self.quantized = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def quantize(self, tokenizer, quant_config=None, calib_data=None):
            self.quantized = (tokenizer, quant_config, calib_data)

        def save_quantized(self, output_dir):
            self.saved_to = output_dir

    def _fake_load_dataset(path, split=None):
        if split:
            return instruction_dataset["train"]
        return instruction_dataset

    monkeypatch.setattr(train_module, "hf_load_dataset", _fake_load_dataset)
    monkeypatch.setitem(sys.modules, "awq", SimpleNamespace(AutoAWQForCausalLM=DummyAWQModel))

    cfg = train_module.TrainingConfig(
        model_path="model-id",
        output_dir=str(tmp_path),
        awq=train_module.AWQConfig(enabled=True),
    )
    metrics = train_module._perform_awq_quantization(cfg, fake_tokenizer)

    assert metrics["awq_output_dir"].endswith("awq")
    assert metrics["awq_quant_config"]["w_bit"] == 4
    assert metrics["awq_calibration_samples"] == len(instruction_dataset["train"])
    results_path = tmp_path / "awq_results.json"
    assert results_path.exists()
