import builtins
import importlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset, DatasetDict


class _TorchStub(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.float16 = np.float16
        self.float32 = np.float32
        self.bfloat16 = np.float16
        self.int64 = np.int64
        self.int32 = np.int32
        self.int16 = np.int16
        self.int8 = np.int8
        self.uint8 = np.uint8
        self.bool = bool
        self.complex64 = np.complex64
        self.complex128 = np.complex128
        self.Generator = type("Generator", (), {})

    class Tensor:
        def __init__(self, data, dtype=None):
            self.data = np.array(data, dtype=dtype)
            self.dtype = self.data.dtype

        def detach(self):
            return self

        def cpu(self):
            return self

        def reshape(self, shape):
            self.data = self.data.reshape(shape)
            return self

        def numpy(self):  # pragma: no cover - compatibility helper
            return self.data

        def tolist(self):
            return self.data.tolist()

        def item(self):
            return self.data.item()

        @property
        def shape(self):
            return self.data.shape

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def __call__(self, *args, **kwargs):  # pragma: no cover - compatibility helper
        return self

    def tensor(self, data, dtype=None):
        return self.Tensor(data, dtype=dtype)

    def from_numpy(self, array):
        return self.Tensor(np.array(array))

    def is_tensor(self, obj):
        return isinstance(obj, self.Tensor)

    def exp(self, tensor):
        return self.Tensor(np.exp(getattr(tensor, "data", tensor)))

    def no_grad(self):
        return self._NoGrad()


# Replace heavy modules with lightweight stubs for isolated unit tests
torch_stub = _TorchStub()
sys.modules["torch"] = torch_stub
torch_nn = types.ModuleType("torch.nn")
torch_nn.Module = type("Module", (), {})
sys.modules["torch.nn"] = torch_nn
torch_stub.nn = torch_nn
sys.modules["torch.nn.functional"] = types.SimpleNamespace(cross_entropy=lambda *args, **kwargs: torch_stub.tensor(0.0))


class _TransformersStub(types.ModuleType):
    def __init__(self):
        super().__init__("transformers")
        self.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (object,), {})

    class TrainingArguments:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Trainer:
        def __init__(self, *args, **kwargs):
            self.args = kwargs.get("args")
            self.kwargs = kwargs

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
            return {"input_ids": [[0] * len(t) for t in texts]}

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _TransformersStub._Tokenizer()

    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return object()


transformers_stub = _TransformersStub()
transformers_stub.TrainingArguments = _TransformersStub.TrainingArguments
transformers_stub.Trainer = _TransformersStub.Trainer
transformers_stub.BitsAndBytesConfig = _TransformersStub.BitsAndBytesConfig
transformers_stub.DataCollatorForLanguageModeling = _TransformersStub.DataCollatorForLanguageModeling
transformers_stub.AutoTokenizer = _TransformersStub.AutoTokenizer
transformers_stub.AutoModelForCausalLM = _TransformersStub.AutoModelForCausalLM
sys.modules["transformers"] = transformers_stub


class _PeftStub(types.ModuleType):
    class LoraConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    @staticmethod
    def get_peft_model(model, config):
        return model

    @staticmethod
    def prepare_model_for_kbit_training(model):
        return model


peft_stub = _PeftStub("peft")
peft_stub.LoraConfig = _PeftStub.LoraConfig
peft_stub.get_peft_model = _PeftStub.get_peft_model
peft_stub.prepare_model_for_kbit_training = _PeftStub.prepare_model_for_kbit_training
sys.modules["peft"] = peft_stub

train = importlib.import_module("src.deepthought.train")


class FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.last_texts = []
        self.history = []

    def __call__(self, texts, truncation=None, max_length=None, padding=None):
        if isinstance(texts, str):
            texts = [texts]
        self.last_texts = list(texts)
        self.history.extend(texts)
        input_ids = []
        attention_mask = []
        for text in texts:
            length = min(len(text.split()), max_length or len(text.split()))
            ids = list(range(length))
            mask = [1] * length
            if padding == "max_length" and max_length:
                padding_len = max_length - length
                ids += [0] * padding_len
                mask += [0] * padding_len
            input_ids.append(ids)
            attention_mask.append(mask)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


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
def patched_hf_load_dataset(monkeypatch, instruction_dataset):
    def _fake_load_dataset(path, split=None):
        if split:
            return instruction_dataset["train"]
        return instruction_dataset

    monkeypatch.setattr(train, "hf_load_dataset", _fake_load_dataset)
    return _fake_load_dataset


@pytest.fixture()
def fake_tokenizer():
    return FakeTokenizer()


def test_parse_args_defaults():
    args = train.parse_args([])
    assert args.dataset_path == "databricks/databricks-dolly-15k"
    assert args.batch_size == 2
    assert args.pack_sequences == "off"
    assert args.use_provided_splits is True
    assert args.enable_awq is False
    assert args.awq_group_size == 128
    assert args.awq_zero_point is True


def test_dataset_loader_formats_prompts(monkeypatch, patched_hf_load_dataset, fake_tokenizer):
    train_ds, eval_ds = train._hf_dataset_loader(
        "unused/path",
        fake_tokenizer,
        max_seq_length=8,
        pack_sequences="off",
    )
    assert eval_ds is not None
    assert any("### Input:" in text for text in fake_tokenizer.history)
    assert any("### Response:" in text for text in fake_tokenizer.history)

    example_with_context = next(txt for txt in fake_tokenizer.history if "Some context" in txt)
    assert "### Instruction:" in example_with_context
    assert "### Input:\nSome context" in example_with_context
    example_without_context = next(txt for txt in fake_tokenizer.history if "Summarize" in txt)
    assert "### Input:" not in example_without_context

    assert train_ds["input_ids"]


def test_create_trainer_uses_lora_and_args(monkeypatch, fake_tokenizer):
    model = object()
    train_ds = [1, 2]
    eval_ds = [3]
    prepared_models = []
    lora_params = {}
    trainer_kwargs = {}

    def _prepare(model_arg):
        prepared_models.append(model_arg)
        return model_arg

    def _lora_config(**kwargs):
        lora_params.update(kwargs)
        return SimpleNamespace(**kwargs)

    def _peft_model(model_arg, config):
        return (model_arg, config)

    class DummyTrainer:
        def __init__(self, **kwargs):
            trainer_kwargs.update(kwargs)
            self.args = kwargs["args"]

    monkeypatch.setattr(train, "prepare_model_for_kbit_training", _prepare)
    monkeypatch.setattr(train, "LoraConfig", _lora_config)
    monkeypatch.setattr(train, "get_peft_model", _peft_model)
    monkeypatch.setattr(train, "Trainer", DummyTrainer)

    trainer, training_args = train.create_trainer(
        model,
        fake_tokenizer,
        train_ds,
        eval_ds,
        "./out",
        epochs=3,
        batch_size=4,
        lr=5e-4,
        evaluation_strategy="epoch",
        eval_steps=50,
        logging_steps=7,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.2,
        lora_target_modules=("a", "b"),
    )

    assert prepared_models == [model]
    assert lora_params == {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.2,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["a", "b"],
    }
    assert isinstance(training_args.output_dir, str) and training_args.output_dir.endswith("out")
    assert training_args.num_train_epochs == 3
    assert training_args.per_device_train_batch_size == 4
    assert training_args.learning_rate == 5e-4
    assert training_args.evaluation_strategy == "epoch"
    assert training_args.eval_steps == 50
    assert training_args.logging_steps == 7
    assert isinstance(trainer, DummyTrainer)
    assert trainer_kwargs["train_dataset"] == train_ds
    assert trainer_kwargs["eval_dataset"] == eval_ds
    assert trainer_kwargs["compute_metrics"] == train.compute_metrics


def test_run_training_writes_summary(monkeypatch, tmp_path, fake_tokenizer, instruction_dataset):
    output_dir = tmp_path / "results"

    def _fake_load_model(*_, **__):
        return object(), fake_tokenizer

    def _fake_load_dataset(*_, **__):
        return instruction_dataset["train"], None

    class DummyTrainer:
        def __init__(self, *_, **__):
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

    monkeypatch.setattr(train, "load_model", _fake_load_model)
    monkeypatch.setattr(train, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(train, "create_trainer", lambda *args, **kwargs: (DummyTrainer(), None))

    cfg = train.TrainingConfig(output_dir=str(output_dir), evaluation_strategy="steps")
    rc = train.run_training(cfg)

    summary_path = output_dir / "metrics_summary.json"
    assert rc == 0
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert data["output_dir"] == str(output_dir)
    assert data["train_loss"] == 0.42
    assert data["eval_loss"] == 0.2
    assert data["eval_perplexity"] == 1.5


def test_awq_quantization_skip_and_error(monkeypatch, patched_hf_load_dataset, fake_tokenizer, tmp_path):
    cfg_disabled = train.TrainingConfig(output_dir=str(tmp_path), awq=train.AWQConfig(enabled=False))
    assert train._perform_awq_quantization(cfg_disabled, fake_tokenizer) is None

    real_import = builtins.__import__

    def _raise_import(name, *args, **kwargs):
        if name == "awq":
            raise ImportError("no awq")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise_import)

    cfg_enabled = train.TrainingConfig(output_dir=str(tmp_path), awq=train.AWQConfig(enabled=True))
    with pytest.raises(RuntimeError):
        train._perform_awq_quantization(cfg_enabled, fake_tokenizer)


def test_awq_quantization_success(monkeypatch, patched_hf_load_dataset, fake_tokenizer, instruction_dataset, tmp_path):
    class DummyAWQModel:
        def __init__(self, *args, **kwargs):
            self.quantized = None
            self.saved_to = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def quantize(self, tokenizer_arg, quant_config=None, calib_data=None):
            self.quantized = (tokenizer_arg, quant_config, calib_data)

        def save_quantized(self, output_dir):
            self.saved_to = output_dir

    monkeypatch.setitem(sys.modules, "awq", SimpleNamespace(AutoAWQForCausalLM=DummyAWQModel))

    cfg_enabled = train.TrainingConfig(output_dir=str(tmp_path), awq=train.AWQConfig(enabled=True))
    metrics = train._perform_awq_quantization(cfg_enabled, fake_tokenizer)

    assert metrics["awq_output_dir"].endswith("awq")
    assert metrics["awq_quant_config"]["w_bit"] == 4
    assert metrics["awq_calibration_samples"] == len(instruction_dataset["train"])
    results_path = tmp_path / "awq_results.json"
    assert results_path.exists()

