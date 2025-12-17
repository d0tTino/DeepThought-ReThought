import importlib
import sys
import types

import pytest
import torch


def _install_train_stubs():
    peft_mod = types.ModuleType("peft")
    peft_mod.LoraConfig = object
    peft_mod.get_peft_model = lambda *args, **kwargs: None
    peft_mod.prepare_model_for_kbit_training = lambda *args, **kwargs: None
    sys.modules["peft"] = peft_mod

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoModelForCausalLM = object
    transformers_mod.AutoTokenizer = object
    transformers_mod.BitsAndBytesConfig = object
    transformers_mod.DataCollatorForLanguageModeling = object
    transformers_mod.PreTrainedTokenizerBase = type(
        "PreTrainedTokenizerBase", (object,), {}
    )
    transformers_mod.Trainer = object
    transformers_mod.TrainingArguments = object
    sys.modules["transformers"] = transformers_mod


_install_train_stubs()
train = importlib.import_module("deepthought.train")


class DummyTokenizer:
    def __call__(self, text, truncation=None, max_length=None, padding=None):
        texts = text if isinstance(text, list) else [text]
        input_ids = []
        attention_mask = []
        for t in texts:
            base_len = max(1, len(str(t).split()))
            ids = list(range(1, base_len + 1))
            attn = [1] * len(ids)
            if max_length is not None:
                ids = ids[:max_length]
                attn = attn[:max_length]
                if padding == "max_length" and len(ids) < max_length:
                    pad = max_length - len(ids)
                    ids += [0] * pad
                    attn += [0] * pad
            input_ids.append(ids)
            attention_mask.append(attn)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TrackingDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.map_history = []

    def map(self, func, batched=False, remove_columns=None):
        if batched:
            columns = {k: [row[k] for row in self.rows] for k in self.rows[0]}
            result = func(columns)
            length = len(next(iter(result.values())))
            new_rows = [
                {key: values[i] for key, values in result.items()} for i in range(length)
            ]
        else:
            new_rows = [func(row) for row in self.rows]

        if remove_columns:
            new_rows = [
                {key: value for key, value in row.items() if key not in remove_columns}
                for row in new_rows
            ]

        self.map_history.append(new_rows)
        return TrackingDataset(new_rows)

    def filter(self, predicate):
        return TrackingDataset([row for row in self.rows if predicate(row)])

    def select(self, indices):
        return TrackingDataset([self.rows[i] for i in indices])

    def train_test_split(self, test_size=0.2, seed=None):
        split = max(1, int(len(self.rows) * test_size))
        return {"train": TrackingDataset(self.rows[split:]), "test": TrackingDataset(self.rows[:split])}

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def __len__(self):
        return len(self.rows)


class DummyEvalPrediction:
    def __init__(self, logits, labels):
        self.predictions = logits
        self.label_ids = labels

    def __iter__(self):
        return iter((self.predictions, self.label_ids))


def test_hf_dataset_loader_formats_prompts_and_packs(monkeypatch):
    train_split = TrackingDataset(
        [
            {"instruction": "Summarize", "context": "Details", "response": "Done"},
            {"instruction": "Classify", "context": "", "response": "Label"},
        ]
    )
    validation_split = TrackingDataset(
        [{"instruction": "Explain", "context": "More", "response": "OK"}]
    )

    monkeypatch.setattr(train, "hf_load_dataset", lambda _: {"train": train_split, "validation": validation_split})

    tokenizer = DummyTokenizer()
    train_ds, eval_ds = train._hf_dataset_loader(
        "unused/path",
        tokenizer,
        max_seq_length=6,
        pack_sequences="on",
        use_provided_splits=True,
    )

    assert "Below is an instruction that describes a task, paired with an input" in train_split.map_history[0][0]["text"]
    assert "### Input:\nDetails" in train_split.map_history[0][0]["text"]
    assert "### Response:\nLabel" in train_split.map_history[0][1]["text"]

    from datasets import Dataset

    assert isinstance(train_ds, Dataset)
    assert isinstance(eval_ds, Dataset)
    assert len(train_ds["input_ids"]) == 2
    assert all(len(ids) == 6 for ids in train_ds["input_ids"])


def test_compute_metrics_returns_expected_loss_and_perplexity():
    logits = torch.tensor(
        [
            [[2.0, 0.0], [0.5, 1.5]],
            [[1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=torch.float,
    )
    labels = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=torch.long,
    )

    eval_pred = DummyEvalPrediction(logits.numpy(), labels.numpy())
    metrics = train.compute_metrics(eval_pred)

    expected_loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
    )

    assert pytest.approx(metrics["loss"], rel=1e-5) == expected_loss.item()
    assert pytest.approx(metrics["perplexity"], rel=1e-5) == torch.exp(expected_loss).item()
    assert set(metrics.keys()) == {"loss", "perplexity"}
