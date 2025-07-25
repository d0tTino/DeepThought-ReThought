from __future__ import annotations

"""Command line helper for fine-tuning the social perception classifier."""

import argparse
from typing import Dict

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABELS: Dict[int, str] = {0: "flirtation", 1: "avoidance", 2: "manipulation"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return parsed CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, help="HF dataset ID or path")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Base model name")
    parser.add_argument("--output-dir", default="./social_perception_model", help="Where to store the model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    return parser.parse_args(argv)


def _tokenize(tokenizer, example):
    return tokenizer(example["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_dataset(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = dataset.map(lambda ex: _tokenize(tokenizer, ex), batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_ds = tokenized["train"]
    eval_ds = tokenized.get("validation") or tokenized.get("test")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(LABELS))
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

