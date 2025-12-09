from __future__ import annotations

"""Training utilities for fine-tuning language models."""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from importlib import metadata
from typing import Callable, Tuple

import torch
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

__all__ = [
    "load_model",
    "load_dataset",
    "_hf_model_loader",
    "_hf_dataset_loader",
    "create_trainer",
    "compute_metrics",
    "run_training",
    "run",
    "estimate_vram",
    "parse_args",
    "main",
    "TrainingConfig",
]


logger = logging.getLogger(__name__)


_MODEL_GROUP = "dtrt.model_loaders"
_DATASET_GROUP = "dtrt.dataset_loaders"


@dataclass
class TrainingConfig:
    """Configuration options for :func:`run_training`."""

    model_path: str | None = None
    dataset_path: str = "databricks/databricks-dolly-15k"
    model_loader: str = "hf"
    dataset_loader: str = "hf"
    bits: int = 4
    output_dir: str = "./results/lora-adapter"
    max_seq_length: int = 2048
    pack_sequences: str | bool = "off"
    epochs: float = 1.0
    batch_size: int = 2
    lr: float = 2e-4
    resume: bool = False
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    logging_steps: int = 10


def _resolve_plugin(group: str, name: str) -> Callable:
    """Return a plugin callable from entry points."""
    eps = metadata.entry_points()
    ep_iter = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
    for ep in ep_iter:
        if ep.name == name:
            return ep.load()
    if group == _MODEL_GROUP and name == "hf":
        return _hf_model_loader
    if group == _DATASET_GROUP and name == "hf":
        return _hf_dataset_loader
    raise KeyError(f"No plugin named '{name}' in group '{group}'")


def _hf_model_loader(model_path: str | None, bits: int) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Default Hugging Face model loader."""
    base_model_id = model_path or "meta-llama/Llama-3.2-3B-Instruct"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    except Exception as exc:
        logger.exception("Failed to load model %s: %s", base_model_id, exc)
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_model(
    model_path: str | None,
    bits: int,
    *,
    loader: str = "hf",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model using the specified loader plugin."""
    fn = _resolve_plugin(_MODEL_GROUP, loader)
    return fn(model_path, bits)


def _hf_dataset_loader(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 2048,
    pack_sequences: str | bool = "off",
) -> Tuple[Dataset, Dataset]:
    """Default Hugging Face dataset loader."""
    raw_dataset = hf_load_dataset(dataset_path)

    def format_prompt(example):
        instruction = example["instruction"]
        context = example.get("context", "")
        response = example["response"]
        if context and len(context.strip()) > 0:
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n{response}"
            )
        else:
            prompt = (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                f"\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
            )
        return {"text": prompt}

    formatted_dataset = raw_dataset["train"].map(format_prompt)

    if pack_sequences == "auto":
        sample = formatted_dataset.select(range(min(1000, len(formatted_dataset))))
        tokenized = tokenizer(sample["text"])
        avg_len = sum(len(ids) for ids in tokenized["input_ids"]) / len(tokenized["input_ids"])
        pack_sequences = avg_len < 0.7 * max_seq_length
    elif isinstance(pack_sequences, str):
        pack_sequences = pack_sequences == "on"

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    if pack_sequences:

        def _pack(ds: Dataset) -> Dataset:
            flat_ids = [tid for ids in ds["input_ids"] for tid in ids]
            flat_mask = [m for mask in ds["attention_mask"] for m in mask]
            total = len(flat_ids) // max_seq_length
            input_ids = [flat_ids[i * max_seq_length : (i + 1) * max_seq_length] for i in range(total)]  # noqa: E203
            attention_mask = [
                flat_mask[i * max_seq_length : (i + 1) * max_seq_length] for i in range(total)  # noqa: E203
            ]
            return Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask})

        split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
        train_ds = _pack(split_dataset["train"])
        eval_ds = _pack(split_dataset["test"])
    else:
        filtered_dataset = tokenized_dataset.filter(lambda ex: len(ex["input_ids"]) <= max_seq_length)
        split_dataset = filtered_dataset.train_test_split(test_size=0.05, seed=42)
        train_ds, eval_ds = split_dataset["train"], split_dataset["test"]

    return train_ds, eval_ds


def load_dataset(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 2048,
    pack_sequences: str | bool = "off",
    *,
    loader: str = "hf",
) -> Tuple[Dataset, Dataset]:
    """Load datasets using the specified loader plugin."""
    fn = _resolve_plugin(_DATASET_GROUP, loader)
    return fn(dataset_path, tokenizer, max_seq_length=max_seq_length, pack_sequences=pack_sequences)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute perplexity for language modeling evaluations."""

    import torch.nn.functional as F

    predictions, labels = eval_pred
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    logits = torch.from_numpy(logits) if not torch.is_tensor(logits) else logits
    labels = torch.from_numpy(labels) if not torch.is_tensor(labels) else labels

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    perplexity = torch.exp(loss).item()
    return {"perplexity": perplexity}


def create_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    *,
    epochs: float = 1,
    batch_size: int = 2,
    lr: float = 2e-4,
    evaluation_strategy: str = "steps",
    eval_steps: int | None = None,
    logging_steps: int = 10,
) -> Tuple[Trainer, TrainingArguments]:
    """Create the Trainer instance used for fine-tuning."""
    os.makedirs(output_dir, exist_ok=True)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=batch_size,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        save_steps=100,
        save_total_limit=3,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=50,
        fp16=False,
        optim="adamw_torch",
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer, training_args


def estimate_vram(
    model: AutoModelForCausalLM,
    batch_size: int,
    seq_length: int,
    gradient_accumulation_steps: int = 1,
    bits: int = 16,
) -> float:
    """Roughly estimate VRAM (in GB) required for training."""
    params = sum(p.numel() for p in model.parameters())
    param_bytes = params * bits / 8
    hidden_size = getattr(model.config, "hidden_size", 0)
    activation_bytes = batch_size * gradient_accumulation_steps * seq_length * hidden_size * 2
    return (param_bytes + activation_bytes) / (1024**3)


def run_training(config: TrainingConfig) -> int:
    """Execute training using :class:`TrainingConfig`."""
    model, tokenizer = load_model(config.model_path, config.bits, loader=config.model_loader)
    train_ds, eval_ds = load_dataset(
        config.dataset_path,
        tokenizer,
        max_seq_length=config.max_seq_length,
        pack_sequences=config.pack_sequences,
        loader=config.dataset_loader,
    )
    trainer, _ = create_trainer(
        model,
        tokenizer,
        train_ds,
        eval_ds,
        config.output_dir,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
    )
    train_result = trainer.train(resume_from_checkpoint=config.resume)
    trainer.save_metrics("train", train_result.metrics)
    train_metrics_path = os.path.join(config.output_dir, "train_results.json")
    logger.info("Training metrics saved to %s", train_metrics_path)
    eval_metrics: dict[str, float] | None = None
    if config.evaluation_strategy != "no":
        eval_metrics = trainer.evaluate()
        trainer.save_metrics("eval", eval_metrics)
        eval_metrics_path = os.path.join(config.output_dir, "eval_results.json")
        logger.info("Evaluation metrics saved to %s", eval_metrics_path)
    trainer.save_model()
    trainer.save_state()
    summary = {
        "output_dir": config.output_dir,
        "train_loss": train_result.metrics.get("train_loss"),
    }
    if eval_metrics:
        summary["eval_loss"] = eval_metrics.get("eval_loss")
        summary["eval_perplexity"] = eval_metrics.get("eval_perplexity")
    summary_path = os.path.join(config.output_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Training summary saved to %s", summary_path)
    logger.info("Training summary:\n%s", json.dumps(summary, indent=2))
    print("Training summary:")
    print(json.dumps(summary, indent=2))
    return 0


def run(args: argparse.Namespace) -> int:
    """Execute training using high level helper functions."""
    cfg = TrainingConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        model_loader=args.model_loader,
        dataset_loader=args.dataset_loader,
        bits=args.bits,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        pack_sequences=args.pack_sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
    )
    return run_training(cfg)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line options for fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA")
    parser.add_argument("--model-path", default=None, help="Model ID or local path to the base model")
    parser.add_argument(
        "--dataset-path",
        default="databricks/databricks-dolly-15k",
        help="Dataset path or HF dataset identifier",
    )
    parser.add_argument(
        "--model-loader",
        default="hf",
        help="Name of the model loader plugin to use",
    )
    parser.add_argument(
        "--dataset-loader",
        default="hf",
        help="Name of the dataset loader plugin to use",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits for loading the model",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/lora-adapter",
        help="Directory to save results",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--pack-sequences",
        choices=["on", "off", "auto"],
        default="off",
        help="Sequence packing mode. 'auto' uses heuristics to reduce padding",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Estimate VRAM and exit without loading the dataset",
    )
    parser.add_argument(
        "--estimate-vram",
        action="store_true",
        help="Print VRAM estimate before training",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument(
        "--evaluation-strategy",
        choices=["no", "steps", "epoch"],
        default="steps",
        help="Frequency for evaluation during training",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Number of update steps between evaluations when using step-based evaluation",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Number of update steps between logging events",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.estimate_vram or args.estimate_only:
        model, _ = load_model(args.model_path, args.bits, loader=args.model_loader)
        vram = estimate_vram(
            model,
            batch_size=args.batch_size,
            seq_length=args.max_seq_length,
            gradient_accumulation_steps=8,
            bits=args.bits,
        )
        print(f"Estimated VRAM requirement: {vram:.2f} GB")
        if args.estimate_only:
            return 0
    cfg = TrainingConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        model_loader=args.model_loader,
        dataset_loader=args.dataset_loader,
        bits=args.bits,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        pack_sequences=args.pack_sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
    )
    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
