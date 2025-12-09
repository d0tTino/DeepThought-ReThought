from __future__ import annotations

"""Training utilities for fine-tuning language models."""

import argparse
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from importlib import metadata
from typing import Callable, Dict, Tuple

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
    "run_training",
    "run",
    "estimate_vram",
    "compute_metrics",
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
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    use_nf4: bool = True
    use_double_quant: bool = True
    compute_dtype: str = "bfloat16"


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


def _hf_model_loader(
    model_path: str | None,
    bits: int,
    *,
    use_nf4: bool = True,
    use_double_quant: bool = True,
    compute_dtype: str = "bfloat16",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Default Hugging Face model loader."""
    if bits not in {4, 8}:
        raise ValueError("Only 4-bit and 8-bit quantization are supported for the HF loader")

    base_model_id = model_path or "meta-llama/Llama-3.2-3B-Instruct"
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    quantization_config: BitsAndBytesConfig
    if bits == 4:
        normalized_dtype = str(compute_dtype).lower()
        if normalized_dtype not in dtype_map:
            raise ValueError(
                f"Unsupported compute dtype '{compute_dtype}'. Choose from {', '.join(dtype_map.keys())}."
            )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4" if use_nf4 else "fp4",
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_compute_dtype=dtype_map[normalized_dtype],
        )
    else:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
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
    use_nf4: bool = True,
    use_double_quant: bool = True,
    compute_dtype: str = "bfloat16",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model using the specified loader plugin."""
    fn = _resolve_plugin(_MODEL_GROUP, loader)
    signature = inspect.signature(fn)
    kwargs = {
        "use_nf4": use_nf4,
        "use_double_quant": use_double_quant,
        "compute_dtype": compute_dtype,
    }

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return fn(model_path, bits, **kwargs)

    supported_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return fn(model_path, bits, **supported_kwargs)


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
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ),
) -> Tuple[Trainer, TrainingArguments]:
    """Create the Trainer instance used for fine-tuning."""
    os.makedirs(output_dir, exist_ok=True)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_target_modules),
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


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics for causal language modeling.

    The callback mirrors Hugging Face expectations so it can be passed directly to
    ``Trainer``. It returns both the mean loss and its exponentiated perplexity.
    """

    import torch.nn.functional as F

    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = torch.from_numpy(logits) if not torch.is_tensor(logits) else logits
    labels = torch.from_numpy(labels) if not torch.is_tensor(labels) else labels

    with torch.no_grad():
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
    perplexity = torch.exp(loss).item()
    return {"eval_loss": loss.item(), "perplexity": perplexity}


def _save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


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


def _prepare_metrics(metrics: dict | None) -> dict:
    if metrics is None:
        return {}
    prepared = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            prepared[key] = value
        else:
            try:
                prepared[key] = float(value)
            except Exception:
                prepared[key] = str(value)
    return prepared


def _print_summary(output_dir: str, train_metrics: dict, eval_metrics: dict | None) -> None:
    header = "\n=== Training Summary ==="
    lines = [header, f"Model artifacts: {output_dir}"]
    if train_metrics:
        lines.append("Train metrics:")
        for key, value in train_metrics.items():
            lines.append(f"  - {key}: {value}")
    if eval_metrics:
        lines.append("Eval metrics:")
        for key, value in eval_metrics.items():
            lines.append(f"  - {key}: {value}")
    print("\n".join(lines))


def run_training(config: TrainingConfig) -> int:
    """Execute training using :class:`TrainingConfig`."""
    model, tokenizer = load_model(
        config.model_path,
        config.bits,
        loader=config.model_loader,
        use_nf4=config.use_nf4,
        use_double_quant=config.use_double_quant,
        compute_dtype=config.compute_dtype,
    )
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
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
    )
    train_result = trainer.train(resume_from_checkpoint=config.resume)
    train_metrics = _prepare_metrics(train_result.metrics)
    trainer.save_metrics("train", train_metrics)
    train_metrics_path = os.path.join(config.output_dir, "train_results.json")
    logger.info("Training metrics saved to %s", train_metrics_path)
    eval_metrics = None
    if config.evaluation_strategy != "no":
        eval_metrics = _prepare_metrics(trainer.evaluate())
        trainer.save_metrics("eval", eval_metrics)
        eval_metrics_path = os.path.join(config.output_dir, "eval_results.json")
        logger.info("Evaluation metrics saved to %s", eval_metrics_path)
        final_eval_metrics_path = os.path.join(config.output_dir, "final_eval_metrics.json")
        _save_json(final_eval_metrics_path, eval_metrics)
        logger.info("Final evaluation metrics written to %s", final_eval_metrics_path)
    trainer.save_model()
    trainer.save_state()
    summary_path = os.path.join(config.output_dir, "metrics_summary.json")
    _save_json(summary_path, {"train": train_metrics, "eval": eval_metrics, "output_dir": config.output_dir})
    logger.info("Metrics summary saved to %s", summary_path)
    _print_summary(config.output_dir, train_metrics or {}, eval_metrics)
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
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(args.lora_target_modules),
        use_nf4=args.use_nf4,
        use_double_quant=args.use_double_quant,
        compute_dtype=args.compute_dtype,
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
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA scaling")
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="List of module names to wrap with LoRA adapters",
    )
    parser.add_argument(
        "--use-nf4",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable NF4 quantization (disable to use FP4)",
    )
    parser.add_argument(
        "--use-double-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable nested quantization for 4-bit weights",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Computation dtype for 4-bit layers",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.estimate_vram or args.estimate_only:
        model, _ = load_model(
            args.model_path,
            args.bits,
            loader=args.model_loader,
            use_nf4=args.use_nf4,
            use_double_quant=args.use_double_quant,
            compute_dtype=args.compute_dtype,
        )
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
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(args.lora_target_modules),
        use_nf4=args.use_nf4,
        use_double_quant=args.use_double_quant,
        compute_dtype=args.compute_dtype,
    )
    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
