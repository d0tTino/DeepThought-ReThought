"""Training utilities for fine-tuning language models."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple
from importlib import metadata

import torch
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
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
    "AWQConfig",
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
    use_provided_splits: bool = True
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
    awq: "AWQConfig" | None = None


@dataclass
class AWQConfig:
    """Configuration options for post-training AWQ quantization."""

    enabled: bool = False
    dataset_path: str | None = None
    output_dir: str | None = None
    calibration_samples: int = 128
    q_group_size: int = 128
    w_bit: int = 4
    zero_point: bool = True
    version: str = "GEMM"


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
    *,
    use_provided_splits: bool = True,
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

    formatted_train = raw_dataset["train"].map(format_prompt)

    if pack_sequences == "auto":
        sample = formatted_train.select(range(min(1000, len(formatted_train))))
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

    def process_split(ds: Dataset) -> Dataset:
        tokenized = ds.map(tokenize_function, batched=True, remove_columns=["text"])
        if pack_sequences:

            def _pack(split: Dataset) -> Dataset:
                flat_ids = [tid for ids in split["input_ids"] for tid in ids]
                flat_mask = [m for mask in split["attention_mask"] for m in mask]
                total = len(flat_ids) // max_seq_length
                input_ids = [flat_ids[i * max_seq_length : (i + 1) * max_seq_length] for i in range(total)]  # noqa: E203
                attention_mask = [
                    flat_mask[i * max_seq_length : (i + 1) * max_seq_length]
                    for i in range(total)
                ]
                return Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask})

            return _pack(tokenized)

        return tokenized.filter(lambda ex: len(ex["input_ids"]) <= max_seq_length)

    has_provided_eval = use_provided_splits and any(
        split in raw_dataset for split in ("validation", "test")
    )

    if has_provided_eval:
        train_ds = process_split(formatted_train)
        eval_ds = None
        if "validation" in raw_dataset:
            formatted_validation = raw_dataset["validation"].map(format_prompt)
            eval_ds = process_split(formatted_validation)
        elif "test" in raw_dataset:
            formatted_test = raw_dataset["test"].map(format_prompt)
            eval_ds = process_split(formatted_test)
        return train_ds, eval_ds

    split_dataset = formatted_train.train_test_split(test_size=0.05, seed=42)
    train_ds = process_split(split_dataset["train"])
    eval_ds = process_split(split_dataset["test"])
    return train_ds, eval_ds


def load_dataset(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 2048,
    pack_sequences: str | bool = "off",
    use_provided_splits: bool = True,
    *,
    loader: str = "hf",
) -> Tuple[Dataset, Dataset]:
    """Load datasets using the specified loader plugin."""
    fn = _resolve_plugin(_DATASET_GROUP, loader)
    return fn(
        dataset_path,
        tokenizer,
        max_seq_length=max_seq_length,
        pack_sequences=pack_sequences,
        use_provided_splits=use_provided_splits,
    )

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
    return {"loss": loss.item(), "perplexity": perplexity}


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


def _get_eval_metric(metrics: dict[str, Any], key: str) -> Any:
    """Return an evaluation metric value, preferring Trainer-prefixed keys."""

    prefixed_key = f"eval_{key}"
    if prefixed_key in metrics:
        return metrics[prefixed_key]
    return metrics.get(key)


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


def _perform_awq_quantization(config: TrainingConfig, tokenizer) -> dict | None:
    """Optionally run AWQ quantization on the fine-tuned adapter."""

    awq_cfg = config.awq or AWQConfig()
    if not awq_cfg.enabled:
        return None

    if not config.model_path:
        logger.warning(
            "AWQ quantization requested but no base model path was provided; skipping AWQ",
        )
        return None
    base_model_path = Path(config.model_path)
    if not base_model_path.exists():
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:  # pragma: no cover - exercised via tests with mocks
            raise RuntimeError(
                "AWQ quantization requested but the 'huggingface_hub' package is not available",
            ) from exc
        try:
            resolved_path = snapshot_download(repo_id=config.model_path)
            base_model_path = Path(resolved_path)
            logger.info(
                "Resolved base model %s to local cache at %s for AWQ quantization",
                config.model_path,
                base_model_path,
            )
        except Exception:
            logger.warning(
                "AWQ quantization requested but base model %s could not be resolved; skipping AWQ",
                config.model_path,
            )
            return None

    try:
        from awq import AutoAWQForCausalLM
    except Exception as exc:  # pragma: no cover - exercised via tests with mocks
        raise RuntimeError("AWQ quantization requested but the 'awq' package is not available") from exc

    quant_config = {
        "zero_point": awq_cfg.zero_point,
        "q_group_size": awq_cfg.q_group_size,
        "w_bit": awq_cfg.w_bit,
        "version": awq_cfg.version,
    }

    awq_output_dir = awq_cfg.output_dir or os.path.join(config.output_dir, "awq")
    calibration_path = awq_cfg.dataset_path or config.dataset_path
    calibration_dataset = hf_load_dataset(calibration_path, split="train")
    sample_count = awq_cfg.calibration_samples
    if hasattr(calibration_dataset, "select"):
        try:
            calibration_dataset = calibration_dataset.select(range(sample_count))
        except Exception:
            logger.warning("Unable to select calibration subset; using full dataset")
            sample_count = len(calibration_dataset) if hasattr(calibration_dataset, "__len__") else sample_count

    logger.info(
        "Preparing merged checkpoint for AWQ from base model %s with adapters in %s",
        base_model_path,
        config.output_dir,
    )
    try:
        base_model = AutoModelForCausalLM.from_pretrained(str(base_model_path), trust_remote_code=True)
        peft_model = PeftModel.from_pretrained(base_model, config.output_dir)
        merged_model = peft_model.merge_and_unload()
    except Exception:
        logger.exception(
            "Failed to merge base model from %s with adapters in %s; skipping AWQ quantization",
            base_model_path,
            config.output_dir,
        )
        return None

    with tempfile.TemporaryDirectory(prefix="awq-merged-") as merged_dir:
        merged_checkpoint_path = os.path.join(merged_dir, "merged")
        merged_model.save_pretrained(merged_checkpoint_path)
        logger.info("Merged checkpoint for AWQ saved to %s", merged_checkpoint_path)
        model = AutoAWQForCausalLM.from_pretrained(
            merged_checkpoint_path,
            safetensors=True,
            trust_remote_code=True,
            device_map="auto",
        )
        model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_dataset)
        model.save_quantized(awq_output_dir)

    awq_metrics = {
        "awq_output_dir": awq_output_dir,
        "awq_quant_config": quant_config,
        "awq_calibration_samples": sample_count,
    }
    awq_results_path = os.path.join(config.output_dir, "awq_results.json")
    _save_json(awq_results_path, awq_metrics)
    logger.info("AWQ results saved to %s", awq_results_path)
    return awq_metrics


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
        use_provided_splits=config.use_provided_splits,
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
    eval_metrics: dict[str, float] | None = None
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
    summary = {
        "output_dir": config.output_dir,
        "train_loss": train_result.metrics.get("train_loss"),
    }
    if eval_metrics:
        summary["eval_loss"] = _get_eval_metric(eval_metrics, "loss")
        summary["eval_perplexity"] = _get_eval_metric(eval_metrics, "perplexity")
    awq_metrics = _perform_awq_quantization(config, tokenizer)
    if awq_metrics:
        summary.update(awq_metrics)
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
    awq_cfg = AWQConfig(
        enabled=args.enable_awq,
        dataset_path=args.awq_dataset_path,
        output_dir=args.awq_output_dir,
        calibration_samples=args.awq_calibration_samples,
        q_group_size=args.awq_group_size,
        w_bit=args.awq_w_bit,
        zero_point=args.awq_zero_point,
        version=args.awq_version,
    )
    cfg = TrainingConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        model_loader=args.model_loader,
        dataset_loader=args.dataset_loader,
        bits=args.bits,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        pack_sequences=args.pack_sequences,
        use_provided_splits=args.use_provided_splits,
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
        awq=awq_cfg,
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
        "--use-provided-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use dataset validation/test splits when available; disable to always create an automatic "
            "train/test split (default: use provided splits)."
        ),
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
    parser.add_argument("--enable-awq", action="store_true", help="Enable post-training AWQ quantization")
    parser.add_argument(
        "--awq-dataset-path",
        default=None,
        help="Dataset path to use for AWQ calibration (defaults to the training dataset)",
    )
    parser.add_argument(
        "--awq-output-dir",
        default=None,
        help="Output directory for the quantized model (default: <output_dir>/awq)",
    )
    parser.add_argument(
        "--awq-calibration-samples",
        type=int,
        default=128,
        help="Number of samples to use for AWQ calibration",
    )
    parser.add_argument(
        "--awq-group-size",
        type=int,
        default=128,
        help="Group size for AWQ quantization",
    )
    parser.add_argument(
        "--awq-w-bit",
        type=int,
        default=4,
        help="Bit width for AWQ quantization",
    )
    parser.add_argument(
        "--awq-zero-point",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable zero-point optimization in AWQ quantization",
    )
    parser.add_argument(
        "--awq-version",
        default="GEMM",
        help="AWQ kernel version to use (for example, GEMM)",
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
    awq_cfg = AWQConfig(
        enabled=args.enable_awq,
        dataset_path=args.awq_dataset_path,
        output_dir=args.awq_output_dir,
        calibration_samples=args.awq_calibration_samples,
        q_group_size=args.awq_group_size,
        w_bit=args.awq_w_bit,
        zero_point=args.awq_zero_point,
        version=args.awq_version,
    )
    cfg = TrainingConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        model_loader=args.model_loader,
        dataset_loader=args.dataset_loader,
        bits=args.bits,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        pack_sequences=args.pack_sequences,
        use_provided_splits=args.use_provided_splits,
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
        awq=awq_cfg,
    )
    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
