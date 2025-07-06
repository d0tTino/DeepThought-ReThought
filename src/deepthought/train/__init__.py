from __future__ import annotations

"""Training utilities for fine-tuning language models."""

import argparse
import os
from typing import Tuple

import torch
from datasets import Dataset, load_dataset
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
    "create_trainer",
    "run",
    "estimate_vram",
]


def load_model(model_path: str | None, bits: int) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model and tokenizer with the given quantization bits."""
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
    except Exception:
        base_model_id = "HuggingFaceH4/zephyr-7b-beta"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_dataset(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 2048,
    pack_sequences: bool = False,
):
    """Load and tokenize the dataset used for fine-tuning."""
    raw_dataset = load_dataset(dataset_path)

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


def create_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
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
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=50,
        fp16=False,
        optim="adamw_torch",
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
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


def run(args: argparse.Namespace) -> int:
    """Execute training using high level helper functions."""
    model, tokenizer = load_model(args.model_path, args.bits)
    train_ds, eval_ds = load_dataset(
        args.dataset_path,
        tokenizer,
        max_seq_length=args.max_seq_length,
        pack_sequences=args.pack_sequences,
    )
    trainer, _ = create_trainer(model, tokenizer, train_ds, eval_ds, args.output_dir)
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model()
    trainer.save_state()
    return 0
