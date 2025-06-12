# Combined training script for fine-tuning with LoRA
# This script implements Subtask 5 from the instruction

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import gc
import time
import os

print("\n=== Starting Fine-tuning Process ===\n")

# ===== Step 1: Load Model & Tokenizer =====
print("Step 1: Loading model and tokenizer...")

# Try to use Llama 3.2, fall back to Zephyr if access is restricted
try:
    base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Attempting to load model: {base_model_id}")
    
    # Configure 4-bit quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, 
        trust_remote_code=True
    )
    
except Exception as e:
    print(f"Error loading Llama 3.2: {e}")
    print("Falling back to Zephyr-7B...")
    
    base_model_id = "HuggingFaceH4/zephyr-7b-beta"
    print(f"Attempting to load model: {base_model_id}")
    print("Note: Using Zephyr-7B instead of Llama 3.2 due to access restrictions")
    
    # Configure 4-bit quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, 
        trust_remote_code=True
    )

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

print("Model loaded successfully with 4-bit quantization.")
print(f"Model memory footprint: {model.get_memory_footprint() / (1024 ** 2):.2f} MB")
print(f"Model type: {model.__class__.__name__}")
print("Tokenizer loaded successfully.")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")

# ===== Step 2: Load & Prepare Dataset =====
print("\nStep 2: Loading and preparing dataset...")

# Define dataset ID
dataset_id = "databricks/databricks-dolly-15k"
print(f"Loading dataset: {dataset_id}")

# Load dataset
raw_dataset = load_dataset(dataset_id)
print(f"Dataset loaded successfully with {len(raw_dataset['train'])} samples.")

# Define prompt formatting function
def format_prompt(example):
    """Format a single example into a prompt with response."""
    instruction = example["instruction"]
    context = example.get("context", "")
    response = example["response"]
    
    if context and len(context.strip()) > 0:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{context}

### Response:
{response}"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""
    
    return {"text": prompt}

# Apply formatting
print("Formatting dataset...")
formatted_dataset = raw_dataset["train"].map(format_prompt)
print("Dataset formatted successfully.")

# Define max sequence length
max_seq_length = 2048
print(f"Using max_seq_length: {max_seq_length}")

# Tokenize the dataset
def tokenize_function(examples):
    """Tokenize examples for training."""
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=max_seq_length,
        padding="max_length"
    )

print("Tokenizing dataset...")
tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Filter long sequences
print(f"Filtering sequences longer than {max_seq_length} tokens...")
filtered_dataset = tokenized_dataset.filter(
    lambda example: len(example['input_ids']) <= max_seq_length
)

# Split dataset into train and evaluation
print("Splitting dataset into train/eval sets...")
split_dataset = filtered_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# ===== Step 3: Configure LoRA =====
print("\nStep 3: Configuring LoRA for fine-tuning...")

# Prepare model for k-bit training
print("Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)

# Define LoRA Configuration
lora_config = LoraConfig(
    r=32,                   # Rank
    lora_alpha=64,          # Alpha parameter for LoRA scaling
    lora_dropout=0.1,       # Dropout probability for LoRA layers
    bias="none",            # No bias training
    task_type="CAUSAL_LM",  # Task type for causal language modeling
    target_modules=[        # Target attention modules for the model
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj"
    ]
)

print("LoRA configuration created.")
print(f"LoRA parameters: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
print(f"Target modules: {lora_config.target_modules}")

# Apply LoRA config to the model
model = get_peft_model(model, lora_config)
print("Applied LoRA configuration to model")

# ===== Step 4: Set Training Arguments =====
print("\nStep 4: Setting training arguments...")

# Define output directory
output_dir = "./results/lora-adapter"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Define training arguments with simplified parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,                # Reduced for testing
    per_device_train_batch_size=2,     # Reduced to avoid OOM errors
    gradient_accumulation_steps=8,     # Increases effective batch size
    per_device_eval_batch_size=2,      # Evaluation batch size
    logging_dir=f"{output_dir}/logs",  # Directory for logs
    logging_steps=10,                  # Log every 10 steps
    save_steps=100,                    # Save every 100 steps
    save_total_limit=3,                # Keep only 3 checkpoints
    learning_rate=2e-4,                # Learning rate
    weight_decay=0.01,                 # Weight decay
    warmup_steps=50,                   # Warmup steps
    fp16=False,                        # Disabled due to 4-bit quantization
    optim="adamw_torch",               # Standard optimizer
)

print("Training arguments created.")
print(f"Training for {training_arguments.num_train_epochs} epochs")
print(f"Effective batch size: {training_arguments.per_device_train_batch_size * training_arguments.gradient_accumulation_steps}")
print(f"Learning rate: {training_arguments.learning_rate}")

# ===== Step 5: Initialize and Run Trainer =====
print("\n--- Starting Subtask 5: Initialize and Run Trainer ---")

# Define a data collator that will handle the batching
from transformers import DataCollatorForLanguageModeling  # noqa: E402
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("\nInitializing Trainer...")
trainer = None  # Initialize trainer variable

try:
    # Use standard Trainer instead of SFTTrainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    print("Trainer initialized successfully.")

except Exception as e:
    print(f"Error initializing Trainer: {e}")
    print("Check arguments passed to Trainer.")
    raise e

# Start training
print("\nStarting training process...")
print(f"Training for {training_arguments.num_train_epochs} epochs.")
print(f"Effective batch size: {training_arguments.per_device_train_batch_size * training_arguments.gradient_accumulation_steps}")
print("Logs will appear below. This may take a significant amount of time...")
print("-" * 50)  # Separator for clarity

start_time = time.time()
train_result = None  # Initialize train_result

try:
    # This is the core training command
    train_result = trainer.train()

    end_time = time.time()
    training_time = end_time - start_time
    print("-" * 50)  # Separator for clarity
    print("\nTraining finished.")
    print(f"Total training time: {training_time / 60:.2f} minutes ({training_time:.2f} seconds)")

    # Log and save final metrics & model adapter
    if train_result:
        metrics = train_result.metrics
        print("\nLogging and saving training metrics...")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)  # Saves to output_dir/train_results.json
        trainer.save_state()  # Saves trainer state, including optimizer state etc.
        print(f"Training metrics saved. Final metrics summary: {metrics}")

        # Explicitly save the final adapter weights
        print(f"\nSaving final LoRA adapter weights to: {training_arguments.output_dir}")
        trainer.save_model()  # Saves adapter config & weights to output_dir
        print("Adapter weights saved successfully.")

    else:
        print("Training result object not found, metrics/model not explicitly saved by this block.")

except Exception as e:
    print(f"\nError during training: {e}")
    print("Potential causes include:")
    print(" - Out Of Memory (OOM): Try reducing 'per_device_train_batch_size' in TrainingArguments.")
    print(" - Out Of Memory (OOM): Try reducing 'max_seq_length'.")
    print(" - Issues with dataset formatting or tokenization.")
    print(" - Compatibility issues between libraries (check versions).")
    # Attempt to clean up memory if OOM occurred
    print("Attempting to clear CUDA cache...")
    if 'model' in locals():
        del model
    if 'trainer' in locals():
        del trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("CUDA cache cleared (attempted).")
    raise e

# Final Check
if train_result is not None:
    print("\nSubtask 5 Check: trainer.train() executed.")
    # Check training loss if available in metrics
    if hasattr(train_result, 'metrics') and 'train_loss' in train_result.metrics:
        print(f"  Final Training Loss reported: {train_result.metrics['train_loss']:.4f}")
    else:
        print("  Training metrics might be available in trainer logs or state.")
else:
    print("\nSubtask 5 Check: trainer.train() did not complete successfully or result not captured.")

print("\n--- Finished Fine-tuning Process ---")

# Attempt to clean up memory after successful training
print("Attempting to clear CUDA cache post-training...")
if 'model' in locals():
    del model
if 'trainer' in locals():
    del trainer
gc.collect()
torch.cuda.empty_cache()
print("CUDA cache cleared post-training (attempted).") 
