# LLM Fine-Tuning Report (Task 1)

## 1. Introduction

This report covers the LLM (Large Language Model) Fine-Tuning task, designated as Task 1 in the project plan for DeepThought-ReThought. The fine-tuning utilities in `deepthought.train` allow users to replicate the process. The initial fine-tuning and iterative model performance evaluation were conducted by the project owner, 'd0tTino'. This document serves as an overview of the process, tools, and general methodology involved.

## 2. Summary from README.md

The `README.md` provides the following context for the LLM fine-tuning efforts:

**Optimized Language Model (Architecture Section):**
> "(In Progress) Utilizing small, open-source LLMs (< 3B parameters, e.g., Llama 3.2 3B Instruct) fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) techniques like QLoRA and further optimized with post-training quantization (e.g., AWQ)."

**Current Status Section:**
> * "Environment setup for LLM fine-tuning (Roadmap Step 3) is complete (CUDA enabled)."
> * "Currently proceeding with the LLM fine-tuning task: Loading `meta-llama/Llama-3.2-3B-Instruct` for QLoRA fine-tuning on the `databricks/databricks-dolly-15k` dataset."

This highlights the core components of the task:
*   **Model:** `meta-llama/Llama-3.2-3B-Instruct`
*   **Dataset:** `databricks/databricks-dolly-15k`
*   **Technique:** QLoRA (Quantized Low-Rank Adaptation), a form of Parameter-Efficient Fine-Tuning (PEFT).

## 3. Relevant Scripts/Configuration

*   **Primary Script:** `deepthought.train` (accessible via ``dtrt finetune``).
*   **Purpose:** The training module is responsible for orchestrating the fine-tuning process. Its key functions include:
    *   Loading the base language model (`meta-llama/Llama-3.2-3B-Instruct`) and its associated tokenizer.
    *   Preparing the `databricks/databricks-dolly-15k` dataset, which involves formatting prompts, tokenizing the text, and splitting it into training and evaluation sets.
    *   Configuring LoRA parameters (such as rank, alpha, and target modules) for efficient fine-tuning.
    *   Utilizing the Hugging Face `Trainer` class to manage the training loop, evaluation, and saving of artifacts.
*   **Key Libraries Used:**
    *   `torch`: For tensor operations and neural network functionalities.
    *   `transformers`: From Hugging Face, for accessing models, tokenizers, `TrainingArguments`, and the `Trainer`.
    *   `datasets`: From Hugging Face, for loading and processing the `databricks/databricks-dolly-15k` dataset.
    *   `peft`: From Hugging Face, for implementing QLoRA and other PEFT techniques.
    *   `bitsandbytes`: For model quantization (specifically 4-bit quantization in QLoRA).

## 4. General QLoRA Fine-Tuning Steps

The typical workflow for QLoRA fine-tuning, implemented in the training module, involves these stages:

1.  **Model Loading:**
    *   Load the pre-trained base model (e.g., `meta-llama/Llama-3.2-3B-Instruct`).
    *   Apply quantization during loading using `BitsAndBytesConfig` (e.g., `load_in_4bit=True`) to reduce memory footprint.

2.  **Tokenizer Setup:**
    *   Load the tokenizer corresponding to the base model.
    *   Ensure a padding token is defined. If the model doesn't have one, it's common to set it to the EOS (End Of Sentence) token.

3.  **Dataset Preparation:**
    *   Load the chosen dataset (e.g., `databricks/databricks-dolly-15k`).
    *   Format the data into a prompt structure suitable for instruction fine-tuning (e.g., combining "instruction", "context", and "response" fields).
    *   Tokenize the formatted prompts.
    *   Split the dataset into training and evaluation subsets.

4.  **LoRA Configuration:**
    *   Define `LoraConfig` from the `peft` library. This includes parameters like:
        *   `r` (rank of the LoRA matrices).
        *   `lora_alpha` (LoRA scaling factor).
        *   `target_modules` (specifying which layers of the model to apply LoRA to, e.g., query/key/value layers in attention blocks).
        *   `lora_dropout`.
        *   `bias` (e.g., "none").
        *   `task_type` (e.g., "CAUSAL_LM").

5.  **Model Preparation for K-bit Training:**
    *   Use `prepare_model_for_kbit_training` from `peft` to further prepare the quantized model for LoRA.
    *   Apply the LoRA configuration to the base model using `get_peft_model`.

6.  **Training Arguments:**
    *   Define `TrainingArguments` from the `transformers` library. This includes hyperparameters and settings like:
        *   `num_train_epochs`.
        *   `per_device_train_batch_size`, `per_device_eval_batch_size`.
        *   `learning_rate`.
        *   `weight_decay`.
        *   `evaluation_strategy` (e.g., "steps").
        *   `logging_steps`, `eval_steps`.
        *   `output_dir` (for saving checkpoints and the final adapter).
        *   `save_steps`.

7.  **Trainer Initialization:**
    *   Instantiate the `Trainer` from `transformers`, providing the model, training arguments, training dataset, evaluation dataset, and tokenizer.

8.  **Training Execution:**
    *   Call the `trainer.train()` method to start the fine-tuning process.

9.  **Saving Artifacts:**
    *   After training, save the trained LoRA adapter using `model.save_pretrained(output_dir)`.
    *   The Trainer typically saves training metrics and logs.

## 5. Monitoring and Evaluation (General Guidance)

Effective LLM fine-tuning requires diligent monitoring and evaluation:

*   **Monitoring:**
    *   **Metrics:** Observe training loss and evaluation loss/metrics (e.g., perplexity, accuracy for specific tasks) throughout the training process. The training script uses `logging_steps` and `evaluation_strategy` in `TrainingArguments` for this.
    *   **Tools:** This data is often logged to the console and can be integrated with tools like TensorBoard or Weights & Biases for visualization and tracking of experiments.
    *   **Resource Usage:** Monitor GPU utilization and memory to ensure efficient training.

*   **Evaluation:**
    *   **Held-out Test Set:** After training, the performance of the fine-tuned LoRA adapter is assessed on a separate, unseen test portion of the dataset.
    *   **Quantitative Metrics:** Calculate metrics relevant to the downstream tasks the LLM is intended for. Examples include:
        *   Perplexity (for language modeling).
        *   ROUGE scores (for summarization).
        *   BLEU scores (for translation).
        *   Accuracy, F1-score (for classification or question answering).
    *   **Qualitative Assessment:** Manually review generated text samples to assess coherence, relevance, creativity, and potential biases or failure modes.
    *   The training utilities include an `eval_dataset`, and the `Trainer` computes evaluation metrics. Saved metrics provide a basis for this.

## 6. Conclusion

This report outlines the approach and key elements for the LLM fine-tuning task within the DeepThought-ReThought project, utilizing the `meta-llama/Llama-3.2-3B-Instruct` model, the `databricks/databricks-dolly-15k` dataset, and the QLoRA technique. While the initial fine-tuning, iterative refinement, and detailed monitoring were conducted by the project owner ('d0tTino'), the provided training utilities contain the specific implementation details allowing users to reproduce or extend this fine-tuning work.

## 7. Running ``dtrt finetune``

Follow these steps to execute the fine-tuning process:

1. **Install dependencies** using `pip install -r requirements.txt`.
2. **Launch training** with optional arguments:

   ```bash
   dtrt finetune --model-path meta-llama/Llama-3.2-3B-Instruct \
       --dataset-path databricks/databricks-dolly-15k
   ```

   Use `--resume` to continue from the latest checkpoint in `./results/lora-adapter`.
3. **Review metrics** in `./results/lora-adapter/train_results.json`.

### Example Metrics

```
{
  "train_runtime": 1234.0,
  "train_loss": 1.23,
  "eval_loss": 1.10
}
```
