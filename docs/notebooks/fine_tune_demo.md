# Fine-Tune Demo

This notebook demonstrates how to run `dtrt finetune` with VRAM estimation and sequence packing. It also compares runtime with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Install

```bash
pip install deepthought-rethought
```

## Estimate VRAM

```bash
dtrt finetune --model-path meta-llama/Llama-3.2-3B-Instruct --estimate-vram
```

Example output:

```
Estimated VRAM requirement: 6.4 GB
```

Skip dataset loading when only estimating:

```bash
dtrt finetune --estimate-only --model-path meta-llama/Llama-3.2-3B-Instruct
```

## Enable Sequence Packing

Reduce padding by enabling automatic packing:

```bash
dtrt finetune --pack-sequences auto --max-seq-length 2048
```

The `auto` mode samples the dataset and activates packing when the average sequence length is under 70% of the maximum.

## Run Training

```bash
dtrt finetune --dataset-path databricks/databricks-dolly-15k \
    --model-path meta-llama/Llama-3.2-3B-Instruct \
    --epochs 1 --batch-size 1 --pack-sequences auto
```

## Timing Comparison

On a single A100 (80GB) running one epoch of the Dolly dataset:

| Framework       | Runtime |
| --------------- | ------- |
| `dtrt finetune` | ~45 min |
| LLaMA-Factory   | ~55 min |

Sequence packing reduces the total token count, giving `dtrt finetune` a small speed advantage.

