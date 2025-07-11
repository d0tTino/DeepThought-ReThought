# Finetune CLI Quickstart

Install the package with:

```bash
conda install deepthought-rethought
```

Build the training image and launch finetuning:

```bash
docker build -f docker/Dockerfile.finetune -t dtrt-finetune .
```

Run training with GPU support:

```bash
docker run --gpus all dtrt-finetune \
    --dataset-path <path> --model-path <model-id>
```

If you only have CPU resources, try the [Colab notebook](https://colab.research.google.com/github/d0tTino/DeepThought-ReThought/blob/main/docs/Finetune_CPU.ipynb).

You can estimate the required VRAM before starting:

```bash
dtrt finetune --estimate-vram
```

To only calculate the estimate without loading the dataset, use:

```bash
dtrt finetune --estimate-only
```

Sequence packing can be controlled automatically using heuristics:

```bash
dtrt finetune --pack-sequences auto --max-seq-length 2048
```

The `auto` option samples up to 1000 records from the dataset and enables
packing when the average tokenized length is below 70% of the configured
maximum. These heuristics mirror the behaviour of Predibase's automatic
packing logic and aim to reduce padding without manual tuning.
