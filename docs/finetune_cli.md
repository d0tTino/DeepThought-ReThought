# Finetune CLI Quickstart

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
