# Finetune CLI Quickstart

Install the package with:

```bash
conda install deepthought-rethought
pip install deepthought-rethought
```
The PyPI distribution exposes both ``dtrt`` and ``dtrt-finetune`` entry points.

Build the training image and launch finetuning:

```bash
docker build -f docker/Dockerfile.finetune -t dtrt-finetune .
```

Run training with GPU support:

```bash
docker run --gpus all dtrt-finetune \
    --dataset-path <path> --model-path <model-id>
```

If the specified model cannot be loaded, the CLI logs the underlying
exception and exits instead of falling back to a different model.

If you only have CPU resources, try the [Colab notebook](https://colab.research.google.com/github/d0tTino/DeepThought-ReThought/blob/main/docs/Finetune_CPU.ipynb).

You can estimate the required VRAM before starting:

```bash
dtrt finetune --estimate-vram
```
Example output:

```text
Estimated VRAM requirement: 6.4 GB
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

Additional hyperparameters can be configured from the command line:

```bash
dtrt finetune \
    --epochs 3 \
    --batch-size 4 \
    --lr 1e-4
```

These options map directly to `TrainingArguments.num_train_epochs`,
`per_device_train_batch_size` and `learning_rate` respectively.

## Registering custom loaders

`deepthought.train` discovers model and dataset loaders through entry point
plugins. To register your own loader, declare an entry point in your project's
`pyproject.toml`:

```toml
[project.entry-points."dtrt.model_loaders"]
my_loader = "my_package.loaders:load_model"

[project.entry-points."dtrt.dataset_loaders"]
my_dataset = "my_package.loaders:load_dataset"
```

After installing the package, select the loader with `--model-loader` or
`--dataset-loader`:

```bash
dtrt finetune --model-loader my_loader --dataset-loader my_dataset
```
