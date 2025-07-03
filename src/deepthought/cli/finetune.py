from __future__ import annotations

import argparse
from importlib import import_module


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dtrt finetune", description="Fine-tune a language model with LoRA")
    parser.add_argument("--model-path", default=None, help="Model ID or local path to the base model")
    parser.add_argument(
        "--dataset-path",
        default="databricks/databricks-dolly-15k",
        help="Dataset path or HF dataset identifier",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits for loading the model",
    )
    parser.add_argument("--output-dir", default="./results/lora-adapter", help="Directory to save results")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for fine-tuning."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    train_script = import_module("deepthought.train_script")
    return train_script.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
