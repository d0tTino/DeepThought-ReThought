from __future__ import annotations

import argparse
import sys

from deepthought import train as train_utils


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA")
    parser.add_argument("--model-path", default=None, help="Model ID or local path to the base model")
    parser.add_argument(
        "--dataset-path",
        default="databricks/databricks-dolly-15k",
        help="Dataset path or HF dataset identifier",
    )
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits for loading the model")
    parser.add_argument("--output-dir", default="./results/lora-adapter", help="Directory to save results")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    return parser.parse_args(args)


def run(args: argparse.Namespace) -> int:
    """Run training using helper functions from :mod:`deepthought.train`."""
    return train_utils.run(args)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - top-level script errors
        print(f"Fatal error: {exc}")
        sys.exit(1)
