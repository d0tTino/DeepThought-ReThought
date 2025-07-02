"""Command line interface for DeepThought-ReThought."""

from __future__ import annotations

import argparse

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepThought training utilities")
    parser.add_argument("--model", dest="model_path", default=None, help="Model ID or local path")
    parser.add_argument(
        "--dataset",
        dest="dataset_path",
        default="databricks/databricks-dolly-15k",
        help="Dataset path or HF dataset identifier",
    )
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4, help="Quantization bits")
    parser.add_argument(
        "--output-dir", dest="output_dir", default="./results/lora-adapter", help="Directory to save results"
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from . import train_script

    return train_script.run(args)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
