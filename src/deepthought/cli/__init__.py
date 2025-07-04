from __future__ import annotations

import argparse
import shutil
from importlib import import_module
from pathlib import Path


def _to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def init_service(name: str) -> None:
    dest = Path("src/deepthought/services") / name
    if dest.exists():
        raise SystemExit(f"Service '{name}' already exists")
    # templates live at repo root/tools; navigate up to repository base
    template = Path(__file__).resolve().parents[3] / "tools" / "template_service"

    shutil.copytree(template, dest)
    class_name = _to_camel(name) + "Service"
    for path in dest.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        text = text.replace("TemplateService", class_name)
        text = text.replace("template_service", name)
        path.write_text(text, encoding="utf-8")


def _cmd_init_service(args: argparse.Namespace) -> None:
    init_service(args.name)


def _cmd_finetune(args: argparse.Namespace) -> int:
    train_script = import_module("deepthought.train_script")
    return train_script.run(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dtrt")
    sub = parser.add_subparsers(dest="command")

    finetune_p = sub.add_parser("finetune", description="Fine-tune a language model with LoRA")
    finetune_p.add_argument("--model-path", default=None, help="Model ID or local path to the base model")
    finetune_p.add_argument(
        "--dataset-path",
        default="databricks/databricks-dolly-15k",
        help="Dataset path or HF dataset identifier",
    )
    finetune_p.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits for loading the model",
    )
    finetune_p.add_argument(
        "--output-dir",
        default="./results/lora-adapter",
        help="Directory to save results",
    )
    finetune_p.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    finetune_p.set_defaults(func=_cmd_finetune)

    init_p = sub.add_parser("init")
    init_sub = init_p.add_subparsers(dest="target")

    svc_p = init_sub.add_parser("service")
    svc_p.add_argument("name")
    svc_p.set_defaults(func=_cmd_init_service)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        return args.func(args) or 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
