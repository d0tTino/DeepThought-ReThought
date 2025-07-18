from __future__ import annotations

import argparse
import asyncio
import os
import shutil
from importlib import import_module
from pathlib import Path

from .template_helpers import apply_bus_substitutions, find_template


def _to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def init_service(
    name: str,
    *,
    template_name: str = "service",
    stream_name: str | None = None,
    tls_cert: str | None = None,
    tls_key: str | None = None,
    tls_ca: str | None = None,
    js_storage: str | None = None,
    max_msgs: int | None = None,
) -> None:
    dest = Path("src/deepthought/services") / name
    if dest.exists():
        raise SystemExit(f"Service '{name}' already exists")

    template = find_template(template_name)

    shutil.copytree(template, dest)
    class_name = _to_camel(name) + "Service"
    for path in dest.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        text = text.replace("TemplateService", class_name)
        text = text.replace("template_service", name)
        if stream_name:
            text = text.replace("template_stream", stream_name)
        if tls_cert:
            text = text.replace("template_tls_cert", tls_cert)
        if tls_key:
            text = text.replace("template_tls_key", tls_key)
        if tls_ca:
            text = text.replace("template_tls_ca", tls_ca)
        path.write_text(text, encoding="utf-8")

    env_file = dest / "nats.env.example"
    if env_file.exists():
        text = env_file.read_text(encoding="utf-8")
        text = apply_bus_substitutions(
            text,
            stream_name=stream_name,
            tls_cert=tls_cert,
            tls_key=tls_key,
            tls_ca=tls_ca,
            js_storage=js_storage,
            max_msgs=max_msgs,
        )
        env_file.write_text(text, encoding="utf-8")

    docker_file = dest / "Dockerfile"
    if docker_file.exists():
        text = docker_file.read_text(encoding="utf-8")
        text = apply_bus_substitutions(
            text,
            tls_cert=tls_cert,
            tls_key=tls_key,
            tls_ca=tls_ca,
            js_storage=js_storage,
            max_msgs=max_msgs,
        )
        docker_file.write_text(text, encoding="utf-8")


def init_project(
    name: str,
    *,
    stream_name: str | None = None,
    tls_cert: str | None = None,
    tls_key: str | None = None,
    tls_ca: str | None = None,
    js_storage: str | None = None,
    max_msgs: int | None = None,
    template_name: str = "bus_project",
) -> None:
    """Create a new bus project with a default service.

    The function copies the ``bus_project`` template to ``name`` and then calls
    :func:`init_service` to scaffold the initial service within that directory.
    Any provided stream name or TLS file paths are substituted into the copied
    ``docker-compose.yml`` file.
    """
    dest = Path(name)
    if dest.exists():
        raise SystemExit(f"Project '{name}' already exists")

    template = find_template(template_name)

    shutil.copytree(template, dest)
    compose_file = dest / "docker-compose.yml"
    if compose_file.exists():
        text = compose_file.read_text(encoding="utf-8")
        text = apply_bus_substitutions(
            text,
            stream_name=stream_name,
            tls_cert=tls_cert,
            tls_key=tls_key,
            tls_ca=tls_ca,
        )
        compose_file.write_text(text, encoding="utf-8")
    cwd = Path.cwd()
    try:
        os.chdir(dest)
        init_service(
            name,
            template_name="bus_service",
            stream_name=stream_name,
            tls_cert=tls_cert,
            tls_key=tls_key,
            tls_ca=tls_ca,
            js_storage=js_storage,
            max_msgs=max_msgs,
        )
    finally:
        os.chdir(cwd)


def _cmd_init_service(args: argparse.Namespace) -> None:
    if args.template == "bus_project":
        init_project(
            args.name,
            stream_name=getattr(args, "stream_name", None),
            tls_cert=getattr(args, "tls_cert", None),
            tls_key=getattr(args, "tls_key", None),
            tls_ca=getattr(args, "tls_ca", None),
            js_storage=getattr(args, "js_storage", None),
            max_msgs=getattr(args, "max_msgs", None),
        )
    else:
        template = args.template
        if args.template == "bus_service" and args.name == "reward":
            template = "reward_service"
        init_service(
            args.name,
            template_name=template,
            stream_name=getattr(args, "stream_name", None),
            tls_cert=getattr(args, "tls_cert", None),
            tls_key=getattr(args, "tls_key", None),
            tls_ca=getattr(args, "tls_ca", None),
            js_storage=getattr(args, "js_storage", None),
            max_msgs=getattr(args, "max_msgs", None),
        )


def _cmd_finetune(args: argparse.Namespace) -> int:
    training = import_module("deepthought.train")
    argv = [
        "--dataset-path",
        args.dataset_path,
        "--bits",
        str(args.bits),
        "--output-dir",
        args.output_dir,
        "--model-loader",
        args.model_loader,
        "--dataset-loader",
        args.dataset_loader,
        "--max-seq-length",
        str(args.max_seq_length),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
    ]
    if args.model_path:
        argv[0:0] = ["--model-path", args.model_path]
    if args.pack_sequences != "off":
        argv.extend(["--pack-sequences", args.pack_sequences])
    if args.estimate_only:
        argv.append("--estimate-only")
    elif args.estimate_vram:
        argv.append("--estimate-vram")
    if args.resume:
        argv.append("--resume")
    return training.main(argv)


def _cmd_orchestrate(args: argparse.Namespace) -> int:
    from .. import orchestrator

    asyncio.run(orchestrator.run(args.config))
    return 0


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
        "--model-loader",
        default="hf",
        help="Name of the model loader plugin to use",
    )
    finetune_p.add_argument(
        "--dataset-loader",
        default="hf",
        help="Name of the dataset loader plugin to use",
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
    finetune_p.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    finetune_p.add_argument(
        "--pack-sequences",
        choices=["on", "off", "auto"],
        default="off",
        help="Sequence packing mode. 'auto' uses heuristics to reduce padding",
    )
    finetune_p.add_argument(
        "--epochs",
        type=float,
        default=1,
        help="Number of training epochs",
    )
    finetune_p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size",
    )
    finetune_p.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    finetune_p.add_argument(
        "--estimate-only",
        action="store_true",
        help="Estimate VRAM and exit without loading the dataset",
    )
    finetune_p.add_argument(
        "--estimate-vram",
        action="store_true",
        help="Print VRAM estimate before training",
    )
    finetune_p.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    finetune_p.set_defaults(func=_cmd_finetune)

    orchestrate_p = sub.add_parser(
        "orchestrate",
        description="Launch multiple services from a config file",
    )
    orchestrate_p.add_argument("config", help="Path to YAML or JSON config")
    orchestrate_p.set_defaults(func=_cmd_orchestrate)

    init_p = sub.add_parser("init")
    init_sub = init_p.add_subparsers(dest="target")

    svc_p = init_sub.add_parser("service")
    svc_p.add_argument("name")
    svc_p.set_defaults(func=_cmd_init_service, template="service")

    bus_p = sub.add_parser("bus")
    bus_sub = bus_p.add_subparsers(dest="bus_cmd")

    bus_init = bus_sub.add_parser("init")
    bus_init_sub = bus_init.add_subparsers(dest="target")

    bus_svc = bus_init_sub.add_parser("service")
    bus_svc.add_argument("name")
    bus_svc.add_argument(
        "--stream-name",
        default="deepthought_events",
        help="JetStream stream name to use",
    )
    bus_svc.add_argument("--tls-cert", default="", help="Path to the client certificate")
    bus_svc.add_argument("--tls-key", default="", help="Path to the client key")
    bus_svc.add_argument("--tls-ca", default="", help="Path to the CA certificate")
    bus_svc.add_argument(
        "--js-storage",
        choices=["memory", "file"],
        default="memory",
        help="JetStream storage backend",
    )
    bus_svc.add_argument(
        "--max-msgs",
        type=int,
        default=10000,
        help="Maximum messages per subject",
    )
    bus_svc.set_defaults(func=_cmd_init_service, template="bus_service")

    bus_proj = bus_init_sub.add_parser("project")
    bus_proj.add_argument("name")
    bus_proj.add_argument(
        "--stream-name",
        default="deepthought_events",
        help="JetStream stream name to use",
    )
    bus_proj.add_argument("--tls-cert", default="", help="Path to the client certificate")
    bus_proj.add_argument("--tls-key", default="", help="Path to the client key")
    bus_proj.add_argument("--tls-ca", default="", help="Path to the CA certificate")
    bus_proj.add_argument(
        "--js-storage",
        choices=["memory", "file"],
        default="memory",
        help="JetStream storage backend",
    )
    bus_proj.add_argument(
        "--max-msgs",
        type=int,
        default=10000,
        help="Maximum messages per subject",
    )
    bus_proj.set_defaults(func=_cmd_init_service, template="bus_project")

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
