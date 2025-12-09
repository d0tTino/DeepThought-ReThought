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
    for path in dest.rglob("*.*"):
        if path.suffix not in {".py", ".go", ".ts", ".json", ".mod"}:
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace("TemplateService", class_name)
        text = text.replace("template_service", name)
        text = text.replace("template-service", name.replace("_", "-"))
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
    language: str = "python",
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
            service_name=name,
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
            template_name=("bus_service" if language == "python" else f"bus_service_{language}"),
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
            language=getattr(args, "language", "python"),
        )
    else:
        template = args.template
        if args.template == "bus_service" and args.name == "reward":
            template = "reward_service"
        init_service(
            args.name,
            template_name=(
                template if getattr(args, "language", "python") == "python" else f"{template}_{args.language}"
            ),
            stream_name=getattr(args, "stream_name", None),
            tls_cert=getattr(args, "tls_cert", None),
            tls_key=getattr(args, "tls_key", None),
            tls_ca=getattr(args, "tls_ca", None),
            js_storage=getattr(args, "js_storage", None),
            max_msgs=getattr(args, "max_msgs", None),
        )


def _cmd_finetune(args: argparse.Namespace) -> int:
    training = import_module("deepthought.train")
    cfg = training.TrainingConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        model_loader=args.model_loader,
        dataset_loader=args.dataset_loader,
        bits=args.bits,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        pack_sequences=args.pack_sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(args.lora_target_modules),
        use_nf4=args.use_nf4,
        use_double_quant=args.use_double_quant,
        compute_dtype=args.compute_dtype,
    )

    if args.estimate_vram or args.estimate_only:
        model, _ = training.load_model(
            cfg.model_path,
            cfg.bits,
            loader=cfg.model_loader,
            use_nf4=cfg.use_nf4,
            use_double_quant=cfg.use_double_quant,
            compute_dtype=cfg.compute_dtype,
        )
        vram = training.estimate_vram(
            model,
            batch_size=cfg.batch_size,
            seq_length=cfg.max_seq_length,
            gradient_accumulation_steps=8,
            bits=cfg.bits,
        )
        print(f"Estimated VRAM requirement: {vram:.2f} GB")
        if args.estimate_only:
            return 0

    return training.run_training(cfg)


def _cmd_orchestrate(args: argparse.Namespace) -> int:
    from .. import orchestrator

    asyncio.run(orchestrator.run(args.config))
    return 0


def _cmd_perception_run(args: argparse.Namespace) -> int:
    from nats.aio.client import Client as NATS

    from ..services.perception.publisher import PerceptionPublisher
    from ..services.perception.service import PerceptionService

    async def _main() -> None:
        nc = NATS()
        await nc.connect(args.nats_url)
        js = nc.jetstream()
        publisher = PerceptionPublisher(nc, js)
        service = PerceptionService(publisher)
        await service.run(message_id=args.message_id, user_id=args.user_id)
        await nc.drain()

    asyncio.run(_main())
    return 0


def _cmd_perception_delete_user(args: argparse.Namespace) -> int:
    from ..services.perception.delete_user_data import delete_user_data

    delete_user_data(args.user_id, nats_url=args.nats_url)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    from ..services.perception.config import PerceptionConfig

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
    finetune_p.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    finetune_p.add_argument("--lora-alpha", type=int, default=64, help="LoRA scaling")
    finetune_p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="Dropout probability for LoRA layers",
    )
    finetune_p.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="List of module names to wrap with LoRA adapters",
    )
    finetune_p.add_argument(
        "--use-nf4",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable NF4 quantization (disable to use FP4)",
    )
    finetune_p.add_argument(
        "--use-double-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable nested quantization for 4-bit weights",
    )
    finetune_p.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Computation dtype for 4-bit layers",
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

    perception_p = sub.add_parser("perception", description="Perception utilities")
    perception_sub = perception_p.add_subparsers(dest="perception_cmd")
    perception_run = perception_sub.add_parser("run", description="Run perception service")
    perception_run.add_argument("--nats-url", default=PerceptionConfig().nats_url)
    perception_run.add_argument("--message-id", required=True)
    perception_run.add_argument("--user-id", required=True)
    perception_run.set_defaults(func=_cmd_perception_run)

    perception_delete = perception_sub.add_parser("delete-user", description="Delete cached perception data for a user")
    perception_delete.add_argument("--user-id", required=True)
    perception_delete.add_argument("--nats-url", default=PerceptionConfig().nats_url)
    perception_delete.set_defaults(func=_cmd_perception_delete_user)

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
    bus_svc.add_argument(
        "--language",
        choices=["python", "go", "ts"],
        default="python",
        help="Service language",
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
    bus_proj.add_argument(
        "--language",
        choices=["python", "go", "ts"],
        default="python",
        help="Service language",
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
