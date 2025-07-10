from __future__ import annotations

import argparse
import shutil
from importlib import import_module, resources
from pathlib import Path


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
) -> None:
    dest = Path("src/deepthought/services") / name
    if dest.exists():
        raise SystemExit(f"Service '{name}' already exists")

    template = None
    try:
        templ_res = resources.files("deepthought.templates").joinpath(template_name)
        with resources.as_file(templ_res) as path:
            if path.exists():
                template = Path(path)
    except ModuleNotFoundError:
        template = None
    if template is None or not template.exists():
        # templates live under ``templates/<template_name>`` during development
        candidate = Path(__file__).resolve().parents[3] / "templates" / template_name
        if candidate.exists():
            template = candidate
        else:
            # fallback to package data when installed from a wheel
            template = Path(__file__).resolve().parents[2] / "tools" / "template_service"

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
        if stream_name:
            text = text.replace("deepthought_events", stream_name)
        if tls_cert is not None:
            text = text.replace("NATS_TLS_CERT=", f"NATS_TLS_CERT={tls_cert}")
        if tls_key is not None:
            text = text.replace("NATS_TLS_KEY=", f"NATS_TLS_KEY={tls_key}")
        if tls_ca is not None:
            text = text.replace("NATS_TLS_CA=", f"NATS_TLS_CA={tls_ca}")
        env_file.write_text(text, encoding="utf-8")

    docker_file = dest / "Dockerfile"
    if docker_file.exists():
        text = docker_file.read_text(encoding="utf-8")
        if tls_cert is not None:
            text = text.replace("NATS_TLS_CERT=", f"NATS_TLS_CERT={tls_cert}")
        if tls_key is not None:
            text = text.replace("NATS_TLS_KEY=", f"NATS_TLS_KEY={tls_key}")
        if tls_ca is not None:
            text = text.replace("NATS_TLS_CA=", f"NATS_TLS_CA={tls_ca}")
        docker_file.write_text(text, encoding="utf-8")


def _cmd_init_service(args: argparse.Namespace) -> None:
    init_service(
        args.name,
        template_name=args.template,
        stream_name=getattr(args, "stream_name", None),
        tls_cert=getattr(args, "tls_cert", None),
        tls_key=getattr(args, "tls_key", None),
        tls_ca=getattr(args, "tls_ca", None),
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
        "--max-seq-length",
        str(args.max_seq_length),
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
    bus_svc.set_defaults(func=_cmd_init_service, template="bus_service")

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
