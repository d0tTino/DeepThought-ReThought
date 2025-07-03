from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def init_service(name: str) -> None:
    dest = Path("src/deepthought/services") / name
    if dest.exists():
        raise SystemExit(f"Service '{name}' already exists")
    template = Path(__file__).resolve().parents[3] / "templates" / "service"
    shutil.copytree(template, dest)
    class_name = _to_camel(name) + "Service"
    for path in dest.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        text = text.replace("TemplateService", class_name)
        text = text.replace("template_service", name)
        path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dtrt")
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init")
    init_sub = init_p.add_subparsers(dest="target")

    svc_p = init_sub.add_parser("service")
    svc_p.add_argument("name")

    args = parser.parse_args(argv)

    if args.command == "init" and args.target == "service":
        init_service(args.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
