from __future__ import annotations

from . import main as cli_main


def main(argv: list[str] | None = None) -> int:
    return cli_main(["finetune", *(argv or [])])


if __name__ == "__main__":
    raise SystemExit(main())
