from __future__ import annotations

import sys

from . import main as cli_main


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args or args[0] != "bus":
        args.insert(0, "bus")
    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
