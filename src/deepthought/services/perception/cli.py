"""Command-line interface for the perception service."""

from __future__ import annotations

import argparse
import asyncio

from nats.aio.client import Client as NATS

from .config import PerceptionConfig
from .publisher import PerceptionPublisher
from .service import PerceptionService
from .service import run as run_service


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the perception service")
    parser.add_argument("--nats-url", default=PerceptionConfig().nats_url)
    parser.add_argument("--message-id", required=True)
    parser.add_argument("--user-id", required=True)
    args = parser.parse_args()

    nc = NATS()
    await nc.connect(args.nats_url)
    js = nc.jetstream()

    publisher = PerceptionPublisher(nc, js)
    service = PerceptionService(publisher)
    await run_service(
        message_id=args.message_id,
        user_id=args.user_id,
        service=service,
    )
    await nc.drain()


def main() -> None:
    """Synchronous entry point for ``python -m`` execution."""

    asyncio.run(_main())


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
