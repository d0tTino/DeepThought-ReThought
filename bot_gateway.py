"""Entrypoint for running the Discord gateway."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from contextlib import suppress
from typing import Any, Dict

from nats.aio.client import Client as NATS

from src.deepthought.config import load_config_from_env
from src.deepthought.eda.publisher import Publisher
from src.deepthought.eda.subscriber import Subscriber
from src.deepthought.edge.discord_gateway import DiscordGateway


async def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

    discord_token = os.getenv("DISCORD_BOT_TOKEN")
    if not discord_token:
        raise RuntimeError("DISCORD_BOT_TOKEN environment variable must be set.")

    config = load_config_from_env()
    nats_client = NATS()
    connect_kwargs: Dict[str, Any] = {"servers": config.nats_url}
    creds_file = os.getenv("NATS_CREDS_FILE")
    if creds_file:
        connect_kwargs["user_credentials"] = creds_file
    await nats_client.connect(**connect_kwargs)
    js = nats_client.jetstream()

    publisher = Publisher(nats_client, js)
    subscriber = Subscriber(nats_client, js)
    gateway = DiscordGateway(publisher=publisher, subscriber=subscriber)

    stop_event = asyncio.Event()

    def _handle_stop(*_: object) -> None:
        logging.info("Received shutdown signal")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handle_stop)

    runner = asyncio.create_task(gateway.run(discord_token))

    await stop_event.wait()
    runner.cancel()
    with suppress(asyncio.CancelledError):
        await runner

    await gateway.stop()
    await nats_client.drain()


if __name__ == "__main__":
    asyncio.run(_main())
