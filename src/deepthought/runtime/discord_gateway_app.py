from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any

from ..services.discord_gateway_service import DiscordGatewayService

logger = logging.getLogger(__name__)


class DiscordGatewayRuntime:
    """Owns Discord and NATS lifecycle for the Discord gateway application."""

    def __init__(
        self,
        *,
        token: str,
        nats_url: str,
        gateway_factory: Callable[..., DiscordGatewayService] = DiscordGatewayService,
        discord_client_factory: Callable[[Any], Any] | None = None,
        intents_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._token = token
        self._nats_url = nats_url
        self._gateway_factory = gateway_factory
        self._discord_client_factory = discord_client_factory
        self._intents_factory = intents_factory
        self._gateway: DiscordGatewayService | None = None
        self._client: Any | None = None

    @staticmethod
    def _should_route_message(message: Any) -> bool:
        author = getattr(message, "author", None)
        return not bool(getattr(author, "bot", False))

    def _build_intents(self) -> Any:
        if self._intents_factory is not None:
            return self._intents_factory()

        import discord

        intents = discord.Intents.default()
        intents.guilds = True
        intents.messages = True
        intents.message_content = True
        return intents

    def _build_discord_client(self, intents: Any) -> Any:
        if self._discord_client_factory is not None:
            return self._discord_client_factory(intents)

        import discord

        return discord.Client(intents=intents)

    async def run(self) -> None:
        intents = self._build_intents()
        client = self._build_discord_client(intents)
        gateway = self._gateway_factory(discord_client=client, nats_url=self._nats_url)

        self._client = client
        self._gateway = gateway

        @client.event
        async def on_message(message: Any) -> None:
            if not self._should_route_message(message):
                return
            await gateway.handle_discord_message(message)
            content = str(getattr(message, "content", "")).lower()
            if content.startswith("correction:") or content.startswith("fix:"):
                await gateway.handle_discord_correction(message)

        @client.event
        async def on_message_edit(before: Any, after: Any) -> None:
            await gateway.handle_discord_message_edit(before, after)

        @client.event
        async def on_reaction_add(reaction: Any, user: Any) -> None:
            if bool(getattr(user, "bot", False)):
                return
            await gateway.handle_discord_reaction(reaction, user)

        try:
            started = await gateway.start()
            if not started:
                raise RuntimeError("Failed to start DiscordGatewayService")
            await client.start(self._token)
        finally:
            if getattr(client, "is_closed", lambda: True)() is False:
                await client.close()
            await gateway.stop()


async def run_discord_gateway(*, token: str, nats_url: str) -> None:
    runtime = DiscordGatewayRuntime(token=token, nats_url=nats_url)
    await runtime.run()


def main(*, token: str, nats_url: str) -> int:
    asyncio.run(run_discord_gateway(token=token, nats_url=nats_url))
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the canonical Discord gateway runtime."""

    parser = argparse.ArgumentParser(description="Run DeepThought Discord gateway runtime")
    parser.add_argument("--token", default=os.getenv("DISCORD_BOT_TOKEN", ""))
    parser.add_argument("--nats-url", default=os.getenv("NATS_URL", "nats://localhost:4222"))
    args = parser.parse_args(argv)

    token = str(args.token or "").strip()
    if not token:
        raise SystemExit("DISCORD_BOT_TOKEN is required (use --token or set DISCORD_BOT_TOKEN)")
    return main(token=token, nats_url=args.nats_url)


if __name__ == "__main__":  # pragma: no cover - module execution helper
    raise SystemExit(cli_main())
