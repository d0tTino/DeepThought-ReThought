from __future__ import annotations

import os

from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext

from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from ..motivate import Ledger, RewardManager


class RewardManagerService:
    """Service wrapper for :class:`RewardManager`."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        subscriber = Subscriber(nats_client, js_context)
        ledger = Ledger(nats_client, js_context)
        publisher = Publisher(nats_client, js_context)
        token = os.getenv("DISCORD_TOKEN", "")
        self._manager = RewardManager(subscriber, ledger, publisher, token)

    async def start(self, durable_name: str = "reward_manager_service") -> bool:
        return await self._manager.start_listening(durable_name=durable_name)

    async def stop(self) -> None:
        await self._manager.stop_listening()

    async def __aenter__(self) -> "RewardManagerService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
