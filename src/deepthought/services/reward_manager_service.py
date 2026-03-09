from __future__ import annotations

from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext

from ..config import load_discord_bot_token
from ..motivate import Ledger, RewardManager
from .base import BaseService


class RewardManagerService(BaseService):
    """Service wrapper for :class:`RewardManager`."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        *,
        connect_retries: int = 1,
    ) -> None:
        super().__init__(nats_client, js_context, connect_retries=connect_retries)
        ledger = Ledger(nats_client, js_context)
        token = load_discord_bot_token()
        self._manager = RewardManager(self._subscriber, ledger, self._publisher, token)

    async def start(self, durable_name: str = "reward_manager_service") -> bool:
        return await self._manager.start_listening(durable_name=durable_name)

    async def stop(self) -> None:
        await self._manager.stop_listening()
        await super().stop()

    async def __aenter__(self) -> "RewardManagerService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
